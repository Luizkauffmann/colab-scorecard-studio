"""Engine behaviour: fitting, manual refinement, labels, divergence wiring."""

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import BinningEngine
from scorecard_studio import binning as binning_mod
from scorecard_studio.metrics import auc_score, gini_from_auc


# ── target validation ────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [[0, 1, 2], [0, 1, np.nan], [1, 1, 1]])
def test_target_must_be_binary_and_complete(bad):
    with pytest.raises(ValueError):
        BinningEngine(pd.DataFrame({"x": [1, 2, 3], "y": bad}), "y")


# ── issue #5: labels follow (lower, upper] ────────────────────────────────

def test_labels_and_boundaries_follow_lower_exclusive_upper_inclusive(tiny_df):
    eng = BinningEngine(tiny_df, "y")
    eng.fit("x", dtype="numerical", max_bins=2, min_bin_size=0.05, monotonic="none")
    r = eng.adjust_cutoffs("x", [3, 7])
    labels = [b.label for b in r.regular_bins]
    assert labels == ["<= 3", "(3, 7]", "> 7"]
    # a value equal to the cutoff belongs to the lower bin
    assert [b.count for b in r.regular_bins] == [3, 4, 3]
    art = eng.get_scoring_artifact("x")
    assert art.score_value(3)["group"] == 1
    assert art.score_value(3.0000001)["group"] == 2
    assert art.score_value(7)["group"] == 2
    assert art.score_value(1e9)["group"] == 3
    assert art.score_value(-1e9)["group"] == 1
    assert r.regular_bins[0].lower is None and r.regular_bins[-1].upper is None


def test_single_bin_label(tiny_df):
    eng = BinningEngine(tiny_df, "y")
    eng.fit("x", dtype="numerical", monotonic="none")
    r = eng.adjust_cutoffs("x", [])
    assert [b.label for b in r.regular_bins] == ["All values"]


# ── numerical fit ────────────────────────────────────────────────────────

def test_numerical_fit_monotonic_and_stats_consistent(fitted_engine, demo_df):
    r = fitted_engine.get_result("CREDIT_UTILIZATION")
    assert r.dtype == "numerical"
    assert r.is_monotonic and r.monotonic_direction == "increasing"
    assert sum(b.count for b in r.bins) == len(demo_df)
    assert sum(b.event_count for b in r.bins) == demo_df["DEFAULT_12M"].sum()
    # Gini from the bins equals row-level Gini of the WOE-transformed variable
    woe = fitted_engine.transform(demo_df, ["CREDIT_UTILIZATION"], metrics=["woe"])
    row_gini = gini_from_auc(auc_score(demo_df["DEFAULT_12M"], woe["opt_CREDIT_UTILIZATION_woe"]))
    assert r.gini == pytest.approx(row_gini, abs=1e-12)


def test_min_bin_size_is_share_of_all_rows(fitted_engine, demo_df):
    # MONTHS_SINCE_LAST_DELINQ is ~59% special codes: bins must still hold
    # >= 5% of ALL rows, not 5% of the non-special subset.
    r = fitted_engine.get_result("MONTHS_SINCE_LAST_DELINQ")
    n = len(demo_df)
    assert all(b.count / n >= 0.05 - 1e-9 for b in r.regular_bins)


def test_adjust_cutoffs_recomputes_and_validates(fitted_engine):
    eng = fitted_engine
    before = eng.get_result("AGE")
    r = eng.adjust_cutoffs("AGE", [30, 45, 30])  # duplicate is removed
    assert r.cutoffs == [30.0, 45.0]
    assert len(r.regular_bins) == 3
    with pytest.raises(ValueError):
        eng.adjust_cutoffs("AGE", [float("inf")])
    with pytest.raises(ValueError):
        eng.adjust_cutoffs("EMPLOYMENT_TYPE", [1.0])
    eng.adjust_cutoffs("AGE", before.cutoffs)  # restore for other tests


# ── categorical fit ──────────────────────────────────────────────────────

def test_categorical_fit_keeps_every_training_category(fitted_engine, demo_df):
    r = fitted_engine.get_result("EMPLOYMENT_TYPE")
    grouped = {c for g in r.cat_groups for c in g}
    assert grouped == set(demo_df["EMPLOYMENT_TYPE"].unique())  # incl. rare "Contractor"
    art = fitted_engine.get_scoring_artifact("EMPLOYMENT_TYPE")
    assert art.score_value("Contractor")["group"] > 0


def test_merge_categories(fitted_engine):
    eng = fitted_engine
    before = eng.get_result("HOME_OWNERSHIP")
    assign = {"Mortgage": 1, "Own": 1, "Rent": 2, "Other": 2}
    r = eng.merge_categories("HOME_OWNERSHIP", assign)
    assert sorted(map(sorted, r.cat_groups)) == [["Mortgage", "Own"], ["Other", "Rent"]]
    with pytest.raises(ValueError, match="without a group"):
        eng.merge_categories("HOME_OWNERSHIP", {"Mortgage": 1})
    eng.merge_categories("HOME_OWNERSHIP",
                         {c: i for i, g in enumerate(before.cat_groups, 1) for c in g})


def test_coded_numeric_categorical(fitted_engine):
    # EDUCATION_LEVEL is float (1.0..5.0 with NaN) but categorical: categories are "1".."5"
    r = fitted_engine.get_result("EDUCATION_LEVEL")
    assert r.dtype == "categorical" and r.value_type == "numeric"
    assert {c for g in r.cat_groups for c in g} == {"1", "2", "3", "4", "5"}
    art = fitted_engine.get_scoring_artifact("EDUCATION_LEVEL")
    assert art.score_value(3)["group"] == art.score_value(3.0)["group"] == art.score_value(np.int64(3))["group"]


# ── issue #2: divergence is passed to the optimiser ──────────────────────

def test_divergence_is_wired_to_optbinning(monkeypatch, demo_df):
    import optbinning

    captured = {}
    real = optbinning.OptimalBinning

    def spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(optbinning, "OptimalBinning", spy)
    eng = BinningEngine(demo_df, "DEFAULT_12M")
    r = eng.fit("DEBT_TO_INCOME", divergence="js", min_bin_n_event=25)
    assert captured["divergence"] == "js"
    assert captured["min_bin_n_event"] == 25
    assert r.divergence == "js"
    assert all(b.event_count >= 25 for b in r.regular_bins)


def test_invalid_options_raise(demo_df):
    eng = BinningEngine(demo_df, "DEFAULT_12M")
    with pytest.raises(ValueError, match="divergence"):
        eng.fit("AGE", divergence="gini")
    with pytest.raises(ValueError, match="monotonic"):
        eng.fit("AGE", monotonic="up")
    with pytest.raises(ValueError):
        eng.fit("DEFAULT_12M")
    with pytest.raises(ValueError):
        eng.fit("APPLICATION_DATE")


def test_monotonic_map_covers_ui_options():
    assert set(binning_mod.MONOTONIC_MAP) == {"none", "auto", "increasing", "decreasing"}


def test_iv_summary_sorted_and_complete(fitted_engine):
    s = fitted_engine.get_iv_summary()
    assert list(s["IV"]) == sorted(s["IV"], reverse=True)
    assert set(s["Variable"]) == set(fitted_engine.variable_types())  # ID and date are excluded
