"""Issue #3: missing values and special codes get real, explicit bins."""

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import MISSING_GROUP, SPECIAL_GROUP, BinningEngine


def test_missing_bin_counts_and_woe_reflect_training_data(fitted_engine, demo_df):
    r = fitted_engine.get_result("MONTHLY_INCOME")
    miss = r.bin_by_group(MISSING_GROUP)
    is_na = demo_df["MONTHLY_INCOME"].isna()
    assert miss.count == int(is_na.sum()) > 0
    assert miss.event_count == int(demo_df.loc[is_na, "DEFAULT_12M"].sum())
    # In the demo data missing income is riskier than average -> positive WOE,
    # not the neutral 0 the old engine assigned.
    assert miss.woe > 0.1
    out = fitted_engine.transform(pd.DataFrame({"MONTHLY_INCOME": [np.nan, None]}),
                                  ["MONTHLY_INCOME"])
    assert (out["opt_MONTHLY_INCOME_group"] == MISSING_GROUP).all()
    assert out["opt_MONTHLY_INCOME_woe"].tolist() == [miss.woe, miss.woe]


def test_rows_are_never_dropped(fitted_engine, demo_df):
    for v, r in fitted_engine.results.items():
        assert sum(b.count for b in r.bins) == len(demo_df), v


def test_special_codes_are_pooled_and_excluded_from_cutoffs(fitted_engine, demo_df):
    r = fitted_engine.get_result("MONTHS_SINCE_LAST_DELINQ")
    sp = r.bin_by_group(SPECIAL_GROUP)
    is_sp = demo_df["MONTHS_SINCE_LAST_DELINQ"].isin([-999, -998])
    assert sp.count == int(is_sp.sum())
    assert all(c > 0 for c in r.cutoffs), "special codes must not shape the cutoffs"
    art = fitted_engine.get_scoring_artifact("MONTHS_SINCE_LAST_DELINQ")
    assert art.score_value(-999)["group"] == SPECIAL_GROUP
    assert art.score_value(-998.0)["group"] == SPECIAL_GROUP
    assert art.score_value(-997)["group"] == 1  # not a special code -> lowest bin


def test_categorical_special_codes():
    rng = np.random.default_rng(3)
    n = 3000
    cat = rng.choice(["A", "B", "C", "UNKNOWN"], n, p=[0.4, 0.3, 0.2, 0.1])
    y = (rng.random(n) < np.where(cat == "UNKNOWN", 0.3, 0.1)).astype(int)
    df = pd.DataFrame({"c": cat, "y": y})
    eng = BinningEngine(df, "y", special_codes={"c": ["UNKNOWN"]})
    r = eng.fit("c")
    assert "UNKNOWN" not in {x for g in r.cat_groups for x in g}
    assert r.bin_by_group(SPECIAL_GROUP).count == int((cat == "UNKNOWN").sum())


def test_no_missing_in_training_is_flagged_and_neutral(fitted_engine):
    r = fitted_engine.get_result("AGE")
    miss = r.bin_by_group(MISSING_GROUP)
    assert miss.count == 0 and miss.woe == 0.0
    assert any("No missing values" in w for w in r.warnings)


def test_unseen_category_scores_as_missing(fitted_engine):
    art = fitted_engine.get_scoring_artifact("HOME_OWNERSHIP")
    assert art.score_value("Houseboat")["group"] == MISSING_GROUP
    out = art.transform_series(pd.Series(["Houseboat", None, "Rent"]))
    assert out["group"][0] == MISSING_GROUP and out["group"][1] == MISSING_GROUP
    assert out["group"][2] > 0


def test_pandas_na_and_nullable_dtypes(fitted_engine):
    art = fitted_engine.get_scoring_artifact("NUM_INQUIRIES_6M")
    s = pd.Series([1, pd.NA, 7], dtype="Int64")
    out = art.transform_series(s)
    assert out["group"][1] == MISSING_GROUP
    assert art.score_value(pd.NA)["group"] == MISSING_GROUP
    assert [art.score_value(v)["group"] for v in (1, 7)] == [out["group"][0], out["group"][2]]


def test_zero_event_bin_is_smoothed_and_flagged():
    df = pd.DataFrame({"x": np.arange(200, dtype=float),
                       "y": [0] * 100 + [0, 1] * 50})
    eng = BinningEngine(df, "y")
    eng.fit("x", monotonic="none")
    r = eng.adjust_cutoffs("x", [99.5])
    first = r.regular_bins[0]
    assert first.event_count == 0 and np.isfinite(first.woe)
    assert any("no events" in w for w in r.warnings)
