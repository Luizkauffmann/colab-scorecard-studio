"""Logistic regression (woe / bins / raw) and the Siddiqi scorecard."""

import math

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import StudioConfig, make_split, run_intake, screen_variables
from scorecard_studio.app import BinningSession
from scorecard_studio.datasets import CREDIT_RISK_EXCLUDE, CREDIT_RISK_PLAUSIBILITY
from scorecard_studio.model import ModelError, fit_logistic
from scorecard_studio.scaling import ScalingParams
from scorecard_studio.scorecard import build_scorecard, load_scorecard, performance

from conftest import make_credit_risk_like

PLAUS = {"rules": {k: list(v) for k, v in CREDIT_RISK_PLAUSIBILITY.items()}, "action": "special", "code": -99999}


@pytest.fixture(scope="module")
def prepared():
    raw = make_credit_risk_like(n=6000, seed=11)
    raw["noise"] = np.random.default_rng(0).normal(size=len(raw))
    cfg = StudioConfig(target="loan_status", exclude=CREDIT_RISK_EXCLUDE, plausibility=CREDIT_RISK_PLAUSIBILITY)
    it = run_intake(cfg, df=raw)
    sample = make_split(it.df, "loan_status", seed=1)
    train = it.df[sample == "train"]
    screen = screen_variables(train, "loan_status", iv_min=0.0, iv_max=50, fit_params=cfg.fit_params,
                              exclude=cfg.exclude, special_codes=it.special_codes, id_col=it.id_col)
    s = BinningSession(train, "loan_status", full=it.df, sample=sample, id_col=it.id_col, screen=screen,
                       special_codes=it.special_codes, fit_params=cfg.fit_params)
    names = ["person_income", "loan_percent_income", "person_home_ownership", "person_emp_length",
             "loan_amnt", "cb_person_default_on_file", "noise"]
    for v in names:
        s.set_included(v, True)
    # the impossible employment lengths (2 rows) form a Special bin with no events:
    # combine Special with Missing, as a modeler would
    s.set_fixed("person_emp_length", special_to=0)
    model_df = s.build_output(names)
    bundle = s.engine.build_scoring_bundle(names)
    return raw, it, sample, s, model_df, bundle


def test_siddiqi_scaling_constants():
    p = ScalingParams(pdo=20, base_score=600, base_odds=50)
    assert p.factor == pytest.approx(28.8539, abs=1e-4)
    assert p.offset == pytest.approx(487.1229, abs=1e-4)
    assert p.score_from_pd(1 / 51) == pytest.approx(600)         # odds 50:1 -> 600
    assert p.score_from_pd(1 / 101) == pytest.approx(620, abs=0.1)  # double the odds -> +PDO


def test_woe_model_fits_on_train_only(prepared):
    raw, it, sample, s, md, bundle = prepared
    reps = {v: "woe" for v in ["person_income", "loan_percent_income", "person_home_ownership"]}
    m = fit_logistic(md, "loan_status", reps, bundle)
    assert all(b > 0 for b in m.coef.values())
    flipped = md.copy()
    flipped.loc[flipped["sample"] == "test", "loan_status"] = 1 - flipped.loc[flipped["sample"] == "test", "loan_status"]
    m2 = fit_logistic(flipped, "loan_status", reps, bundle)
    assert m2.coef == pytest.approx(m.coef) and m2.intercept == pytest.approx(m.intercept)
    perf = m.performance.set_index("sample")
    assert set(perf.index) == {"train", "test"}
    assert perf.loc["train", "Gini"] == pytest.approx(2 * perf.loc["train", "AUC"] - 1)


def test_backward_selection_drops_noise(prepared):
    raw, it, sample, s, md, bundle = prepared
    reps = {v: "woe" for v in ["person_income", "loan_percent_income", "person_home_ownership", "noise"]}
    m = fit_logistic(md, "loan_status", reps, bundle, selection="backward")
    assert "noise" not in m.representations and any("noise" in x for x in m.selection_log)
    kept = fit_logistic(md, "loan_status", reps, bundle, selection="backward", keep=["noise"])
    assert "noise" in kept.representations


def test_bins_representation_uses_most_populous_reference(prepared):
    raw, it, sample, s, md, bundle = prepared
    m = fit_logistic(md, "loan_status", {"person_home_ownership": "bins", "person_income": "woe"}, bundle)
    ref = m.reference["person_home_ownership"]
    train = md[md["sample"] == "train"]
    labels = train["opt_person_home_ownership"].value_counts()
    art = bundle.artifacts["person_home_ownership"]
    ref_label = next(b["label"] for b in art.regular if b["group"] == ref)
    assert labels.idxmax() == ref_label
    assert m.bin_coef("person_home_ownership", ref) == 0.0
    # missing values present in Train -> the Missing bin gets its own dummy (a missing flag)
    m2 = fit_logistic(md, "loan_status", {"person_emp_length": "bins"}, bundle)
    assert 0 in m2.bin_coefs["person_emp_length"]


def test_raw_representation_rules(prepared):
    raw, it, sample, s, md, bundle = prepared
    with pytest.raises(ModelError, match="missing values"):
        fit_logistic(md, "loan_status", {"person_emp_length": "raw"}, bundle)
    m = fit_logistic(md, "loan_status", {"loan_amnt": "raw", "person_income": "woe"}, bundle)
    assert "loan_amnt" in m.coef


def test_raw_rejects_special_codes(prepared):
    raw, it, sample, s, md, bundle = prepared
    s.set_included("person_age", True)
    md2 = s.build_output(["person_age", "person_income"])
    b2 = s.engine.build_scoring_bundle(["person_age", "person_income"])
    with pytest.raises(ModelError, match="special codes"):
        fit_logistic(md2, "loan_status", {"person_age": "raw", "person_income": "woe"}, b2)
    s.set_included("person_age", False)


@pytest.mark.parametrize("base_points", ["spread", "separate"])
def test_unrounded_points_reproduce_the_model_exactly(prepared, base_points):
    raw, it, sample, s, md, bundle = prepared
    reps = {"person_income": "woe", "person_home_ownership": "bins", "person_emp_length": "bins",
            "loan_amnt": "raw", "loan_percent_income": "woe"}
    m = fit_logistic(md, "loan_status", reps, bundle)
    sc = ScalingParams(20, 600, 50)
    card = build_scorecard(m, bundle, sc, base_points=base_points, round_points=False, plausibility=PLAUS)
    scored = card.score(raw)
    expected = sc.offset - sc.factor * m.predict_logit(md, bundle)
    np.testing.assert_allclose(scored["score"].to_numpy(), expected, atol=1e-8)
    np.testing.assert_allclose(scored["pd"].to_numpy(), m.predict_proba(md, bundle), atol=1e-10)
    pts = scored[[c for c in scored if c.startswith("pts_")]].sum(axis=1)
    base = card.base_points if base_points == "separate" else 0
    np.testing.assert_allclose(pts + base, scored["score"])


def test_rounded_scorecard_and_rescoring(prepared, tmp_path):
    raw, it, sample, s, md, bundle = prepared
    reps = {"person_income": "woe", "person_home_ownership": "bins", "loan_percent_income": "woe"}
    m = fit_logistic(md, "loan_status", reps, bundle)
    card = build_scorecard(m, bundle, ScalingParams(20, 600, 50), plausibility=PLAUS)
    t = card.table()
    assert (t["Points"].dropna() % 1 == 0).all()
    scored = card.score(raw, keep=["loan_status"])
    assert (scored["score"] % 1 == 0).all()
    # rescoring the raw table gives the same as the processed one (plausibility is idempotent)
    np.testing.assert_array_equal(card.score(it.df)["score"], scored["score"])
    path = card.save(str(tmp_path / "scorecard.json"))
    again = load_scorecard(path).score(raw)
    np.testing.assert_array_equal(again["score"], scored["score"])
    perf = performance(raw["loan_status"], scored["score"], sample)
    assert perf.set_index("sample").loc["test", "Gini"] > 0.3


def test_manual_points_override(prepared):
    raw, it, sample, s, md, bundle = prepared
    m = fit_logistic(md, "loan_status", {"person_home_ownership": "woe", "person_income": "woe"}, bundle)
    card = build_scorecard(m, bundle, plausibility=PLAUS)
    before = card.score(raw)["score"]
    row = card.points[(card.points["Variable"] == "person_home_ownership") & (card.points["Kind"] == "regular")].iloc[0]
    card.set_points("person_home_ownership", int(row["Group"]), row["Points"] + 10)
    after = card.score(raw)["score"]
    groups = bundle.artifacts["person_home_ownership"].transform_series(it.df["person_home_ownership"])["group"]
    hit = groups == int(row["Group"])
    np.testing.assert_allclose((after - before)[hit], 10)
    np.testing.assert_allclose((after - before)[~hit], 0)
    assert card.table()["Manual"].sum() == 1
    card.reset_points()
    np.testing.assert_array_equal(card.score(raw)["score"], before)


def test_merged_missing_scores_as_its_bin(prepared):
    raw, it, sample, s, md, bundle = prepared
    s.set_fixed("person_emp_length", missing_to=1)
    md2 = s.build_output(["person_emp_length", "person_income"])
    b2 = s.engine.build_scoring_bundle(["person_emp_length", "person_income"])
    m = fit_logistic(md2, "loan_status", {"person_emp_length": "bins", "person_income": "woe"}, b2)
    assert 0 not in m.bin_coefs["person_emp_length"]           # no separate Missing dummy
    card = build_scorecard(m, b2, plausibility=PLAUS)
    rows = card.points[card.points["Variable"] == "person_emp_length"]
    assert not rows.duplicated(["Group"]).any() and 0 not in set(rows["Group"])
    scored = card.score(raw)
    miss = raw["person_emp_length"].isna().to_numpy()
    g1 = float(rows.loc[rows["Group"] == 1, "Points"].iloc[0])
    assert (scored.loc[miss, "pts_person_emp_length"] == g1).all()
    s.set_fixed("person_emp_length", missing_to=None, special_to=0)


def test_bins_with_one_class_get_a_clear_error(prepared):
    raw, it, sample, s, md, bundle = prepared
    s.set_fixed("person_emp_length", special_to=None)          # 2-row Special bin, no events
    md2 = s.build_output(["person_emp_length"])
    b2 = s.engine.build_scoring_bundle(["person_emp_length"])
    with pytest.raises(ModelError, match=r"person_emp_length\[Special\] \(\d+ rows\)"):
        fit_logistic(md2, "loan_status", {"person_emp_length": "bins"}, b2)
    fit_logistic(md2, "loan_status", {"person_emp_length": "woe"}, b2)      # WOE is smoothed: fine
    s.set_fixed("person_emp_length", special_to=0)
