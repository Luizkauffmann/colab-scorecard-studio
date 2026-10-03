"""The exported scoring code (standalone Python and SQL) must reproduce
Scorecard.score row for row: plausibility rules, bins (incl. Missing/Special
choices), points after manual overrides, score, pd and decision."""

import csv
import importlib.util
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import StudioConfig, make_split, run_intake, screen_variables
from scorecard_studio.app import BinningSession
from scorecard_studio.datasets import CREDIT_RISK_EXCLUDE, CREDIT_RISK_PLAUSIBILITY
from scorecard_studio.model import fit_logistic
from scorecard_studio.scaling import ScalingParams
from scorecard_studio.scorecard import build_scorecard, load_scorecard

from conftest import make_credit_risk_like

VARS = ["person_income", "loan_percent_income", "person_home_ownership", "person_emp_length",
        "loan_amnt", "person_age", "loan_intent"]


def _sqlite_has_exp():
    try:
        sqlite3.connect(":memory:").execute("SELECT EXP(1.0)")
        return True
    except sqlite3.OperationalError:
        return False


@pytest.fixture(scope="module")
def setup():
    raw = make_credit_risk_like(n=5000, seed=21)
    cfg = StudioConfig(target="loan_status", exclude=CREDIT_RISK_EXCLUDE, plausibility=CREDIT_RISK_PLAUSIBILITY)
    it = run_intake(cfg, df=raw)
    sample = make_split(it.df, "loan_status", seed=1)
    train = it.df[sample == "train"]
    screen = screen_variables(train, "loan_status", iv_min=0.0, iv_max=50, fit_params=cfg.fit_params,
                              exclude=cfg.exclude, special_codes=it.special_codes, id_col=it.id_col)
    s = BinningSession(train, "loan_status", full=it.df, sample=sample, id_col=it.id_col, screen=screen,
                       special_codes=it.special_codes, fit_params=cfg.fit_params)
    for v in VARS:
        s.set_included(v, True)
    s.set_fixed("person_emp_length", missing_to=2, special_to=0)    # Missing (+Special) into G2
    s.set_fixed("person_age", special_to=1)                         # impossible ages into G1
    md = s.build_output(VARS)
    bundle = s.engine.build_scoring_bundle(VARS)
    reps = {"person_income": "woe", "loan_percent_income": "woe", "person_home_ownership": "bins",
            "person_emp_length": "bins", "loan_amnt": "raw", "person_age": "woe", "loan_intent": "woe"}
    model = fit_logistic(md, "loan_status", reps, bundle)
    card = build_scorecard(model, bundle, ScalingParams(20, 600, 50), plausibility=cfg.plausibility_spec())
    card.set_points("person_home_ownership", 1, 777, reason="policy")      # manual override
    card.cutoff = float(np.median(card.score(raw)["score"]))
    # stress rows: missing, impossible, special code itself, unseen category, exact cutoffs
    edge = []
    base = raw.iloc[0].to_dict()
    for v in VARS:
        if v == "loan_amnt":
            continue
        edge.append({**base, v: None})
    edge += [{**base, "person_age": 144}, {**base, "person_age": -99999}, {**base, "person_emp_length": 123.0},
             {**base, "person_home_ownership": "NEVER_SEEN"}, {**base, "loan_intent": "O'Brien \"loan\""}]
    for v in ("person_income", "loan_percent_income"):
        for c in bundle.artifacts[v].cutoffs:
            edge.append({**base, v: c})
    data = pd.concat([raw, pd.DataFrame(edge)], ignore_index=True)
    data["person_age"] = data["person_age"].astype(float)
    return card, data


def _load(path):
    spec = importlib.util.spec_from_file_location("gen_scorer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_python_script_matches(setup, tmp_path):
    card, data = setup
    exp = card.score(data)
    path = tmp_path / "scorer.py"
    path.write_text(card.to_python())
    mod = _load(path)
    recs = data.astype(object).where(data.notna(), None).to_dict("records")
    got = pd.DataFrame([mod.score_record(r) for r in recs])
    for c in [c for c in exp.columns if c.startswith("pts_")] + ["score", "pd"]:
        np.testing.assert_allclose(got[c].to_numpy(float), exp[c].to_numpy(float), rtol=0, atol=1e-9, err_msg=c)
    assert (got["decision"].to_numpy() == exp["decision"].to_numpy()).all()


def test_python_script_runs_on_a_csv(setup, tmp_path):
    card, data = setup
    exp = card.score(data)
    (tmp_path / "scorer.py").write_text(card.to_python())
    data.to_csv(tmp_path / "in.csv", index=False)
    subprocess.run([sys.executable, str(tmp_path / "scorer.py"), str(tmp_path / "in.csv"), str(tmp_path / "out.csv")],
                   check=True, cwd=tmp_path)
    out = pd.read_csv(tmp_path / "out.csv")
    assert len(out) == len(data) and list(out.columns[:len(data.columns)]) == list(data.columns)
    np.testing.assert_allclose(out["score"].to_numpy(), exp["score"].to_numpy(), atol=1e-9)
    assert (out["decision"].to_numpy() == exp["decision"].to_numpy()).all()


@pytest.mark.parametrize("include_pd", [True, False])
def test_sql_matches(setup, include_pd):
    card, data = setup
    if include_pd and not _sqlite_has_exp():
        pytest.skip("this SQLite build has no EXP()")
    exp = card.score(data)
    con = sqlite3.connect(":memory:")
    data.assign(__row=np.arange(len(data))).to_sql("input_table", con, index=False)
    got = pd.read_sql_query(card.to_sql("input_table", include_pd=include_pd) + "\nORDER BY __row", con)
    for c in [c for c in exp.columns if c.startswith("pts_")] + ["score"] + (["pd"] if include_pd else []):
        np.testing.assert_allclose(got[c].to_numpy(float), exp[c].to_numpy(float), atol=1e-9, err_msg=c)
    assert (got["decision"].to_numpy() == exp["decision"].to_numpy()).all()
    assert "pd" not in got or include_pd


def test_sql_dialects_quote_names(setup):
    card, _ = setup
    assert "`pts_person_income`" in card.to_sql(dialect="spark")
    assert "`pts_person_income`" in card.to_sql(dialect="bigquery")
    assert "'O''Brien" not in card.to_sql()          # unseen value is not a rule, nothing to escape
    with pytest.raises(ValueError):
        card.to_sql(dialect="oracle")


def test_override_and_cutoff_survive_save_load(setup, tmp_path):
    card, data = setup
    p = card.save(str(tmp_path / "card.json"))
    back = load_scorecard(p)
    assert back.cutoff == card.cutoff
    row = back.points[(back.points["Variable"] == "person_home_ownership") & (back.points["Group"] == 1)].iloc[0]
    assert row["Points"] == 777 and row["Manual"] and row["Reason"] == "policy"
    pd.testing.assert_frame_equal(back.score(data), card.score(data))


def test_rescale_keeps_overrides_and_flags_them(setup):
    card, data = setup
    c = card.copy()
    before = c.points.set_index(["Variable", "Group"])["Points"]
    c.rescale(ScalingParams(40, 600, 50))
    after = c.points.set_index(["Variable", "Group"])
    assert after.loc[("person_home_ownership", 1), "Points"] == 777
    assert bool(after.loc[("person_home_ownership", 1), "Stale"])
    auto = after[~after["Manual"].astype(bool)]
    assert not np.allclose(auto["Points"], before[auto.index])           # rescaled
    c.reset_points()
    assert not c.points["Manual"].any()


def test_good_class_one_flips_direction(setup):
    card, data = setup
    c = card.copy()
    c.reset_points()
    s0 = c.score(data)["score"].to_numpy()
    c.rescale(good_class=1)
    s1 = c.score(data)["score"].to_numpy()
    # good = 1: scores rise with P(target=1); the ranking is exactly reversed (up to rounding ties)
    assert np.corrcoef(s0, s1)[0, 1] < -0.99
    pd0 = card.copy()
    assert c.scale_signature() != pd0.scale_signature()
