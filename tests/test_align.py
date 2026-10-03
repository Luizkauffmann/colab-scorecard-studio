"""Scorecard alignment app: settings, cutoff modes, overrides, final table,
exported code, persistence and routes."""

import json
import os
import sqlite3
import urllib.request

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import StudioConfig, make_split, run_intake, screen_variables
from scorecard_studio.align import AlignError, AlignSession, launch_alignment
from scorecard_studio.align.server import create_align_app
from scorecard_studio.app import BinningSession
from scorecard_studio.datasets import CREDIT_RISK_EXCLUDE, CREDIT_RISK_PLAUSIBILITY
from scorecard_studio.model import fit_logistic
from scorecard_studio.scaling import ScalingParams
from scorecard_studio.scorecard import build_scorecard, load_scorecard
from scorecard_studio.store import RunStore

from conftest import make_credit_risk_like


@pytest.fixture(scope="module")
def built():
    raw = make_credit_risk_like(n=5000, seed=31)
    cfg = StudioConfig(target="loan_status", exclude=CREDIT_RISK_EXCLUDE, plausibility=CREDIT_RISK_PLAUSIBILITY)
    it = run_intake(cfg, df=raw)
    sample = make_split(it.df, "loan_status", seed=1)
    train = it.df[sample == "train"]
    screen = screen_variables(train, "loan_status", iv_min=0.05, iv_max=50, fit_params=cfg.fit_params,
                              exclude=cfg.exclude, special_codes=it.special_codes, id_col=it.id_col)
    s = BinningSession(train, "loan_status", full=it.df, sample=sample, id_col=it.id_col, screen=screen,
                       special_codes=it.special_codes, fit_params=cfg.fit_params)
    md = s.build_output()
    bundle = s.engine.build_scoring_bundle(s.included)
    reps = {v: "woe" for v in s.included}
    reps[s.included[0]] = "bins"
    model = fit_logistic(md, "loan_status", reps, bundle)
    card = build_scorecard(model, bundle, ScalingParams(20, 600, 50), plausibility=cfg.plausibility_spec())
    return card, it, sample


def _session(built, tmp_path=None, **kw):
    card, it, sample = built
    return AlignSession(card, it.df, "loan_status", sample=sample, id_col=it.id_col,
                        save_dir=None if tmp_path is None else str(tmp_path / "08_alignment"),
                        cost_bad=1000, benefit_good=100, **kw)


def test_defaults_and_strategy(built):
    a = _session(built)
    assert a.settings["strategy_sample"] == "test"            # no OOT: Test, not in-sample Train
    st = a.strategy()
    assert st["mode"] == "profit" and st["cutoff"] == st["suggested"]
    test = (built[2] == "test").to_numpy()
    assert st["kpis"]["rows"] == test.sum()
    assert st["discrimination"]["AUC"] > 0.5                   # higher score = better: not inverted
    np.testing.assert_allclose(a.score, built[0].score(built[1].df)["score"])


def test_cutoff_modes(built):
    a = _session(built)
    a.set_settings(mode="approval", value=0.6)
    assert a.strategy()["kpis"]["approval_rate"] >= 0.6
    a.set_settings(mode="bad_rate", value=0.1)
    assert a.strategy()["kpis"]["bad_rate_approved"] <= 0.1
    a.set_settings(mode="manual", value=555)
    assert a.strategy()["cutoff"] == 555
    with pytest.raises(AlignError):
        a.set_settings(mode="approval", value=2)
    with pytest.raises(AlignError):
        a.set_settings(mode="nope")


def test_override_changes_scores_and_is_flagged(built):
    a = _session(built)
    rows = a.card.points[a.card.points["Kind"] == "regular"]
    r = rows.iloc[0]
    before = a.score.copy()
    a.set_points(r["Variable"], int(r["Group"]), r["Points"] + 25, "policy")
    hit = a.groups[r["Variable"]] == int(r["Group"])
    np.testing.assert_allclose((a.score - before)[hit], 25)
    np.testing.assert_allclose((a.score - before)[~hit], 0)
    sc = a.scorecard_rows()
    assert sc["n_manual"] == 1
    a.set_settings(pdo=40)
    assert a.scorecard_rows()["n_stale"] == 1
    a.reset_points()
    assert a.scorecard_rows()["n_manual"] == 0


def test_order_flag(built):
    a = _session(built)
    woe_vars = [v for v, r in a.card.representations.items() if r == "woe"]
    v = woe_vars[0]
    rows = a.card.points[(a.card.points["Variable"] == v) & (a.card.points["Kind"] == "regular")].sort_values("WOE")
    riskiest = rows.iloc[-1]
    a.set_points(v, int(riskiest["Group"]), rows["Points"].max() + 50, "test")
    assert a.scorecard_rows()["n_order_flags"] >= 1


def test_good_class_one_gives_the_same_scorecard(built):
    """A model fit on a target coded 1 = good, read with good_class=1, gives
    the same scores (up to rounding) as the 1 = bad model with good_class=0."""
    card, it, sample = built
    raw = make_credit_risk_like(n=5000, seed=31)
    cfg = StudioConfig(target="loan_status", exclude=CREDIT_RISK_EXCLUDE, plausibility=CREDIT_RISK_PLAUSIBILITY)
    train = it.df[sample == "train"]
    screen = screen_variables(train, "loan_status", iv_min=0.05, iv_max=50, fit_params=cfg.fit_params,
                              exclude=cfg.exclude, special_codes=it.special_codes, id_col=it.id_col)
    s = BinningSession(train, "loan_status", full=it.df, sample=sample, id_col=it.id_col, screen=screen,
                       special_codes=it.special_codes, fit_params=cfg.fit_params)
    md = s.build_output()
    bundle = s.engine.build_scoring_bundle(s.included)
    reps = {v: "woe" for v in s.included}
    bad_model = fit_logistic(md, "loan_status", reps, bundle)
    good_model = fit_logistic(md.assign(paid=1 - md["loan_status"]), "paid", reps, bundle)
    c0 = build_scorecard(bad_model, bundle, plausibility=cfg.plausibility_spec())
    c1 = build_scorecard(good_model, bundle, plausibility=cfg.plausibility_spec(), good_class=1)
    s0, s1 = c0.score(raw)["score"].to_numpy(), c1.score(raw)["score"].to_numpy()
    assert np.abs(s0 - s1).max() <= len(reps)                  # at most 1 point of rounding per variable
    a1 = AlignSession(c1, it.df.assign(paid=1 - it.df["loan_status"]), "paid", sample=sample,
                      cost_bad=1000, benefit_good=100)
    a0 = AlignSession(c0, it.df, "loan_status", sample=sample, cost_bad=1000, benefit_good=100)
    assert a1.strategy()["discrimination"]["AUC"] == pytest.approx(a0.strategy()["discrimination"]["AUC"], abs=0.01)
    assert a1.strategy()["discrimination"]["AUC"] > 0.5


def test_stats(built):
    a = _session(built)
    s = a.stats()
    samples = [r["sample"] for r in s["per_sample"]]
    assert samples == ["train", "test", "all"]
    assert "test" in s["psi"] and sum(g["rows"] for g in s["gains"]) == (built[2] == "test").sum()
    for r in s["per_sample"]:
        assert r["Gini"] == pytest.approx(2 * r["AUC"] - 1)


def test_finalize_and_code_agree(built, tmp_path):
    a = _session(built, tmp_path)
    a.set_settings(mode="approval", value=0.65)
    r = a.card.points[a.card.points["Kind"] == "regular"].iloc[1]
    a.set_points(r["Variable"], int(r["Group"]), r["Points"] - 7, "judgment")
    info = a.finalize()
    for k in ("table", "csv", "scorecard", "points_table"):
        assert os.path.exists(info["paths"][k])
    final = pd.read_parquet(info["paths"]["table"])
    assert len(final) == len(built[1].df)
    assert {f"pts_{v}" for v in a.card.variables} <= set(final.columns)
    np.testing.assert_allclose(final[[f"pts_{v}" for v in a.card.variables]].sum(axis=1), final["score"])
    assert set(final["decision"]) <= {"approve", "decline"}
    # final scorecard reproduces the final table
    card = load_scorecard(info["paths"]["scorecard"])
    assert card.cutoff == info["cutoff"]
    rescored = card.score(built[1].df)
    np.testing.assert_allclose(rescored["score"].to_numpy(), final["score"].to_numpy())
    assert (rescored["decision"].to_numpy() == final["decision"].to_numpy()).all()
    # exported Python and SQL agree with the final table
    py_path = a.save_code("python")
    ns = {}
    exec(compile(open(py_path).read(), py_path, "exec"), ns)
    data = built[1].df
    recs = data.astype(object).where(data.notna(), None).to_dict("records")
    got = [ns["score_record"](x) for x in recs]
    np.testing.assert_allclose([g["score"] for g in got], final["score"].to_numpy())
    assert [g["decision"] for g in got] == final["decision"].tolist()
    sql_path = a.save_code("sql", "standard", "apps")
    con = sqlite3.connect(":memory:")
    data.assign(__row=np.arange(len(data))).to_sql("apps", con, index=False)
    try:
        con.execute("SELECT EXP(1.0)")
        sql = open(sql_path).read()
    except sqlite3.OperationalError:
        sql = a.card.to_sql("apps", include_pd=False)
    out = pd.read_sql_query(sql + "\nORDER BY __row", con)
    np.testing.assert_allclose(out["score"].to_numpy(), final["score"].to_numpy())
    assert out["decision"].tolist() == final["decision"].tolist()


def test_state_saved_and_resumed(built, tmp_path):
    card, it, sample = built
    store = RunStore(str(tmp_path / "runs"), run_id="run_20260101_000000")
    app = launch_alignment(card, it.df, "loan_status", sample=sample, store=store, show=False,
                           cost_bad=500, benefit_good=50)
    try:
        r = app.session.card.points[app.session.card.points["Kind"] == "regular"].iloc[0]
        app.session.set_points(r["Variable"], int(r["Group"]), r["Points"] + 3, "kept")
        app.session.set_settings(mode="approval", value=0.5, pdo=30)
        with urllib.request.urlopen(app.url + "api/state", timeout=10) as resp:
            assert json.load(resp)["state"]["pdo"] == 30
    finally:
        app.stop()
    store2 = RunStore(str(tmp_path / "runs"), run_id="run_20260101_000001")
    app2 = launch_alignment(card, it.df, "loan_status", sample=sample, store=store2, show=False)
    try:
        s = app2.session
        assert s.card.scaling.pdo == 30 and s.settings["mode"] == "approval" and s.settings["cost_bad"] == 500
        assert s.scorecard_rows()["n_manual"] == 1
    finally:
        app2.stop()


def test_routes(built, tmp_path):
    a = _session(built, tmp_path)
    c = create_align_app(a).test_client()
    assert b"Scorecard Alignment" in c.get("/").data
    assert c.get("/static/align.js").status_code == 200
    st = c.get("/api/state").get_json()
    assert st["state"]["target"] == "loan_status" and st["strategy"]["kpis"]["rows"] > 0
    r = c.post("/api/settings", json={"mode": "approval", "value": 0.5}).get_json()
    assert r["strategy"]["kpis"]["approval_rate"] >= 0.5
    sc = c.get("/api/scorecard").get_json()
    row = next(x for x in sc["rows"] if x["Kind"] == "regular")
    r = c.post("/api/points", json={"variable": row["Variable"], "group": row["Group"],
                                    "points": row["Points"] + 1, "reason": "x"}).get_json()
    assert r["scorecard"]["n_manual"] == 1
    assert c.post("/api/reset", json={}).get_json()["scorecard"]["n_manual"] == 0
    assert c.get("/api/stats?sample=train").get_json()["sample"] == "train"
    f = c.post("/api/finalize", json={}).get_json()
    assert f["final"]["rows"] == len(built[1].df) and f["preview"]["rows"]
    code = c.get("/api/code?lang=sql&dialect=spark&table=t").get_json()["code"]
    assert "FROM t AS src" in code
    assert c.get("/api/code?lang=cobol").status_code == 400
    assert c.post("/api/settings", json={"pdo": -5}).status_code == 400
    assert c.post("/api/points", json={"variable": "nope", "group": 1, "points": 1}).status_code == 400


# ── dataset and target chosen in the app ─────────────────────────────

def test_switch_dataset_and_target(built, tmp_path):
    card, it, sample = built
    rng = np.random.default_rng(7)
    other = make_credit_risk_like(n=1500, seed=77)               # raw table, same schema
    other["default_24m"] = np.where(rng.random(len(other)) < 0.8, other["loan_status"], 1 - other["loan_status"])
    other.loc[other.index[:200], "default_24m"] = np.nan          # unknown outcomes (e.g. rejects)
    a = AlignSession(card, it.df, "loan_status", sample=sample, id_col=it.id_col,
                     save_dir=str(tmp_path / "08_alignment"), datasets={"new vintage": other})
    st = a.state()
    assert st["datasets"] == ["model dataset", "new vintage"] and st["dataset"] == "model dataset"
    assert "loan_status" in st["target_options"]

    points_before = a.card.points["Points"].tolist()
    a.set_data("new vintage")                                   # target kept: loan_status exists there
    assert a.target == "loan_status" and a.samples == ["all"] and len(a.score) == len(other)
    np.testing.assert_allclose(a.score, card.score(other)["score"])     # same scorecard, raw data scored
    assert a.card.points["Points"].tolist() == points_before

    assert set(a.state()["target_options"]) >= {"loan_status", "default_24m"}
    a.set_data(target="default_24m")
    k = a.strategy()["kpis"]
    assert k["rows"] == len(other)                               # approval rate on all rows
    known = other["default_24m"].notna()
    assert a.state()["outcome_rows"] == int(known.sum())

    with pytest.raises(AlignError, match="0/1"):
        a.set_data(target="person_income")
    assert a.target == "default_24m"                             # unchanged after a refused switch
    with pytest.raises(AlignError, match="Unknown dataset"):
        a.set_data("nope")

    table = a.final_table()
    assert len(table) == len(other) and "default_24m" in table

    # the choice is saved and restored with the rest of the state
    saved = json.loads((tmp_path / "08_alignment" / "alignment_state.json").read_text())
    assert saved["data"] == {"dataset": "new vintage", "target": "default_24m"}
    b = AlignSession(card, it.df, "loan_status", sample=sample, id_col=it.id_col,
                     datasets={"new vintage": other})
    b.load(saved)
    assert (b.dataset_name, b.target) == ("new vintage", "default_24m")


def test_data_route(built, tmp_path):
    card, it, sample = built
    other = make_credit_risk_like(n=800, seed=78)
    a = AlignSession(card, it.df, "loan_status", sample=sample, id_col=it.id_col, datasets={"other": other})
    c = create_align_app(a).test_client()
    r = c.post("/api/data", json={"dataset": "other"}).get_json()
    assert r["state"]["dataset"] == "other" and r["strategy"]["kpis"]["rows"] == len(other)
    bad = c.post("/api/data", json={"target": "person_age"})
    assert bad.status_code == 400 and "0/1" in bad.get_json()["error"]
