"""Binning app: session logic, Flask routes, persistence, output dataset."""

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import StudioConfig, make_split, run_intake, screen_variables
from scorecard_studio.app import AppError, BinningSession, find_saved_configs, launch_app
from scorecard_studio.app.server import create_app
from scorecard_studio.datasets import CREDIT_RISK_EXCLUDE, CREDIT_RISK_PLAUSIBILITY
from scorecard_studio.store import RunStore


@pytest.fixture
def setup(credit_risk_like):
    cfg = StudioConfig(target="loan_status", exclude=CREDIT_RISK_EXCLUDE,
                       plausibility=CREDIT_RISK_PLAUSIBILITY)
    it = run_intake(cfg, df=credit_risk_like)
    sample = make_split(it.df, "loan_status", seed=1)
    train = it.df[sample == "train"]
    screen = screen_variables(train, "loan_status", iv_min=0.10, iv_max=5.0,
                              fit_params=cfg.fit_params, exclude=cfg.exclude,
                              special_codes=it.special_codes, id_col=it.id_col)
    return cfg, it, sample, train, screen


def _session(setup, tmp_path=None, **kw):
    cfg, it, sample, train, screen = setup
    return BinningSession(train, "loan_status", full=it.df, sample=sample, id_col=it.id_col,
                          screen=screen, special_codes=it.special_codes, fit_params=cfg.fit_params,
                          save_dir=None if tmp_path is None else str(tmp_path / "04_binning"), **kw)


# ── starting point ───────────────────────────────────────────────────

def test_starts_from_screening_shortlist_and_fits(setup):
    s = _session(setup)
    screen = setup[4]
    assert set(s.included) == set(screen.selected)
    for v in screen.selected:
        assert s.result(v).cutoffs == screen.result(v).cutoffs
        assert s.result(v).cat_groups == screen.result(v).cat_groups
    st = {v["name"]: v for v in s.state()["variables"]}
    assert not st["loan_grade"]["addable"]                     # excluded in config
    assert "row_id" not in st and "loan_status" not in st


def test_include_rules(setup):
    s = _session(setup)
    low = next(v.name for v in s.vars.values() if v.status == "low_iv")
    s.set_included(low, True)
    assert low in s.included
    with pytest.raises(AppError, match="config cell"):
        s.set_included("loan_grade", True)
    s.set_included(low, False)
    assert low not in s.included


# ── numerical edits ──────────────────────────────────────────────────

def test_cutoffs_split_merge(setup):
    s = _session(setup)
    v = "person_income"
    r = s.set_cutoffs(v, [30000, 50000, 80000])["variable"]
    assert r["cutoffs"] == [30000, 50000, 80000] and s.vars[v].edited
    r = s.split(v, 2)["variable"]                          # (30000, 50000] at its median
    assert len(r["cutoffs"]) == 4
    mid = r["cutoffs"][1]
    assert 30000 < mid < 50000 and float(mid).is_integer()  # integer data: data-valued cut
    counts = [b["count"] for b in r["bins"] if b["kind"] == "regular"]
    assert all(c > 0 for c in counts)
    r = s.merge(v, [2, 3])["variable"]
    assert r["cutoffs"] == [30000, 50000, 80000]
    with pytest.raises(AppError, match="adjacent"):
        s.merge(v, [1, 3])
    with pytest.raises(AppError, match="fixed bin"):
        s.merge(v, [1, 0])                                  # Missing is group 0
    with pytest.raises(AppError, match="outside"):
        s.split(v, 1, at=60000)


def test_split_refuses_single_value_bin(setup):
    s = _session(setup)
    x = setup[3]["person_age"]
    v = int(x[x > 0].mode().iloc[0])
    s.set_cutoffs("person_age", [v - 1, v])                 # bin (v-1, v] holds one value
    with pytest.raises(AppError, match="single distinct value"):
        s.split("person_age", 2)


def test_fit_marks_edited_and_reset_clears(setup):
    s = _session(setup)
    r = s.fit("person_income", max_bins=3)["variable"]
    assert len([b for b in r["bins"] if b["kind"] == "regular"]) <= 3 and s.vars["person_income"].edited
    s.reset("person_income")
    assert not s.vars["person_income"].edited
    with pytest.raises(AppError, match="Binning failed"):
        s.fit("person_income", min_bin_n_event=10**6)


# ── categorical edits ────────────────────────────────────────────────

def test_categorical_merge_and_regroup(setup):
    s = _session(setup)
    v = "loan_intent"
    p = s.variable_payload(v)
    n0 = len(p["cat_groups"])
    assert {c["value"] for c in p["categories"]} == {c for g in p["cat_groups"] for c in g}
    r = s.merge(v, [1, 2])["variable"]
    assert len(r["cat_groups"]) == n0 - 1
    cats = sorted({c for g in r["cat_groups"] for c in g})
    r = s.set_categories(v, {c: (1 if i % 2 else 2) for i, c in enumerate(cats)})["variable"]
    assert len(r["cat_groups"]) == 2
    with pytest.raises(AppError, match="unassigned"):
        s.set_categories(v, {cats[0]: 1})
    with pytest.raises(AppError, match="numerical"):
        s.set_cutoffs(v, [1])


# ── persistence ──────────────────────────────────────────────────────

def test_every_edit_is_saved_and_reloads_identically(setup, tmp_path):
    s = _session(setup, tmp_path)
    path = tmp_path / "04_binning" / "binning_config.json"
    s.set_cutoffs("person_income", [25000, 45000, 90000])
    saved = json.loads(path.read_text())
    assert saved["engine"]["variables"]["person_income"]["cutoffs"] == [25000, 45000, 90000]
    s.merge("loan_intent", [1, 2])
    s.set_included("loan_intent", True)
    saved = json.loads(path.read_text())
    assert "loan_intent" in saved["included"] and set(saved["edited"]) == {"person_income", "loan_intent"}

    fresh = _session(setup)
    assert fresh.load(saved) == []
    for v in ("person_income", "loan_intent"):
        a, b = s.result(v), fresh.result(v)
        assert a.cutoffs == b.cutoffs and a.cat_groups == b.cat_groups
        assert [x.woe for x in a.bins] == [x.woe for x in b.bins]
    assert fresh.included == s.included and fresh.vars["person_income"].edited


def test_config_for_another_train_sample_is_refused(setup, tmp_path):
    s = _session(setup, tmp_path)
    saved = s.to_config()
    cfg, it, _, _, screen = setup
    other = make_split(it.df, "loan_status", seed=99)
    s2 = BinningSession(it.df[other == "train"], "loan_status", id_col=it.id_col, screen=screen,
                        special_codes=it.special_codes, fit_params=cfg.fit_params)
    with pytest.raises(AppError, match="different Train sample"):
        s2.load(saved)
    s2.load(saved, force=True)


# ── output dataset ───────────────────────────────────────────────────

def test_output_dataset_columns_and_samples(setup):
    s = _session(setup)
    s.set_included("loan_intent", True)
    out = s.build_output()
    _, it, sample, _, _ = setup
    assert len(out) == len(it.df) and set(out["sample"]) == {"train", "test"}
    for v in s.included:
        assert {v, f"opt_{v}", f"woe_{v}"} <= set(out.columns)
        assert out[f"opt_{v}"].cat.ordered and out[f"opt_{v}"].cat.categories[-1] == "Missing"
        assert out[f"woe_{v}"].notna().all()
    assert list(out.columns[:3]) == ["row_id", "sample", "loan_status"]


def test_output_applies_train_bins_to_test(setup):
    """Test rows get WOE values from the Train bins; nothing is refit."""
    s = _session(setup)
    out = s.build_output(["person_income"])
    train_woes = {round(b.woe, 12) for b in s.result("person_income").bins}
    test_woes = set(out.loc[out["sample"] == "test", "woe_person_income"].round(12))
    assert test_woes <= train_woes


def test_output_matches_exported_scorer(setup, tmp_path):
    """woe_<var> equals what the deployable scorer.py computes."""
    s = _session(setup)
    s.set_cutoffs("person_income", [25000, 45000, 90000])
    names = s.included
    out = s.build_output(names)
    code = s.engine.build_scoring_bundle(names).to_python_module()
    ns = {}
    exec(compile(code, "scorer.py", "exec"), ns)
    base = setup[1].df
    scored = ns["score_dataframe"](base[names], metrics=("woe", "label"))
    for v in names:
        np.testing.assert_allclose(out[f"woe_{v}"].to_numpy(), scored[f"opt_{v}_woe"].to_numpy(float))
        assert out[f"opt_{v}"].astype(str).tolist() == scored[f"opt_{v}_label"].tolist()


def test_save_output_writes_files(setup, tmp_path):
    s = _session(setup, tmp_path)
    info = s.save_output()
    for k in ("dataset", "bundle", "tables"):
        assert os.path.exists(info["paths"][k])
    back = pd.read_parquet(info["paths"]["dataset"])
    assert back.shape == (info["rows"], info["columns"])
    assert str(back[f"opt_{s.included[0]}"].dtype) == "category"


def test_output_name_clash(setup):
    cfg, it, sample, train, screen = setup
    full = it.df.assign(**{f"woe_{screen.selected[0]}": 1.0})
    s = BinningSession(full[sample == "train"], "loan_status", full=full, sample=sample,
                       id_col=it.id_col, screen=screen, special_codes=it.special_codes,
                       fit_params=cfg.fit_params)
    with pytest.raises(AppError, match="already exist"):
        s.build_output()


# ── HTTP routes ──────────────────────────────────────────────────────

def test_routes(setup, tmp_path):
    s = _session(setup, tmp_path)
    c = create_app(s).test_client()
    assert b"Scorecard Binning" in c.get("/").data
    assert c.get("/static/app.js").status_code == 200
    st = c.get("/api/state").get_json()
    assert st["included"] == s.included and st["rows"] == len(setup[3])
    p = c.get("/api/variable?name=person_income").get_json()["variable"]
    assert p["dtype"] == "numerical" and p["hist"]["counts"] and p["hist"]["integer"]

    r = c.post("/api/cutoffs", json={"name": "person_income", "cutoffs": [30000, 60000]}).get_json()
    assert r["variable"]["cutoffs"] == [30000, 60000] and r["state"]["last_saved"]
    r = c.post("/api/split", json={"name": "person_income", "group": 1}).get_json()
    assert len(r["variable"]["cutoffs"]) == 3
    r = c.post("/api/merge", json={"name": "person_income", "groups": [1, 2]}).get_json()
    assert r["variable"]["cutoffs"] == [30000, 60000]
    r = c.post("/api/fit", json={"name": "person_income", "max_bins": 4, "monotonic": "auto",
                                 "min_bin_size": 0.05, "min_bin_n_event": None}).get_json()
    assert r["variable"]["settings"]["max_bins"] == 4
    assert c.post("/api/reset", json={"name": "person_income"}).status_code == 200

    groups = c.get("/api/variable?name=loan_intent").get_json()["variable"]["cat_groups"]
    assign = {cat: 1 for g in groups for cat in g}
    r = c.post("/api/categories", json={"name": "loan_intent", "assignments": assign}).get_json()
    assert len(r["variable"]["cat_groups"]) == 1
    r = c.post("/api/include", json={"name": "loan_intent", "included": True}).get_json()
    assert "loan_intent" in r["state"]["included"]

    bad = c.post("/api/merge", json={"name": "person_income", "groups": [1, 3]})
    assert bad.status_code == 400 and "adjacent" in bad.get_json()["error"]
    assert c.get("/api/variable?name=nope").status_code == 400
    assert c.post("/api/include", json={"name": "loan_grade", "included": True}).status_code == 400

    r = c.post("/api/output", json={}).get_json()
    assert r["output"]["rows"] == len(setup[1].df) and os.path.exists(r["output"]["paths"]["dataset"])
    assert r["state"]["last_output"]["variables"] == s.included


def test_launch_serves_and_resumes(setup, tmp_path):
    cfg, it, sample, train, screen = setup
    kw = dict(full=it.df, sample=sample, id_col=it.id_col, screen=screen,
              special_codes=it.special_codes, fit_params=cfg.fit_params)
    store = RunStore(str(tmp_path / "runs"), run_id="run_20260101_000000")
    app = launch_app(train, "loan_status", store=store, show=False, **kw)
    try:
        with urllib.request.urlopen(app.url + "api/state", timeout=10) as resp:
            assert json.load(resp)["rows"] == len(train)
        app.session.set_cutoffs("person_income", [20000, 40000])
    finally:
        app.stop()
    assert find_saved_configs(store.output_dir)

    time.sleep(0.01)
    store2 = RunStore(str(tmp_path / "runs"), run_id="run_20260101_000001")   # after a runtime reset
    app2 = launch_app(train, "loan_status", store=store2, show=False, **kw)
    try:
        assert app2.session.result("person_income").cutoffs == [20000, 40000]
        assert app2.session.vars["person_income"].edited
    finally:
        app2.stop()

    other = it.df[make_split(it.df, "loan_status", seed=5) == "train"]
    store3 = RunStore(str(tmp_path / "runs"), run_id="run_20260101_000002")
    app3 = launch_app(other, "loan_status", store=store3, show=False,
                      **{**kw, "full": None, "sample": None})
    try:  # different Train sample: earlier bins are not loaded
        assert app3.session.result("person_income").cutoffs == screen.result("person_income").cutoffs
    finally:
        app3.stop()
