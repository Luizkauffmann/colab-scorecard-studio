"""Stage 3: screening statuses, Train-only fit, review of high IV, outputs."""

import json

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import StudioConfig, get_preset, make_split, run_intake, screen_variables
from scorecard_studio.datasets import CREDIT_RISK_EXCLUDE, CREDIT_RISK_PLAUSIBILITY
from scorecard_studio.screening import is_id_like


@pytest.fixture
def prepared(credit_risk_like):
    df = credit_risk_like.copy()
    rng = np.random.default_rng(0)
    df["customer_no"] = rng.permutation(len(df)) + 10_000          # ID-like integer
    df["free_text"] = [f"note {i}" for i in range(len(df))]         # ID-like string
    df["channel"] = "web"                                           # constant
    df["leak_flag"] = np.where(rng.random(len(df)) < 0.97, df["loan_status"], 1 - df["loan_status"])
    df["amount_k"] = (df["loan_amnt"] / 1000).round(1)               # redundant with loan_amnt
    df["noise"] = rng.normal(size=len(df))
    cfg = StudioConfig(target="loan_status", exclude=CREDIT_RISK_EXCLUDE,
                       plausibility=CREDIT_RISK_PLAUSIBILITY)
    it = run_intake(cfg, df=df)
    sample = make_split(it.df, "loan_status", seed=1)
    return cfg, it, sample


def _screen(cfg, it, sample, **over):
    kw = dict(iv_min=cfg.resolved_iv_min, iv_max=cfg.resolved_iv_max, fit_params=cfg.fit_params,
              exclude=cfg.exclude, force_include=cfg.force_include,
              special_codes=it.special_codes, id_col=it.id_col)
    kw.update(over)
    return screen_variables(it.df[sample == "train"], "loan_status", **kw)


def test_statuses(prepared):
    cfg, it, sample = prepared
    res = _screen(cfg, it, sample)
    st = res.table.set_index("variable")["status"]
    assert st["loan_grade"] == "excluded" and st["loan_int_rate"] == "excluded"
    assert st["customer_no"] == "id_like" and st["free_text"] == "id_like"
    assert st["channel"] == "constant"
    assert st["leak_flag"] == "review"
    assert st["noise"] == "low_iv"
    assert st["loan_percent_income"] in ("selected", "review")
    assert "row_id" not in st.index and "loan_status" not in st.index
    assert set(res.selected) == set(st[st.isin(["selected", "forced"])].index)
    assert "leak_flag" in res.needs_review and "leak_flag" not in res.selected


def test_excluded_variables_are_still_profiled(prepared):
    cfg, it, sample = prepared
    t = _screen(cfg, it, sample).table.set_index("variable")
    assert t.at["loan_grade", "iv"] > 0.1 and t.at["loan_grade", "reason"] == "listed in EXCLUDE"


def test_force_include_overrides_iv_rules_but_exclude_wins(prepared):
    cfg, it, sample = prepared
    res = _screen(cfg, it, sample, force_include=["leak_flag", "noise"])
    t = res.table.set_index("variable")
    assert t.at["leak_flag", "status"] == "forced" and "> IV_MAX" in t.at["leak_flag", "reason"]
    assert t.at["noise", "status"] == "forced" and "< IV_MIN" in t.at["noise", "reason"]
    assert {"leak_flag", "noise"} <= set(res.selected)
    res = _screen(cfg, it, sample, exclude=[*cfg.exclude, "leak_flag"])
    assert res.table.set_index("variable").at["leak_flag", "status"] == "excluded"


def test_iv_min_threshold_moves_variables(prepared):
    cfg, it, sample = prepared
    strict = _screen(cfg, it, sample, iv_min=0.3, iv_max=5.0)
    loose = _screen(cfg, it, sample, iv_min=0.0001, iv_max=5.0)
    assert len(strict.selected) < len(loose.selected)
    sel = strict.table[strict.table["status"] == "selected"]
    assert (sel["iv"] >= 0.3).all()


def test_screening_fits_on_train_only_with_preset_settings(prepared):
    cfg, it, sample = prepared
    res = _screen(cfg, it, sample)
    train = it.df[sample == "train"]
    assert len(res.engine.df) == len(train) and res.params["train_rows"] == len(train)
    r = res.result("person_income")
    assert r.settings["max_bins"] == get_preset("credit_pd")["fit_params"]["max_bins"]
    assert sum(b.count for b in r.bins) == len(train)


def test_plausibility_codes_land_in_special_bin(prepared):
    cfg, it, sample = prepared
    res = _screen(cfg, it, sample)
    r = res.result("person_age")
    n_codes = int((it.df.loc[sample == "train", "person_age"] == -99999).sum())
    special = [b for b in r.bins if b.kind == "special"]
    assert special and special[0].count == n_codes


def test_correlated_pairs_ignore_special_codes(prepared):
    cfg, it, sample = prepared
    pairs = _screen(cfg, it, sample).correlated_pairs
    keyed = {frozenset((a, b)) for a, b in zip(pairs["var_1"], pairs["var_2"])}
    assert frozenset(("loan_amnt", "amount_k")) in keyed
    assert all("customer_no" not in k for k in keyed)


def test_shortlist_and_saved_outputs(prepared, tmp_path):
    cfg, it, sample = prepared
    res = _screen(cfg, it, sample)
    sl = json.loads(json.dumps(res.shortlist(), default=str))
    candidates = set(res.table["variable"])
    assert set(sl["selected"]) | set(sl["dropped"]) == candidates
    assert all(v["reason"] for v in sl["dropped"].values())
    paths = res.save(str(tmp_path / "screening"))
    html = open(paths["report"], encoding="utf-8").read()
    assert "leak_flag" in html and "Held for review" in html and "data:image/png" in html
    assert pd.read_csv(paths["table"]).shape[0] == len(candidates)


def test_bad_names_and_thresholds(prepared):
    cfg, it, sample = prepared
    with pytest.raises(ValueError, match="not candidates"):
        _screen(cfg, it, sample, exclude=["nope"])
    with pytest.raises(ValueError, match="iv_min"):
        _screen(cfg, it, sample, iv_min=0.5, iv_max=0.5)


def test_fit_failure_is_reported_not_raised(prepared):
    cfg, it, sample = prepared
    params = {**cfg.fit_params, "min_bin_n_event": 10**6}
    res = _screen(cfg, it, sample, fit_params=params)
    t = res.table.set_index("variable")
    assert t.at["person_income", "status"] == "fit_error"
    assert "INFEASIBLE" in t.at["person_income", "reason"]
    assert res.selected == []
    assert "fit_error" in res.shortlist()["dropped"]["person_income"]["status"]


def test_id_like_rules():
    n = 500
    assert is_id_like(pd.Series(np.arange(n)), "numerical")
    assert not is_id_like(pd.Series(np.random.default_rng(0).normal(size=n)), "numerical")
    assert is_id_like(pd.Series([f"c{i}" for i in range(n)]), "categorical")
    assert not is_id_like(pd.Series(["a", "b"] * 250), "categorical")
