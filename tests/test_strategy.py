"""Cutoff strategy, gains and calibration diagnostics."""

import math

import numpy as np
import pandas as pd
import pytest

from scorecard_studio.scaling import ScalingParams
from scorecard_studio.strategy import (
    break_even_score,
    choose_cutoff,
    cutoff_curve,
    discrimination,
    gains_table,
    kpis,
    realized_scale,
    roc_points,
    score_psi,
)


@pytest.fixture(scope="module")
def scored():
    """Scores built on a known scale (PDO 20, 600 at 50:1), outcomes drawn
    from it, plus 10% rows with an unknown outcome."""
    rng = np.random.default_rng(0)
    sc = ScalingParams(20, 600, 50)
    s = np.round(rng.normal(560, 45, 40_000))
    p_bad = 1 / (1 + np.exp((s - sc.offset) / sc.factor))
    y = (rng.random(len(s)) < p_bad).astype(float)
    y[rng.random(len(s)) < 0.10] = np.nan
    return s, y, sc


def test_curve_matches_brute_force(scored):
    s, y, _ = scored
    curve = cutoff_curve(s, y, cost_bad=1000, benefit_good=100)
    for c in np.quantile(s, [0.05, 0.3, 0.5, 0.8, 0.97]):
        c = float(np.unique(s)[np.searchsorted(np.unique(s), c)])
        row = curve[curve["cutoff"] == c].iloc[0]
        k = kpis(s, y, c, cost_bad=1000, benefit_good=100)
        assert row["approved"] == k["approved"]
        assert row["bads_approved"] == k["approved_bad"] and row["goods_approved"] == k["approved_good"]
        assert row["profit"] == pytest.approx(k["profit"])
        assert row["approval_rate"] == pytest.approx(k["approval_rate"])
    assert curve["approval_rate"].iloc[0] == 1.0 and curve["approval_rate"].iloc[-1] == 0.0


def test_approval_uses_all_rows_bad_rate_only_known(scored):
    s, y, _ = scored
    k = kpis(s, y, -np.inf)
    assert k["approved"] == len(s) and k["approval_rate"] == 1.0
    known = ~np.isnan(y)
    assert k["approved_good"] + k["approved_bad"] == known.sum()
    assert k["bad_rate_approved"] == pytest.approx(np.nanmean(y))


def test_choose_cutoff_modes(scored):
    s, y, _ = scored
    curve = cutoff_curve(s, y, cost_bad=1000, benefit_good=100)
    best = choose_cutoff(curve, "profit")
    assert best["profit"] == pytest.approx(curve["profit"].max())
    a = choose_cutoff(curve, "approval", 0.7)
    assert a["approval_rate"] >= 0.7
    higher = curve[(curve["cutoff"] > a["cutoff"]) & np.isfinite(curve["cutoff"])]
    assert (higher["approval_rate"] < 0.7).all()          # highest cutoff that still approves 70%
    b = choose_cutoff(curve, "bad_rate", 0.05)
    assert b["bad_rate_approved"] <= 0.05
    m = choose_cutoff(curve, "manual", 555.5)
    assert m["cutoff"] == 555.5 and m["approved"] == (s >= 555.5).sum()
    with pytest.raises(ValueError):
        choose_cutoff(curve, "approval", 1.5)


def test_profit_optimum_sits_at_break_even_when_calibrated(scored):
    s, y, sc = scored
    curve = cutoff_curve(s, y, cost_bad=1000, benefit_good=100)
    best = choose_cutoff(curve, "profit")["cutoff"]
    be = break_even_score(sc, 1000, 100)                  # odds 10:1 -> ~553
    assert abs(best - be) < 12


def test_realized_scale_recovers_design(scored):
    s, y, sc = scored
    g = gains_table(s, y, width=20, anchor=600, scaling=sc)
    r = realized_scale(g, 600)
    assert r["realized_pdo"] == pytest.approx(20, rel=0.15)
    assert r["realized_odds_at_base"] == pytest.approx(50, rel=0.3)
    assert g["rows"].sum() == len(s) and g["cum_approval"].iloc[-1] == pytest.approx(1.0)
    assert g["mean_score"].is_monotonic_decreasing


def test_auc_direction_and_roc(scored):
    s, y, _ = scored
    d = discrimination(s, y)
    assert d["AUC"] > 0.5 and d["Gini"] == pytest.approx(2 * d["AUC"] - 1)
    flipped = discrimination(-s, y)
    assert flipped["AUC"] == pytest.approx(1 - d["AUC"])
    roc = roc_points(s, y)
    area = np.trapezoid(roc["tpr"], roc["fpr"]) if hasattr(np, "trapezoid") else np.trapz(roc["tpr"], roc["fpr"])
    assert area == pytest.approx(d["AUC"], abs=0.01)
    # good_class = 1 with the coding flipped gives the same AUC
    y1 = np.where(np.isnan(y), np.nan, 1 - y)
    assert discrimination(s, y1, good_class=1)["AUC"] == pytest.approx(d["AUC"])


def test_psi(scored):
    s, _, _ = scored
    assert score_psi(s[:20000], s[20000:]) < 0.01
    assert score_psi(s, s + 30) > 0.1
