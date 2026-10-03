"""
Cutoff strategy, gains and calibration for a scorecard.

Conventions
-----------
* A higher score means a lower risk. An applicant is **approved when
  ``score >= cutoff``**.
* ``bad`` = the target value that is not ``good_class`` (default: target 1).
* Approval rates use **every** row (the population that would be scored);
  bad rates, capture rates and profit use only rows with a **known**
  outcome. In onboarding the rows without an outcome are mostly past
  rejects, so mixing the two populations biases the cutoff.
* Curves are exact: scores are sorted once and every distinct score is a
  candidate cutoff (no grid).
* Profit = benefit_good x goods approved - cost_bad x bads approved, i.e.
  the margin earned on a good loan and the loss on a bad one.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import auc_score, gini_from_auc, ks_score
from .metrics import psi as _psi


def _bad_and_known(y, good_class: int):
    y = pd.to_numeric(pd.Series(np.asarray(y, dtype=object)), errors="coerce").to_numpy(dtype=float)
    known = ~np.isnan(y)
    bad = known & (y != good_class)
    return bad, known


def cutoff_curve(score: Sequence[float], y: Sequence, good_class: int = 0,
                 cost_bad: float = 1.0, benefit_good: float = 0.0) -> pd.DataFrame:
    """Every distinct cutoff (approve ``score >= cutoff``) with its approval,
    bad rate, capture and profit. Includes a final "approve nobody" row."""
    s = np.asarray(score, dtype=float)
    bad, known = _bad_and_known(y, good_class)
    good = known & ~bad
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    cum_n = np.arange(1, len(s) + 1)
    cum_bad = np.cumsum(bad[order])
    cum_good = np.cumsum(good[order])
    # last position of each distinct score (descending): approving all rows with score >= that value
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s) - 1]
    n, n_bad, n_good = len(s), int(bad.sum()), int(good.sum())
    df = pd.DataFrame({
        "cutoff": s_sorted[last],
        "approved": cum_n[last],
        "goods_approved": cum_good[last],
        "bads_approved": cum_bad[last],
    })
    df = pd.concat([pd.DataFrame([{"cutoff": np.inf, "approved": 0, "goods_approved": 0,
                                   "bads_approved": 0}]), df], ignore_index=True)
    df = df.iloc[::-1].reset_index(drop=True)          # ascending cutoff: approve everybody first
    known_appr = df["goods_approved"] + df["bads_approved"]
    df["approval_rate"] = df["approved"] / max(n, 1)
    df["bad_rate_approved"] = np.where(known_appr > 0, df["bads_approved"] / known_appr.where(known_appr > 0, 1), np.nan)
    df["bads_rejected"] = n_bad - df["bads_approved"]
    df["goods_rejected"] = n_good - df["goods_approved"]
    df["bad_capture"] = df["bads_rejected"] / max(n_bad, 1)          # share of bads declined
    df["good_reject_rate"] = df["goods_rejected"] / max(n_good, 1)
    df["profit"] = benefit_good * df["goods_approved"] - cost_bad * df["bads_approved"]
    df["profit_per_applicant"] = df["profit"] / max(int(known.sum()), 1)
    return df


def choose_cutoff(curve: pd.DataFrame, mode: str = "profit", value: Optional[float] = None) -> Dict[str, Any]:
    """Pick a cutoff from :func:`cutoff_curve`.

    * ``"profit"``: maximum profit (ties: the higher approval rate)
    * ``"approval"``: the highest cutoff that still approves at least ``value``
    * ``"bad_rate"``: the lowest cutoff whose bad rate among approved is at most
      ``value`` (i.e. the most approvals within that risk appetite)
    * ``"manual"``: ``value`` itself
    """
    c = curve[np.isfinite(curve["cutoff"])]
    if c.empty:
        raise ValueError("No scores to choose a cutoff from.")
    if mode == "profit":
        best = c["profit"].max()
        row = c[c["profit"] == best].sort_values("approval_rate", ascending=False).iloc[0]
    elif mode == "approval":
        if value is None or not 0 < value <= 1:
            raise ValueError("Target approval rate must be in (0, 1].")
        ok = c[c["approval_rate"] >= value - 1e-12]
        row = (ok.sort_values("cutoff").iloc[-1] if len(ok) else c.sort_values("cutoff").iloc[0])
    elif mode == "bad_rate":
        if value is None or not 0 <= value <= 1:
            raise ValueError("Target bad rate must be in [0, 1].")
        ok = c[c["bad_rate_approved"] <= value + 1e-12]
        if ok.empty:
            row = c.sort_values("cutoff").iloc[-1]
        else:
            row = ok.sort_values("cutoff").iloc[0]
    elif mode == "manual":
        if value is None:
            raise ValueError("Give the cutoff value.")
        idx = int(np.searchsorted(c["cutoff"].to_numpy(), float(value), side="left"))
        idx = min(idx, len(c) - 1)
        row = c.iloc[idx].copy()
        row["cutoff"] = float(value)
    else:
        raise ValueError("mode must be 'profit', 'approval', 'bad_rate' or 'manual'")
    return {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v) for k, v in row.items()}


def kpis(score: Sequence[float], y: Sequence, cutoff: float, good_class: int = 0,
         cost_bad: float = 1.0, benefit_good: float = 0.0) -> Dict[str, Any]:
    """Confusion matrix and KPIs at one cutoff (approve ``score >= cutoff``)."""
    s = np.asarray(score, dtype=float)
    bad, known = _bad_and_known(y, good_class)
    good = known & ~bad
    appr = s >= cutoff
    n_known = int(known.sum())
    out = {
        "cutoff": float(cutoff), "rows": int(len(s)), "rows_known": n_known,
        "approved": int(appr.sum()), "approval_rate": float(appr.mean()) if len(s) else math.nan,
        "approved_good": int((appr & good).sum()), "approved_bad": int((appr & bad).sum()),
        "declined_good": int((~appr & good).sum()), "declined_bad": int((~appr & bad).sum()),
    }
    ka = out["approved_good"] + out["approved_bad"]
    out["bad_rate_approved"] = out["approved_bad"] / ka if ka else math.nan
    out["bad_rate_all"] = float(bad.sum() / n_known) if n_known else math.nan
    out["bad_capture"] = out["declined_bad"] / max(int(bad.sum()), 1)
    out["good_reject_rate"] = out["declined_good"] / max(int(good.sum()), 1)
    out["profit"] = benefit_good * out["approved_good"] - cost_bad * out["approved_bad"]
    out["profit_per_applicant"] = out["profit"] / max(n_known, 1)
    return out


def discrimination(score: Sequence[float], y: Sequence, good_class: int = 0) -> Dict[str, float]:
    """AUC (probability a random bad scores *lower* than a random good),
    Gini = 2*AUC - 1, and KS, on rows with a known outcome."""
    s = np.asarray(score, dtype=float)
    bad, known = _bad_and_known(y, good_class)
    if bad[known].all() or not bad[known].any():
        return {"AUC": math.nan, "Gini": math.nan, "KS": math.nan}
    auc = auc_score(bad[known].astype(int), -s[known])
    return {"AUC": auc, "Gini": gini_from_auc(auc), "KS": ks_score(bad[known].astype(int), -s[known])}


def roc_points(score: Sequence[float], y: Sequence, good_class: int = 0, max_points: int = 200) -> pd.DataFrame:
    """ROC curve for detecting bads with a low score: x = share of goods
    declined, y = share of bads declined, as the cutoff rises."""
    curve = cutoff_curve(score, y, good_class)
    pts = curve[["good_reject_rate", "bad_capture"]].rename(columns={"good_reject_rate": "fpr", "bad_capture": "tpr"})
    pts = pd.concat([pd.DataFrame([{"fpr": 0.0, "tpr": 0.0}]), pts.iloc[::-1]], ignore_index=True)
    pts = pts.drop_duplicates().sort_values(["fpr", "tpr"]).reset_index(drop=True)
    if len(pts) > max_points:
        idx = np.unique(np.linspace(0, len(pts) - 1, max_points).round().astype(int))
        pts = pts.iloc[idx].reset_index(drop=True)
    return pts


def gains_table(score: Sequence[float], y: Sequence, good_class: int = 0, width: float = 20.0,
                anchor: float = 600.0, scaling=None, min_share: float = 0.005) -> pd.DataFrame:
    """Rows per score band of fixed ``width`` (anchored so a band starts at
    ``anchor``), highest scores first. Thin tail bands (< ``min_share`` of
    rows) are folded into their neighbour. With ``scaling``, adds the bad
    rate the scale expects at the band's mean score."""
    s = np.asarray(score, dtype=float)
    bad, known = _bad_and_known(y, good_class)
    if len(s) == 0:
        return pd.DataFrame()
    lo = anchor + width * math.floor((s.min() - anchor) / width)
    hi = anchor + width * math.ceil((s.max() - anchor) / width + 1e-12)
    edges = np.arange(lo, hi + width / 2, width)
    if len(edges) < 2:
        edges = np.array([lo, lo + width])
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    # fold thin tails
    keep = list(range(len(counts)))
    n = len(s)
    while len(keep) > 1 and counts[keep[0]] < min_share * n:
        counts[keep[1]] += counts[keep[0]]
        idx[idx == keep[0]] = keep[1]
        keep.pop(0)
    while len(keep) > 1 and counts[keep[-1]] < min_share * n:
        counts[keep[-2]] += counts[keep[-1]]
        idx[idx == keep[-1]] = keep[-2]
        keep.pop()
    rows = []
    for i, k in enumerate(keep):
        m = idx == k
        lo_e = -math.inf if i == 0 else edges[k]
        hi_e = math.inf if i == len(keep) - 1 else edges[k + 1]
        nb = int((m & bad).sum())
        ng = int((m & known & ~bad).sum())
        rows.append({"band": f"[{_fmt(lo_e)}, {_fmt(hi_e)})", "lower": lo_e, "upper": hi_e,
                     "rows": int(m.sum()), "mean_score": float(s[m].mean()) if m.any() else math.nan,
                     "goods": ng, "bads": nb, "bad_rate": nb / (ng + nb) if ng + nb else math.nan,
                     "odds": ng / nb if nb else math.inf})
    t = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)      # highest scores first
    t["share"] = t["rows"] / n
    t["ln_odds"] = np.log(t["odds"].replace([np.inf, 0], np.nan))
    t["cum_approval"] = t["rows"].cumsum() / n
    cg, cb = t["goods"].cumsum(), t["bads"].cumsum()
    t["cum_bad_rate"] = cb / (cg + cb).where((cg + cb) > 0, np.nan)
    if scaling is not None:
        t["expected_bad_rate"] = 1.0 / (1.0 + np.exp((t["mean_score"] - scaling.offset) / scaling.factor))
    return t


def realized_scale(gains: pd.DataFrame, base_score: float) -> Dict[str, float]:
    """Fit observed ln(odds) on band mean score (weighted by rows): the
    realized PDO is ln(2)/slope and the realized odds at ``base_score`` come
    from the fitted line. Compare them with the design PDO and base odds."""
    g = gains.dropna(subset=["ln_odds", "mean_score"])
    g = g[np.isfinite(g["ln_odds"])]
    if len(g) < 2:
        return {"realized_pdo": math.nan, "realized_odds_at_base": math.nan, "bands_used": len(g)}
    w = g["rows"].to_numpy(float)
    x, yv = g["mean_score"].to_numpy(float), g["ln_odds"].to_numpy(float)
    xm, ym = np.average(x, weights=w), np.average(yv, weights=w)
    slope = np.sum(w * (x - xm) * (yv - ym)) / np.sum(w * (x - xm) ** 2)
    intercept = ym - slope * xm
    return {"realized_pdo": math.log(2) / slope if slope > 0 else math.nan,
            "realized_odds_at_base": float(math.exp(intercept + slope * base_score)),
            "slope": float(slope), "intercept": float(intercept), "bands_used": int(len(g))}


def score_psi(expected: Sequence[float], actual: Sequence[float], n_bins: int = 10) -> float:
    """PSI of the score between two samples, deciles of ``expected``."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ec = np.histogram(e, edges)[0]
    ac = np.histogram(a, edges)[0]
    return _psi(ec, ac)


def score_histogram(score: Sequence[float], y: Sequence, good_class: int = 0, width: float = 5.0) -> pd.DataFrame:
    """Goods and bads per score bucket of ``width`` (for the distribution chart)."""
    s = np.asarray(score, dtype=float)
    bad, known = _bad_and_known(y, good_class)
    lo = width * math.floor(s.min() / width)
    b = np.floor((s - lo) / width).astype(int)
    n = b.max() + 1
    return pd.DataFrame({"from": lo + width * np.arange(n),
                         "goods": np.bincount(b, weights=(known & ~bad), minlength=n).astype(int),
                         "bads": np.bincount(b, weights=bad, minlength=n).astype(int),
                         "unknown": np.bincount(b, weights=~known, minlength=n).astype(int)})


def break_even_score(scaling, cost_bad: float, benefit_good: float) -> float:
    """The score where the scale says approving breaks even: odds good:bad =
    cost_bad / benefit_good. With calibrated scores the profit-optimal cutoff
    sits close to it; a large gap means the scale is off for this data."""
    if cost_bad <= 0 or benefit_good <= 0:
        return math.nan
    return scaling.offset + scaling.factor * math.log(cost_bad / benefit_good)


def _fmt(x: float) -> str:
    if math.isinf(x):
        return "-inf" if x < 0 else "inf"
    return f"{x:g}"
