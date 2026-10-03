"""
Discrimination and stability metrics with explicit, textbook definitions.

Conventions used across the package
-----------------------------------
* ``event`` = target == 1 (default, fraud, SAR ...), ``non-event`` = target == 0.
* WOE  = ln( %events in bin / %non-events in bin ).
  Positive WOE means the bin is *riskier* than average.
* IV   = sum over bins of (%events - %non-events) * WOE.
* AUC  = probability that a random event scores higher than a random
  non-event (ties count 1/2).
* Gini = 2 * AUC - 1. (It is NOT 2 * KS.)
* KS   = max |CDF_events(s) - CDF_non_events(s)| over the score ``s``.
* PSI  = sum (a% - e%) * ln(a% / e%).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

#: Pseudo-count added to *both* events and non-events of a bin when either
#: is zero, so the WOE stays finite. Such bins are also flagged as warnings.
ZERO_CELL_ADJUSTMENT = 0.5


def woe_iv(
    events: float,
    non_events: float,
    total_events: float,
    total_non_events: float,
    adjustment: float = ZERO_CELL_ADJUSTMENT,
) -> Tuple[float, float, bool]:
    """WOE and IV contribution of one bin.

    Returns ``(woe, iv_contribution, adjusted)``. An empty bin gets WOE 0.
    ``adjusted`` is True when a zero cell had to be smoothed.
    """
    if events + non_events == 0 or total_events <= 0 or total_non_events <= 0:
        return 0.0, 0.0, False
    adjusted = events == 0 or non_events == 0
    ev = events + adjustment if adjusted else events
    ne = non_events + adjustment if adjusted else non_events
    dist_e = ev / total_events
    dist_ne = ne / total_non_events
    woe = float(np.log(dist_e / dist_ne))
    return woe, float((dist_e - dist_ne) * woe), adjusted


def _sorted_counts(events, non_events, scores):
    events = np.asarray(events, dtype=float)
    non_events = np.asarray(non_events, dtype=float)
    scores = np.asarray(scores, dtype=float)
    # Merge bins with identical scores (they are ties for AUC/KS purposes).
    uniq, inv = np.unique(scores, return_inverse=True)
    ev = np.bincount(inv, weights=events, minlength=len(uniq))
    ne = np.bincount(inv, weights=non_events, minlength=len(uniq))
    return ev, ne


def auc_from_bins(events: Sequence[float], non_events: Sequence[float],
                  scores: Sequence[float]) -> float:
    """AUC of a score that is constant within each bin (ties handled exactly)."""
    ev, ne = _sorted_counts(events, non_events, scores)
    te, tne = ev.sum(), ne.sum()
    if te == 0 or tne == 0:
        return 0.5
    ne_below = np.concatenate([[0.0], np.cumsum(ne)[:-1]])
    return float(np.sum(ev * (ne_below + 0.5 * ne)) / (te * tne))


def ks_from_bins(events: Sequence[float], non_events: Sequence[float],
                 scores: Sequence[float]) -> float:
    """KS of a score that is constant within each bin (bins ordered by score)."""
    ev, ne = _sorted_counts(events, non_events, scores)
    te, tne = ev.sum(), ne.sum()
    if te == 0 or tne == 0:
        return 0.0
    return float(np.max(np.abs(np.cumsum(ev) / te - np.cumsum(ne) / tne)))


def gini_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0


def auc_score(y: Sequence[int], score: Sequence[float]) -> float:
    """Row-level AUC (Mann-Whitney, average ranks for ties). No sklearn needed."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    uniq, inv = np.unique(score, return_inverse=True)
    ev = np.bincount(inv, weights=(y == 1).astype(float), minlength=len(uniq))
    ne = np.bincount(inv, weights=(y == 0).astype(float), minlength=len(uniq))
    return auc_from_bins(ev, ne, uniq)


def gini_score(y: Sequence[int], score: Sequence[float]) -> float:
    return gini_from_auc(auc_score(y, score))


def ks_score(y: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    uniq, inv = np.unique(score, return_inverse=True)
    ev = np.bincount(inv, weights=(y == 1).astype(float), minlength=len(uniq))
    ne = np.bincount(inv, weights=(y == 0).astype(float), minlength=len(uniq))
    return ks_from_bins(ev, ne, uniq)


def psi(expected_counts: Sequence[float], actual_counts: Sequence[float],
        epsilon: float = 1e-6) -> float:
    """Population Stability Index between two binned distributions."""
    e = np.asarray(expected_counts, dtype=float)
    a = np.asarray(actual_counts, dtype=float)
    e = np.clip(e / max(e.sum(), 1e-12), epsilon, None)
    a = np.clip(a / max(a.sum(), 1e-12), epsilon, None)
    return float(np.sum((a - e) * np.log(a / e)))


IV_BANDS = [
    (0.00, 0.02, "Useless"),
    (0.02, 0.10, "Weak"),
    (0.10, 0.30, "Medium"),
    (0.30, 0.50, "Strong"),
    (0.50, float("inf"), "Very strong - check for leakage"),
]


def interpret_iv(iv: float) -> str:
    for lo, hi, label in IV_BANDS:
        if lo <= iv < hi:
            return label
    return "Unknown"
