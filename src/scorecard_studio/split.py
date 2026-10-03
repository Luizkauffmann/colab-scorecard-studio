"""
Stage 2 - sample design.

* With a date column: the most recent period is held out as OOT (from
  ``oot_start``, or the last ``oot_share`` of rows by date). The rest is
  split into Train/Test, stratified on the target.
* Without one: stratified random Train/Test only. There is no OOT, so the
  validation measures in-sample generalisation, not stability over time.

Bins, WOE and screening are fit on Train only. Test and OOT are never used
to choose anything.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

TRAIN, TEST, OOT = "train", "test", "oot"
SAMPLES = (TRAIN, TEST, OOT)


def stratified_split(y: pd.Series, test_size: float, seed: int) -> pd.Series:
    """Train/Test labels, with the same event rate (up to rounding) in both."""
    rng = np.random.default_rng(seed)
    labels = pd.Series(TRAIN, index=y.index, dtype=object)
    for cls in sorted(pd.unique(y)):
        idx = y.index[(y == cls).to_numpy()]
        n_test = int(round(len(idx) * test_size))
        chosen = rng.permutation(len(idx))[:n_test]
        labels.loc[idx[chosen]] = TEST
    return labels


def make_split(df: pd.DataFrame, target: str, date_col: Optional[str] = None,
               test_size: float = 0.30, oot_start: Optional[str] = None,
               oot_share: float = 0.20, seed: int = 42) -> pd.Series:
    """Return a Series of ``"train" | "test" | "oot"`` aligned with ``df``."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    y = df[target]
    if date_col is None:
        if oot_start is not None:
            raise ValueError("OOT_START needs DATE_COL.")
        return stratified_split(y, test_size, seed).rename("sample")

    dates = pd.to_datetime(df[date_col])
    if dates.isna().any():
        raise ValueError(f"{date_col!r} has missing dates.")
    if oot_start is not None:
        cut = pd.Timestamp(oot_start)
    else:
        if not 0 < oot_share < 1:
            raise ValueError("oot_share must be in (0, 1)")
        cut = dates.quantile(1 - oot_share)
        # Whole days only: never split rows of the same date across samples.
        cut = cut.normalize() + pd.Timedelta(days=1) if cut.normalize() != cut else cut
    is_oot = (dates >= cut).to_numpy()
    if is_oot.all() or not is_oot.any():
        raise ValueError(f"OOT start {cut.date()} leaves an empty development or OOT sample "
                         f"(dates run {dates.min().date()} to {dates.max().date()}).")
    labels = pd.Series(OOT, index=df.index, dtype=object, name="sample")
    dev = ~is_oot
    labels.loc[dev] = stratified_split(y[dev], test_size, seed)
    return labels


def split_summary(df: pd.DataFrame, target: str, sample: pd.Series,
                  date_col: Optional[str] = None) -> pd.DataFrame:
    """Rows, events and event rate per sample (plus date range when dated)."""
    rows = []
    for s in SAMPLES:
        m = (sample == s).to_numpy()
        if not m.any():
            continue
        y = df.loc[m, target]
        row = {"sample": s, "rows": int(m.sum()), "share": round(float(m.mean()), 4),
               "events": int(y.sum()), "event_rate": round(float(y.mean()), 4)}
        if date_col is not None:
            d = pd.to_datetime(df.loc[m, date_col])
            row.update({"date_from": d.min().date(), "date_to": d.max().date()})
        rows.append(row)
    return pd.DataFrame(rows)


def split_warnings(summary: pd.DataFrame, min_events: int = 50) -> List[str]:
    """Things to look at before moving on."""
    out = []
    if OOT not in set(summary["sample"]):
        out.append("No OOT sample (no DATE_COL). Test measures in-sample generalisation, "
                   "not stability over time; PSI between samples will look flatter than in production.")
    for _, r in summary.iterrows():
        if r["events"] < min_events:
            out.append(f"{r['sample']} has only {r['events']} events; its metrics will be noisy.")
    dev = summary[summary["sample"] == TRAIN]
    oot = summary[summary["sample"] == OOT]
    if len(dev) and len(oot):
        a, b = float(dev["event_rate"].iloc[0]), float(oot["event_rate"].iloc[0])
        if a > 0 and abs(b / a - 1) > 0.25:
            out.append(f"OOT event rate ({b:.2%}) differs from Train ({a:.2%}) by more than 25%. "
                       "Check the performance window and default definition before modeling.")
    return out
