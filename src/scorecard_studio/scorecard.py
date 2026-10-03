"""
Scorecard: points per bin (Siddiqi), scoring and rescoring new data.

Scaling (Siddiqi, *Credit Risk Scorecards*; optbinning's ``pdo_odds``)
----------------------------------------------------------------------
With ``base_odds`` = non-event : event odds at ``base_score``::

    Factor = PDO / ln(2)
    Offset = base_score - Factor * ln(base_odds)
    Score  = Offset + Factor * ln(odds_non_event) = Offset - Factor * logit(p_event)

For a logistic regression ``logit(p) = a + sum_j b_j * x_j`` over ``n``
variables, each attribute (bin) gets::

    WOE variable:   points = -(b_j * WOE_bin + a / n) * Factor + Offset / n
    bins variable:  points = -(b_bin       + a / n) * Factor + Offset / n   (reference bin: b = 0)
    raw variable:   points = -(b_j * x     + a / n) * Factor + Offset / n   (linear in x, no table)

so a record's points add up exactly to its score (before rounding). With
``base_points="separate"`` the intercept and offset go into one base-points
row instead of being spread over the variables.

The scorecard is a plain table you can edit (``set_points``): every row keeps
the model's points and the points in use, and flags manual changes, so a
later adjustment step changes scores without touching the model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .artifacts import ScoringArtifact, ScoringBundle
from .metrics import auc_score, gini_from_auc, ks_score
from .scaling import ScalingParams

SCORECARD_SCHEMA_VERSION = 1
REPRESENTATIONS = ("woe", "bins", "raw")


# ═══════════════════════════════════════════════════════════════════════════
# Applying bins (one code path for the output dataset, rescoring, scorecard)
# ═══════════════════════════════════════════════════════════════════════════

def ordered_labels(art: ScoringArtifact):
    """Unique bin labels in bin order (regular bins, Special, Missing) and the
    label of each group. Missing/Special merged into a regular bin share it."""
    table = art.regular + ([art.special] if art.special else []) + [art.missing]
    seen, order, by_group = set(), [], {}
    for b in table:
        if b["group"] in by_group:
            continue
        label = b["label"]
        if label in seen:
            label = f"{label} [group {b['group']}]"
        seen.add(label)
        order.append(label)
        by_group[b["group"]] = label
    return order, by_group


def apply_bins(df: pd.DataFrame, bundle: ScoringBundle, variables: Optional[Iterable[str]] = None,
               keep_original: bool = True) -> pd.DataFrame:
    """``opt_<var>`` (bin label, ordered category) and ``woe_<var>`` for every
    variable of ``bundle``, in the same layout as the model-ready dataset.
    ``df`` must already have the intake plausibility rules applied (the
    :class:`Scorecard` does that for you)."""
    names = list(variables) if variables is not None else list(bundle.artifacts)
    absent = [v for v in names if v not in df.columns]
    if absent:
        raise KeyError(f"Columns missing from input: {absent}")
    cols: Dict[str, Any] = {}
    for v in names:
        art = bundle.artifacts[v]
        res = art.transform_series(df[v])
        order, by_group = ordered_labels(art)
        if keep_original:
            cols[v] = df[v]
        cols[f"opt_{v}"] = pd.Categorical([by_group[g] for g in res["group"]], categories=order, ordered=True)
        cols[f"woe_{v}"] = res["woe"].astype("float64")
    return pd.DataFrame(cols, index=df.index)


def apply_plausibility_rules(df: pd.DataFrame, rules: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Re-apply the intake plausibility rules to new data (same codes)."""
    if not rules or not rules.get("rules"):
        return df
    from .intake import apply_plausibility
    present = {c: tuple(r) for c, r in rules["rules"].items() if c in df.columns}
    # Idempotent: data that already went through intake keeps its codes.
    out, _, _ = apply_plausibility(df, present, rules.get("action", "special"), rules.get("code", -99999),
                                   allow_existing_code=True)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Scorecard
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Scorecard:
    """Points per bin plus everything needed to score raw data."""

    target: str
    scaling: ScalingParams
    intercept: float
    representations: Dict[str, str]           # variable -> woe | bins | raw
    points: pd.DataFrame                       # one row per (variable, group); raw rows per variable
    bundle: ScoringBundle                      # bins of the woe/bins variables
    base_points_mode: str = "spread"           # spread | separate
    base_points: float = 0.0                   # used when mode == "separate"
    round_points: bool = True
    plausibility: Dict[str, Any] = field(default_factory=dict)
    raw_terms: Dict[str, Dict[str, float]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------

    @property
    def variables(self) -> List[str]:
        return list(self.representations)

    def table(self) -> pd.DataFrame:
        """The scorecard as you would print it."""
        cols = ["Variable", "Representation", "Group", "Kind", "Label", "WOE", "Coefficient",
                "Points", "Model points", "Manual", "Count", "Event rate"]
        t = self.points[[c for c in cols if c in self.points.columns]].copy()
        if self.base_points_mode == "separate":
            base = pd.DataFrame([{"Variable": "(base points)", "Representation": "", "Group": None,
                                  "Kind": "base", "Label": "", "Points": self.base_points,
                                  "Model points": self.base_points, "Manual": False}])
            t = pd.concat([base, t], ignore_index=True)
        return t

    def set_points(self, variable: str, group: int, points: float) -> None:
        """Manual judgment: override one bin's points. Scores use the new value;
        the model's value stays in ``Model points``."""
        m = (self.points["Variable"] == variable) & (self.points["Group"] == group)
        if not m.any():
            raise KeyError(f"No bin {group} for '{variable}'.")
        self.points.loc[m, "Points"] = float(points)
        self.points.loc[m, "Manual"] = True

    def reset_points(self) -> None:
        self.points["Points"] = self.points["Model points"]
        self.points["Manual"] = False

    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame, keep: Sequence[str] = (), points_detail: bool = True) -> pd.DataFrame:
        """Score raw data: plausibility rules -> bins -> points -> score.

        Returns ``keep`` columns, ``pts_<var>`` per variable (if
        ``points_detail``), ``score`` and ``pd`` (from the score, via the
        scaling). Columns not used by the scorecard are ignored."""
        data = apply_plausibility_rules(df, self.plausibility)
        out = pd.DataFrame(index=df.index)
        for c in keep:
            out[c] = df[c]
        total = np.full(len(df), self.base_points if self.base_points_mode == "separate" else 0.0)
        lookup = {(r.Variable, int(r.Group)): float(r.Points)
                  for r in self.points.itertuples() if pd.notna(r.Group)}
        for v, rep in self.representations.items():
            if v not in data.columns:
                raise KeyError(f"Column '{v}' is required by the scorecard.")
            if rep == "raw":
                t = self.raw_terms[v]
                x = pd.to_numeric(data[v], errors="coerce").to_numpy(dtype=float)
                if np.isnan(x).any():
                    raise ValueError(f"'{v}' is used raw (linear) and has missing values in this data.")
                pts = t["points_per_unit"] * x + t["constant"]
                if self.round_points:
                    pts = np.round(pts)
            else:
                groups = self.bundle.artifacts[v].transform_series(data[v])["group"]
                pts = np.array([lookup[(v, int(g))] for g in groups], dtype=float)
            if points_detail:
                out[f"pts_{v}"] = pts
            total = total + pts
        out["score"] = total
        out["pd"] = 1.0 / (1.0 + np.exp(-(self.scaling.offset - total) / self.scaling.factor))
        return out

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        pts = self.points.astype(object).where(self.points.notna(), None)
        return {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "target": self.target,
            "scaling": {"pdo": self.scaling.pdo, "base_score": self.scaling.base_score,
                        "base_odds": self.scaling.base_odds},
            "intercept": self.intercept,
            "representations": self.representations,
            "base_points_mode": self.base_points_mode,
            "base_points": self.base_points,
            "round_points": self.round_points,
            "plausibility": self.plausibility,
            "raw_terms": self.raw_terms,
            "points": pts.to_dict(orient="records"),
            "bundle": self.bundle.to_dict(),
            "meta": self.meta,
        }

    def save(self, path: str) -> str:
        with open(path, "w") as fh:
            json.dump(_json_safe(self.to_dict()), fh, indent=2)
        return path

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scorecard":
        if d.get("schema_version") != SCORECARD_SCHEMA_VERSION:
            raise ValueError(f"Unsupported scorecard schema_version {d.get('schema_version')!r}")
        return cls(
            target=d["target"], scaling=ScalingParams(**d["scaling"]), intercept=float(d["intercept"]),
            representations=dict(d["representations"]), points=pd.DataFrame(d["points"]),
            bundle=ScoringBundle.from_dict(d["bundle"]), base_points_mode=d["base_points_mode"],
            base_points=float(d["base_points"]), round_points=bool(d["round_points"]),
            plausibility=d.get("plausibility") or {}, raw_terms=d.get("raw_terms") or {},
            meta=d.get("meta") or {})


def load_scorecard(path: str) -> Scorecard:
    """Reload a saved scorecard to rescore new data: ``load_scorecard(p).score(df)``."""
    with open(path) as fh:
        return Scorecard.from_dict(json.load(fh))


def build_scorecard(model, bundle: ScoringBundle, scaling: ScalingParams = ScalingParams(),
                    base_points: str = "spread", round_points: bool = True,
                    plausibility: Optional[Dict[str, Any]] = None) -> Scorecard:
    """Turn a fitted :class:`~scorecard_studio.model.LogisticModel` into points."""
    if base_points not in ("spread", "separate"):
        raise ValueError("base_points must be 'spread' or 'separate'")
    reps = dict(model.representations)
    n = len(reps)
    f, off, a = scaling.factor, scaling.offset, float(model.intercept)
    shift = (off - f * a) / n if base_points == "spread" else 0.0
    base = off - f * a if base_points == "separate" else 0.0
    rnd = (lambda x: float(np.round(x))) if round_points else float
    rows, raw_terms = [], {}
    for v, rep in reps.items():
        if rep == "raw":
            beta = model.coef[v]
            raw_terms[v] = {"beta": beta, "points_per_unit": -f * beta, "constant": shift}
            rows.append({"Variable": v, "Representation": "raw", "Group": None, "Kind": "linear",
                         "Label": f"{-f * beta:+.4g} points per unit, {shift:+.4g} constant",
                         "WOE": None, "Coefficient": beta, "Model points": None, "Points": None,
                         "Manual": False, "Count": None, "Event rate": None})
            continue
        art = bundle.artifacts[v]
        emitted = set()
        # Regular bins first, then Missing, then Special: a merged (alias) entry
        # never hides the bin that actually holds the rows.
        for b in sorted(art.bins, key=lambda x: {"regular": 0, "missing": 1, "special": 2}[x["kind"]]):
            if b["group"] in emitted:
                continue    # Missing/Special merged into another bin: scored as that bin
            emitted.add(b["group"])
            if rep == "woe":
                beta = model.coef[v]
                raw_pts = -f * beta * b["woe"] + shift
            else:
                beta = model.bin_coef(v, b["group"])
                raw_pts = -f * beta + shift
            rows.append({"Variable": v, "Representation": rep, "Group": int(b["group"]), "Kind": b["kind"],
                         "Label": b["label"], "WOE": b["woe"], "Coefficient": beta,
                         "Model points": rnd(raw_pts), "Points": rnd(raw_pts), "Manual": False,
                         "Count": b.get("count"), "Event rate": b.get("event_rate")})
    sub = ScoringBundle({v: bundle.artifacts[v] for v, r in reps.items() if r != "raw"}, name="scorecard_bins")
    return Scorecard(target=model.target, scaling=scaling, intercept=a, representations=reps,
                     points=pd.DataFrame(rows), bundle=sub, base_points_mode=base_points,
                     base_points=rnd(base) if base_points == "separate" else 0.0,
                     round_points=round_points, plausibility=plausibility or {}, raw_terms=raw_terms,
                     meta={"n_variables": n, "factor": f, "offset": off})


# ═══════════════════════════════════════════════════════════════════════════
# Performance
# ═══════════════════════════════════════════════════════════════════════════

def performance(y: Sequence[int], score: Sequence[float], sample: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """AUC, Gini (= 2*AUC - 1) and KS of the score per sample. Higher score =
    lower risk, so the event-ranking direction is ``-score``."""
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    s = pd.Series(np.asarray(score, dtype=float)).reset_index(drop=True)
    smp = pd.Series(np.asarray(sample) if sample is not None else np.repeat("all", len(y)))
    rows = []
    order = [x for x in ("train", "test", "oot") if x in set(smp)] + sorted(set(smp) - {"train", "test", "oot"})
    for name in order:
        m = (smp == name).to_numpy()
        auc = auc_score(y[m], -s[m])
        rows.append({"sample": name, "rows": int(m.sum()), "events": int(y[m].sum()),
                     "event_rate": float(y[m].mean()), "AUC": auc, "Gini": gini_from_auc(auc),
                     "KS": ks_score(y[m], -s[m])})
    return pd.DataFrame(rows)


def score_bands(y: Sequence[int], score: Sequence[float], sample: Sequence[str], n_bands: int = 10) -> pd.DataFrame:
    """Rows and event rate per score band; band edges from the Train deciles."""
    df = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score, float), "sample": np.asarray(sample)})
    train = df[df["sample"] == "train"]["score"]
    edges = np.unique(np.quantile(train, np.linspace(0, 1, n_bands + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    df["band"] = pd.cut(df["score"], edges, include_lowest=True)
    t = df.groupby(["band", "sample"], observed=True)["y"].agg(["size", "mean"]).unstack("sample")
    t.columns = [f"{a}_{b}".replace("size", "rows").replace("mean", "event_rate") for a, b in t.columns]
    return t.reset_index()


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
