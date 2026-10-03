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

(signs as written for ``good_class=0``, i.e. target 1 = bad; with
``good_class=1`` the signs of the b terms flip so a higher score still
means a higher probability of good). Bin points are rounded to integers;
raw (linear) terms are not rounded per record, so Python and SQL agree.

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
    """Points per bin plus everything needed to score raw data.

    ``good_class`` is the target value that means *good* (default 0: target 1
    is the event / bad). A higher score always means a higher probability of
    good, whatever the coding, so AUC, cutoffs and decisions can't be
    inverted by a sign slip. ``pd`` in :meth:`score` is the probability of
    *bad*.
    """

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
    good_class: int = 0
    cutoff: Optional[float] = None             # approve if score >= cutoff

    # ------------------------------------------------------------------

    @property
    def variables(self) -> List[str]:
        return list(self.representations)

    @property
    def direction(self) -> int:
        """-1 when target 1 is bad (score falls as P(target=1) rises), +1 otherwise."""
        return -1 if self.good_class == 0 else 1

    def scale_signature(self) -> str:
        s = self.scaling
        return f"PDO {s.pdo:g}, {s.base_score:g} at {s.base_odds:g}:1, {self.base_points_mode}, good={self.good_class}"

    def table(self) -> pd.DataFrame:
        """The scorecard as you would print it."""
        cols = ["Variable", "Representation", "Group", "Kind", "Label", "WOE", "Coefficient",
                "Points", "Model points", "Manual", "Reason", "Stale", "Count", "Event rate"]
        t = self.points[[c for c in cols if c in self.points.columns]].copy()
        if self.base_points_mode == "separate":
            base = pd.DataFrame([{"Variable": "(base points)", "Representation": "", "Group": None,
                                  "Kind": "base", "Label": "", "Points": self.base_points,
                                  "Model points": self.base_points, "Manual": False}])
            t = pd.concat([base, t], ignore_index=True)
        return t

    # ------------------------------------------------------------------ edits

    def set_points(self, variable: str, group: int, points: float, reason: str = "") -> None:
        """Manual judgment: override one bin's points. Scores use the new value;
        the model's value stays in ``Model points``."""
        m = (self.points["Variable"] == variable) & (self.points["Group"] == group)
        if not m.any():
            raise KeyError(f"No bin {group} for '{variable}'.")
        if not math.isfinite(float(points)):
            raise ValueError("Points must be a finite number.")
        self.points.loc[m, "Points"] = float(points)
        self.points.loc[m, "Manual"] = True
        self.points.loc[m, "Reason"] = str(reason or "")
        self.points.loc[m, "Override scale"] = self.scale_signature()
        self.points.loc[m, "Stale"] = False

    def reset_points(self, variable: Optional[str] = None, group: Optional[int] = None) -> None:
        """Back to the model's points: everything, one variable, or one bin."""
        m = pd.Series(True, index=self.points.index)
        if variable is not None:
            m &= self.points["Variable"] == variable
        if group is not None:
            m &= self.points["Group"] == group
        self.points.loc[m, "Points"] = self.points.loc[m, "Model points"]
        self.points.loc[m, "Manual"] = False
        self.points.loc[m, "Reason"] = ""
        self.points.loc[m, "Override scale"] = None
        self.points.loc[m, "Stale"] = False

    def rescale(self, scaling: Optional[ScalingParams] = None, base_points: Optional[str] = None,
                good_class: Optional[int] = None, round_points: Optional[bool] = None) -> "Scorecard":
        """Recompute the model's points from the stored coefficients.

        Manual overrides keep their value; if the scale changed since they were
        set they are flagged ``Stale`` (points under a different scale mean
        something else)."""
        if scaling is not None:
            self.scaling = scaling
        if base_points is not None:
            if base_points not in ("spread", "separate"):
                raise ValueError("base_points must be 'spread' or 'separate'")
            self.base_points_mode = base_points
        if good_class is not None:
            if int(good_class) not in (0, 1):
                raise ValueError("good_class must be 0 or 1")
            self.good_class = int(good_class)
        if round_points is not None:
            self.round_points = bool(round_points)
        f, off, a, s = self.scaling.factor, self.scaling.offset, self.intercept, self.direction
        n = len(self.representations)
        shift = (off + s * f * a) / n if self.base_points_mode == "spread" else 0.0
        base = off + s * f * a
        rnd = (lambda x: float(np.round(x))) if self.round_points else float
        self.base_points = rnd(base) if self.base_points_mode == "separate" else 0.0
        p = self.points
        for col, default in (("Manual", False), ("Reason", ""), ("Stale", False), ("Override scale", None)):
            if col not in p:
                p[col] = default
        for i, r in p.iterrows():
            v = r["Variable"]
            if self.representations.get(v) == "raw":
                beta = float(self.raw_terms[v]["beta"])
                self.raw_terms[v].update(points_per_unit=s * f * beta, constant=shift)
                p.at[i, "Label"] = f"{s * f * beta:+.4g} points per unit, {shift:+.4g} constant"
                continue
            beta = float(r["Coefficient"])
            x = float(r["WOE"]) if r["Representation"] == "woe" else 1.0
            model_pts = rnd(s * f * beta * x + shift)
            p.at[i, "Model points"] = model_pts
            if bool(r["Manual"]):
                p.at[i, "Stale"] = r.get("Override scale") != self.scale_signature()
            else:
                p.at[i, "Points"] = model_pts
        self.meta.update(n_variables=n, factor=f, offset=off)
        return self

    # ------------------------------------------------------------------ scoring

    def bin_groups(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Bin group of every row for each binned variable (after the
        plausibility rules). Points can then be recomputed cheaply."""
        data = apply_plausibility_rules(df, self.plausibility)
        out = {}
        for v, rep in self.representations.items():
            if v not in data.columns:
                raise KeyError(f"Column '{v}' is required by the scorecard.")
            if rep == "raw":
                x = pd.to_numeric(data[v], errors="coerce").to_numpy(dtype=float)
                if np.isnan(x).any():
                    raise ValueError(f"'{v}' is used raw (linear) and has missing values in this data.")
                out[v] = x
            else:
                out[v] = self.bundle.artifacts[v].transform_series(data[v])["group"]
        return out

    def points_from_groups(self, groups: Dict[str, np.ndarray]) -> pd.DataFrame:
        """``pts_<var>`` columns from :meth:`bin_groups` output."""
        cols = {}
        for v, rep in self.representations.items():
            if rep == "raw":
                t = self.raw_terms[v]
                cols[f"pts_{v}"] = t["points_per_unit"] * groups[v] + t["constant"]
            else:
                rows = self.points[self.points["Variable"] == v]
                lookup = dict(zip(rows["Group"].astype(int), rows["Points"].astype(float)))
                cols[f"pts_{v}"] = pd.Series(groups[v]).map(lookup).to_numpy(dtype=float)
        return pd.DataFrame(cols)

    def score(self, df: pd.DataFrame, keep: Sequence[str] = (), points_detail: bool = True) -> pd.DataFrame:
        """Score raw data: plausibility rules -> bins -> points -> score.

        Returns ``keep`` columns, ``pts_<var>`` per variable (if
        ``points_detail``), ``score``, ``pd`` (probability of bad, from the
        score via the scaling) and, when a cutoff is set, ``decision``."""
        pts = self.points_from_groups(self.bin_groups(df))
        pts.index = df.index
        out = pd.DataFrame(index=df.index)
        for c in keep:
            out[c] = df[c]
        if points_detail:
            out = pd.concat([out, pts], axis=1)
        # Same order as the generated Python/SQL: base points, then each variable.
        total = np.full(len(df), float(self.base_points if self.base_points_mode == "separate" else 0.0))
        for c in pts.columns:
            total = total + pts[c].to_numpy()
        out["score"] = total
        out["pd"] = self.pd_from_score(total)
        if self.cutoff is not None:
            out["decision"] = np.where(total >= self.cutoff, "approve", "decline")
        return out

    def pd_from_score(self, score) -> np.ndarray:
        """Probability of bad: Score = Offset + Factor * ln(odds good)."""
        z = (np.asarray(score, dtype=float) - self.scaling.offset) / self.scaling.factor
        return 1.0 / (1.0 + np.exp(z))

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> Dict[str, Any]:
        pts = self.points.astype(object).where(self.points.notna(), None)
        return {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "target": self.target,
            "good_class": self.good_class,
            "cutoff": self.cutoff,
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
        pts = pd.DataFrame(d["points"])
        for col, default in (("Manual", False), ("Reason", ""), ("Stale", False), ("Override scale", None)):
            if col not in pts:
                pts[col] = default
        pts["Manual"] = pts["Manual"].fillna(False).astype(bool)
        pts["Stale"] = pts["Stale"].fillna(False).astype(bool)
        pts["Reason"] = pts["Reason"].fillna("")
        return cls(
            target=d["target"], scaling=ScalingParams(**d["scaling"]), intercept=float(d["intercept"]),
            representations=dict(d["representations"]), points=pts,
            bundle=ScoringBundle.from_dict(d["bundle"]), base_points_mode=d["base_points_mode"],
            base_points=float(d["base_points"]), round_points=bool(d["round_points"]),
            plausibility=d.get("plausibility") or {}, raw_terms=d.get("raw_terms") or {},
            meta=d.get("meta") or {}, good_class=int(d.get("good_class", 0)),
            cutoff=d.get("cutoff"))

    def copy(self) -> "Scorecard":
        return Scorecard.from_dict(json.loads(json.dumps(_json_safe(self.to_dict()))))

    # ------------------------------------------------------------------ scoring code

    def to_python(self) -> str:
        """A standalone Python scoring script (standard library only).

        ``python scorer.py input.csv output.csv`` appends ``pts_<var>``,
        ``score``, ``pd`` and ``decision`` to every row; ``score_record(dict)``
        scores one record. It applies the plausibility rules, the bins
        (including Missing/Special choices) and the points in use, so manual
        overrides are included."""
        return _python_script(self)

    def to_sql(self, source_table: str = "input_table", dialect: str = "standard",
               include_pd: bool = True) -> str:
        """One ``SELECT`` that scores ``source_table``: all its columns plus
        ``pts_<var>``, ``score``, ``pd`` (if ``include_pd``; needs ``EXP``) and
        ``decision`` (when a cutoff is set)."""
        return _sql_script(self, source_table, dialect, include_pd)


def load_scorecard(path: str) -> Scorecard:
    """Reload a saved scorecard to rescore new data: ``load_scorecard(p).score(df)``."""
    with open(path) as fh:
        return Scorecard.from_dict(json.load(fh))


def build_scorecard(model, bundle: ScoringBundle, scaling: ScalingParams = ScalingParams(),
                    base_points: str = "spread", round_points: bool = True,
                    plausibility: Optional[Dict[str, Any]] = None, good_class: int = 0) -> Scorecard:
    """Turn a fitted :class:`~scorecard_studio.model.LogisticModel` into points.

    The table stores each bin's coefficient (and WOE), so the points can be
    rescaled later from the scorecard alone (:meth:`Scorecard.rescale`)."""
    if base_points not in ("spread", "separate"):
        raise ValueError("base_points must be 'spread' or 'separate'")
    reps = dict(model.representations)
    rows, raw_terms = [], {}
    for v, rep in reps.items():
        if rep == "raw":
            raw_terms[v] = {"beta": float(model.coef[v]), "points_per_unit": 0.0, "constant": 0.0}
            rows.append({"Variable": v, "Representation": "raw", "Group": None, "Kind": "linear",
                         "Label": "", "WOE": None, "Coefficient": float(model.coef[v]),
                         "Model points": None, "Points": None, "Manual": False, "Reason": "",
                         "Stale": False, "Override scale": None, "Count": None, "Event rate": None})
            continue
        art = bundle.artifacts[v]
        emitted = set()
        # Regular bins first, then Missing, then Special: a merged (alias) entry
        # never hides the bin that actually holds the rows.
        for b in sorted(art.bins, key=lambda x: {"regular": 0, "missing": 1, "special": 2}[x["kind"]]):
            if b["group"] in emitted:
                continue    # Missing/Special merged into another bin: scored as that bin
            emitted.add(b["group"])
            beta = model.coef[v] if rep == "woe" else model.bin_coef(v, b["group"])
            rows.append({"Variable": v, "Representation": rep, "Group": int(b["group"]), "Kind": b["kind"],
                         "Label": b["label"], "WOE": b["woe"], "Coefficient": float(beta),
                         "Model points": None, "Points": None, "Manual": False, "Reason": "",
                         "Stale": False, "Override scale": None,
                         "Count": b.get("count"), "Event rate": b.get("event_rate")})
    sub = ScoringBundle({v: bundle.artifacts[v] for v, r in reps.items() if r != "raw"}, name="scorecard_bins")
    card = Scorecard(target=model.target, scaling=scaling, intercept=float(model.intercept),
                     representations=reps, points=pd.DataFrame(rows), bundle=sub,
                     base_points_mode=base_points, round_points=round_points,
                     plausibility=plausibility or {}, raw_terms=raw_terms, good_class=good_class)
    return card.rescale()


# ═══════════════════════════════════════════════════════════════════════════
# Scoring code generators
# ═══════════════════════════════════════════════════════════════════════════

def _points_by_group(card: Scorecard, v: str) -> Dict[int, float]:
    rows = card.points[card.points["Variable"] == v]
    return {int(g): float(p) for g, p in zip(rows["Group"], rows["Points"])}


def _python_script(card: Scorecard) -> str:
    from . import __version__
    from .artifacts import _PY_HELPERS, _doc_safe
    s = card.scaling
    binned = [v for v, r in card.representations.items() if r != "raw"]
    raw = {v: (card.raw_terms[v]["points_per_unit"], card.raw_terms[v]["constant"])
           for v, r in card.representations.items() if r == "raw"}
    types = {}
    for v in binned:
        art = card.bundle.artifacts[v]
        types[v] = "num" if (art.dtype == "numerical" or art.value_type == "numeric") else (
            "bool" if art.value_type == "boolean" else "str")
    for v in raw:
        types[v] = "num"
    rules = (card.plausibility or {}).get("rules") or {}
    L = [f'"""Scorecard scoring script, generated by scorecard_studio {__version__}.',
         "",
         f"Target: {_doc_safe(card.target)} (good = {card.good_class}).  Scale: {_doc_safe(card.scale_signature())}.",
         f"Cutoff: {card.cutoff if card.cutoff is not None else 'none'} (approve if score >= cutoff).",
         "Manual overrides are included. Standard library only.",
         "",
         "Usage:",
         "    python scorer.py input.csv output.csv",
         "    from scorer import score_record; score_record({...})",
         '"""',
         "import csv", "import math", "import sys", "",
         _PY_HELPERS, "",
         f"PLAUSIBILITY = {repr({k: tuple(v) for k, v in rules.items()})}",
         f"PLAUSIBILITY_ACTION = {(card.plausibility or {}).get('action', 'special')!r}",
         f"PLAUSIBILITY_CODE = {float((card.plausibility or {}).get('code', -99999))!r}",
         f"TYPES = {types!r}",
         f"POINTS = {{{', '.join(f'{v!r}: {_points_by_group(card, v)!r}' for v in binned)}}}",
         f"RAW = {raw!r}",
         f"BASE_POINTS = {float(card.base_points if card.base_points_mode == 'separate' else 0.0)!r}",
         f"OFFSET = {s.offset!r}",
         f"FACTOR = {s.factor!r}",
         f"CUTOFF = {card.cutoff!r}",
         "", "",
         "def _parse(var, value):",
         '    """CSV text -> typed value (empty -> missing)."""',
         "    if isinstance(value, str):",
         "        if value.strip() == '':",
         "            return None",
         "        if TYPES.get(var) == 'num':",
         "            try:",
         "                return float(value)",
         "            except ValueError:",
         "                return value",
         "        if TYPES.get(var) == 'bool' and value in ('True', 'False', 'true', 'false', '1', '0'):",
         "            return value in ('True', 'true', '1')",
         "    return value",
         "", "",
         "def _plausible(var, v):",
         '    """The intake plausibility rules: impossible values -> special code or missing."""',
         "    if var not in PLAUSIBILITY or _is_missing(v):",
         "        return v",
         "    try:",
         "        x = float(v)",
         "    except (TypeError, ValueError):",
         "        return v",
         "    lo, hi = PLAUSIBILITY[var]",
         "    if PLAUSIBILITY_ACTION == 'special' and x == PLAUSIBILITY_CODE:",
         "        return v",
         "    if (lo is not None and x < lo) or (hi is not None and x > hi):",
         "        if PLAUSIBILITY_ACTION == 'special':",
         "            return PLAUSIBILITY_CODE",
         "        if PLAUSIBILITY_ACTION == 'missing':",
         "            return None",
         "    return v",
         ""]
    fn_names = {}
    for i, v in enumerate(binned):
        fn = f"_bin_{i}"
        fn_names[v] = fn
        L += ["", card.bundle.artifacts[v].to_python(func_name=fn), ""]
    L += ["",
          f"BINNERS = {{{', '.join(f'{v!r}: {fn}' for v, fn in fn_names.items())}}}",
          "", "",
          "def score_record(record):",
          '    """{"column": value, ...} -> {"pts_<var>": ..., "score": ..., "pd": ..., "decision": ...}"""',
          "    out = {}",
          "    total = BASE_POINTS",
          "    for var, fn in BINNERS.items():",
          "        v = _plausible(var, _parse(var, record.get(var)))",
          "        p = POINTS[var][fn(v)['group']]",
          "        out['pts_' + var] = p",
          "        total += p",
          "    for var, (per_unit, constant) in RAW.items():",
          "        v = _plausible(var, _parse(var, record.get(var)))",
          "        if _is_missing(v):",
          "            raise ValueError(var + ' is used linearly and cannot be missing')",
          "        p = per_unit * float(v) + constant",
          "        out['pts_' + var] = p",
          "        total += p",
          "    out['score'] = total",
          "    out['pd'] = 1.0 / (1.0 + math.exp((total - OFFSET) / FACTOR))",
          "    if CUTOFF is not None:",
          "        out['decision'] = 'approve' if total >= CUTOFF else 'decline'",
          "    return out",
          "", "",
          "def score_csv(src, dst):",
          "    with open(src, newline='', encoding='utf-8') as fi, open(dst, 'w', newline='', encoding='utf-8') as fo:",
          "        reader = csv.DictReader(fi)",
          "        writer = None",
          "        for row in reader:",
          "            res = score_record(row)",
          "            if writer is None:",
          "                writer = csv.DictWriter(fo, fieldnames=list(row) + list(res))",
          "                writer.writeheader()",
          "            writer.writerow({**row, **res})",
          "", "",
          "if __name__ == '__main__':",
          "    if len(sys.argv) != 3:",
          "        sys.exit('usage: python scorer.py input.csv output.csv')",
          "    score_csv(sys.argv[1], sys.argv[2])",
          ""]
    return "\n".join(L)


def _sql_num(x: float) -> str:
    f = float(x)
    return str(int(f)) if f.is_integer() else repr(f)


def _sql_script(card: Scorecard, source_table: str, dialect: str, include_pd: bool) -> str:
    from .artifacts import _check_dialect, _sql_comment_safe, quote_ident
    _check_dialect(dialect)
    rules = (card.plausibility or {}).get("rules") or {}
    action = (card.plausibility or {}).get("action", "special")
    code = float((card.plausibility or {}).get("code", -99999))
    q = lambda n: quote_ident(n, dialect)

    def clean(v: str) -> str:
        col = "src." + q(v)
        if v not in rules or action == "flag":
            return col
        lo, hi = rules[v]
        conds = []
        if lo is not None:
            conds.append(f"{col} < {_sql_num(lo)}")
        if hi is not None:
            conds.append(f"{col} > {_sql_num(hi)}")
        guard = f" AND {col} <> {_sql_num(code)}" if action == "special" else ""
        repl = _sql_num(code) if action == "special" else "NULL"
        return f"(CASE WHEN {col} IS NOT NULL{guard} AND ({' OR '.join(conds)}) THEN {repl} ELSE {col} END)"

    exprs = []
    for v, rep in card.representations.items():
        x = clean(v)
        name = q(f"pts_{v}")
        if rep == "raw":
            t = card.raw_terms[v]
            exprs.append(f"    ({_sql_num(t['points_per_unit'])} * {x} + {_sql_num(t['constant'])}) AS {name}")
            continue
        art = card.bundle.artifacts[v]
        pts = _points_by_group(card, v)
        P = lambda b: _sql_num(pts[int(b["group"])])
        lines = ["    CASE", f"        WHEN {x} IS NULL THEN {P(art.missing)}"]
        if art.special:
            codes = ", ".join(art._sql_value(c, dialect) for c in art.special_codes)
            lines.append(f"        WHEN {x} IN ({codes}) THEN {P(art.special)}")
        if art.dtype == "numerical":
            for b in art.regular[:-1]:
                lines.append(f"        WHEN {x} <= {repr(float(b['upper']))} THEN {P(b)}")
            lines.append(f"        ELSE {P(art.regular[-1])}")
        else:
            for b in art.regular:
                cats = ", ".join(art._sql_value(c, dialect) for c in b["categories"])
                lines.append(f"        WHEN {x} IN ({cats}) THEN {P(b)}")
            lines.append(f"        ELSE {P(art.missing)}  -- unseen category")
        lines.append(f"    END AS {name}")
        exprs.append(f"    -- {_sql_comment_safe(v)} ({rep})\n" + "\n".join(lines))

    base = float(card.base_points if card.base_points_mode == "separate" else 0.0)
    total = " + ".join([_sql_num(base)] + [q(f"pts_{v}") for v in card.representations])
    s = card.scaling
    header = [f"-- Scorecard scoring query, generated by scorecard_studio.",
              f"-- Target: {_sql_comment_safe(card.target)} (good = {card.good_class}). "
              f"Scale: {_sql_comment_safe(card.scale_signature())}.",
              f"-- Cutoff: {card.cutoff if card.cutoff is not None else 'none'} (approve if score >= cutoff). "
              "Manual overrides included."]
    final_cols = ["scored.*"]
    if include_pd:
        final_cols.append(f"1.0 / (1.0 + EXP((scored.{q('score')} - {repr(s.offset)}) / {repr(s.factor)})) AS {q('pd')}")
    if card.cutoff is not None:
        final_cols.append(f"CASE WHEN scored.{q('score')} >= {_sql_num(card.cutoff)} THEN 'approve' "
                          f"ELSE 'decline' END AS {q('decision')}")
    return "\n".join(header + [
        "WITH points AS (",
        "  SELECT",
        "    src.*,",
        ",\n".join(exprs),
        f"  FROM {source_table} AS src",
        "), scored AS (",
        f"  SELECT points.*, {total} AS {q('score')}",
        "  FROM points",
        ")",
        "SELECT " + ", ".join(final_cols),
        "FROM scored",
    ])


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
