"""
State behind the interactive binning app.

Everything the UI can do is a method here, so it can be tested without a
browser and driven from the notebook when the iframe is blocked:

* ``fit``        re-run optimal binning with other settings
* ``set_cutoffs``/``split``/``merge``   edit numerical bins
* ``set_categories``/``merge``          regroup categorical bins
* ``set_included``  choose the variables that go to the output dataset
* ``build_output``  write ``opt_<var>`` (bin) and ``woe_<var>`` for every sample

Bins are always fit and edited on **Train only**; the output dataset applies
the same bins to Test and OOT through the scoring artifact, i.e. the exact
code path the exported ``scorer.py``/SQL are tested against.

Every change is saved to ``binning_config.json`` (atomic write), so a Colab
disconnect never loses work. ``resume`` reloads it only when it was saved for
the same Train sample (row count, events and a hash of IDs and target).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..binning import MISSING_GROUP, SPECIAL_GROUP, BinningEngine, BinningResult
from ..dtypes import DEFAULT_MAX_CATEGORIES, detect_dtype, series_to_category_str
from ..metrics import interpret_iv

APP_SCHEMA_VERSION = 1
CONFIG_FILE = "binning_config.json"

#: Screening statuses that can be added back from the app. ``review`` and
#: ``excluded`` need a decision in the config cell, where it is documented.
ADDABLE = ("selected", "forced", "low_iv")
NOT_BINNABLE = ("id_like", "constant", "fit_error")

FIT_KEYS = ("max_bins", "monotonic", "divergence", "min_bin_size", "min_bin_n_event", "cat_cutoff")


class AppError(ValueError):
    """A user action that can't be applied; the message is shown in the UI."""


@dataclass
class VarInfo:
    name: str
    dtype: str
    status: str            # screening status, or "candidate" without screening
    screen_iv: Optional[float]
    reason: str
    included: bool
    edited: bool = False

    @property
    def addable(self) -> bool:
        return self.status in ADDABLE or self.status == "candidate"


class BinningSession:
    """Interactive binning state for one Train sample.

    Parameters
    ----------
    train : the Train sample only.
    target : 0/1 target column.
    full, sample : optional full table and its ``"train"/"test"/"oot"`` labels;
        used only by :meth:`build_output` (never to fit or edit bins).
    screen : a :class:`~scorecard_studio.screening.ScreeningResult`; its
        shortlist is included from the start and its fits are reused.
    variables : without ``screen``, the candidate columns (default: all).
    fit_params : default optbinning settings (the preset's ``fit_params``).
    save_dir : folder for ``binning_config.json`` and the output files.
    """

    def __init__(self, train: pd.DataFrame, target: str, *,
                 full: Optional[pd.DataFrame] = None, sample: Optional[pd.Series] = None,
                 id_col: Optional[str] = None, date_col: Optional[str] = None,
                 screen=None, variables: Optional[Iterable[str]] = None,
                 special_codes: Optional[Dict[str, List[Any]]] = None,
                 fit_params: Optional[Dict[str, Any]] = None,
                 save_dir: Optional[str] = None,
                 max_categories: int = DEFAULT_MAX_CATEGORIES):
        if (full is None) != (sample is None):
            raise ValueError("Pass both `full` and `sample`, or neither.")
        if sample is not None and not sample.index.equals(full.index):
            raise ValueError("`sample` must be aligned with `full`.")
        self.train = train
        self.target = target
        self.full = full
        self.sample = sample
        self.id_col = id_col
        self.date_col = date_col
        self.special_codes = {k: list(v) for k, v in (special_codes or {}).items()}
        self.fit_params = {k: v for k, v in (fit_params or {}).items() if k in FIT_KEYS}
        self.save_dir = save_dir
        self.max_categories = max_categories
        self.lock = threading.RLock()
        self.last_saved: Optional[str] = None
        self.last_output: Optional[Dict[str, Any]] = None

        roles = {target, id_col, date_col} - {None}
        self.engine = BinningEngine(train, target, special_codes=self.special_codes,
                                    exclude=roles - {target}, max_categories=max_categories)
        self.vars: Dict[str, VarInfo] = {}
        if screen is not None:
            for r in screen.table.itertuples():
                if r.status in NOT_BINNABLE or r.type not in ("numerical", "categorical"):
                    continue
                self.vars[r.variable] = VarInfo(
                    name=r.variable, dtype=r.type, status=r.status,
                    screen_iv=None if pd.isna(r.iv) else float(r.iv), reason=r.reason,
                    included=r.status in ("selected", "forced"))
            # Reuse the screening fits: same engine, same settings, same Train.
            cfg = screen.engine.export_config()
            cfg["variables"] = {v: c for v, c in cfg["variables"].items() if v in self.vars}
            self.engine.import_config(cfg)
        else:
            cols = list(variables) if variables is not None else [c for c in train.columns if c not in roles]
            for c in cols:
                if c not in train.columns:
                    raise ValueError(f"Variable {c!r} not in train.")
                dt = detect_dtype(train[c], max_categories)
                if dt not in ("numerical", "categorical"):
                    continue
                self.vars[c] = VarInfo(name=c, dtype=dt, status="candidate", screen_iv=None,
                                       reason="", included=True)
        if not self.vars:
            raise ValueError("No variables to bin.")

    # ══════════════════════════════════════════════════════════════════
    # Reading
    # ══════════════════════════════════════════════════════════════════

    @property
    def included(self) -> List[str]:
        return [v.name for v in self.vars.values() if v.included]

    def result(self, name: str) -> BinningResult:
        """Fitted result, fitting with the default settings on first access."""
        info = self._var(name)
        r = self.engine.get_result(name)
        if r is None:
            try:
                r = self.engine.fit(name, dtype=info.dtype, **self.fit_params)
            except Exception as exc:  # noqa: BLE001
                raise AppError(f"Could not bin '{name}': {exc}") from exc
        return r

    def state(self) -> Dict[str, Any]:
        y = self.train[self.target]
        vars_out = []
        for v in self.vars.values():
            r = self.engine.get_result(v.name)
            vars_out.append({
                "name": v.name, "dtype": v.dtype, "status": v.status, "reason": v.reason,
                "screen_iv": _num(v.screen_iv), "iv": _num(r.iv) if r else None,
                "included": v.included, "addable": v.addable, "edited": v.edited,
                "n_warnings": len(_warnings(r, self._min_share(r))) if r else 0,
            })
        return {
            "target": self.target, "rows": int(len(self.train)), "events": int(y.sum()),
            "event_rate": float(y.mean()), "fit_defaults": self.fit_params,
            "variables": vars_out, "included": self.included,
            "has_full": self.full is not None, "save_dir": self.save_dir,
            "last_saved": self.last_saved, "last_output": self.last_output,
        }

    def variable_payload(self, name: str) -> Dict[str, Any]:
        """Everything the UI needs to draw one variable."""
        with self.lock:
            info = self._var(name)
            r = self.result(name)
            out = serialize_result(r)
            out.update(included=info.included, edited=info.edited, status=info.status,
                       screen_iv=_num(info.screen_iv), reason=info.reason, addable=info.addable,
                       flags=_warnings(r, self._min_share(r)))
            if r.dtype == "numerical":
                out["hist"] = self._histogram(name, r)
            else:
                out["categories"] = self._category_table(name, r)
            return out

    # ══════════════════════════════════════════════════════════════════
    # Editing (each call saves)
    # ══════════════════════════════════════════════════════════════════

    def fit(self, name: str, **params) -> Dict[str, Any]:
        """Optimal binning with ``params`` overriding the defaults."""
        with self.lock:
            info = self._var(name)
            p = {**self.fit_params, **{k: v for k, v in params.items() if k in FIT_KEYS}}
            p = _clean_fit_params(p)
            try:
                self.engine.fit(name, dtype=info.dtype, **p)
            except Exception as exc:  # noqa: BLE001
                raise AppError(f"Binning failed: {exc}") from exc
            info.edited = p != _clean_fit_params(self.fit_params)
            return self._changed(name)

    def reset(self, name: str) -> Dict[str, Any]:
        """Back to the automatic binning with the default settings."""
        with self.lock:
            info = self._var(name)
            try:
                self.engine.fit(name, dtype=info.dtype, **self.fit_params)
            except Exception as exc:  # noqa: BLE001
                raise AppError(f"Binning failed: {exc}") from exc
            info.edited = False
            return self._changed(name)

    def set_cutoffs(self, name: str, cutoffs: Sequence[float]) -> Dict[str, Any]:
        with self.lock:
            self._require_dtype(name, "numerical")
            try:
                cuts = sorted({float(c) for c in cutoffs})
            except (TypeError, ValueError) as exc:
                raise AppError("Cutoffs must be numbers.") from exc
            if any(not math.isfinite(c) for c in cuts):
                raise AppError("Cutoffs must be finite.")
            self.engine.adjust_cutoffs(name, cuts)
            self._var(name).edited = True
            return self._changed(name)

    def split(self, name: str, group: int, at: Optional[float] = None) -> Dict[str, Any]:
        """Split a numerical bin at ``at``, or at the bin's median when omitted."""
        with self.lock:
            r = self._require_dtype(name, "numerical")
            b = _regular_bin(r, group)
            if at is None:
                at = self._median_cut(name, r, b)
            at = float(at)
            lo = -math.inf if b.lower is None else b.lower
            hi = math.inf if b.upper is None else b.upper
            if not lo < at < hi:
                raise AppError(f"Split point {at:g} is outside bin '{b.label}'.")
            return self.set_cutoffs(name, list(r.cutoffs) + [at])

    def merge(self, name: str, groups: Sequence[int]) -> Dict[str, Any]:
        """Merge regular bins. Numerical: must be adjacent. Categorical: any."""
        with self.lock:
            r = self.result(name)
            gs = sorted({int(g) for g in groups})
            if len(gs) < 2:
                raise AppError("Select at least two bins to merge.")
            for g in gs:
                _regular_bin(r, g)  # Missing/Special cannot be merged
            if r.dtype == "numerical":
                if gs != list(range(gs[0], gs[-1] + 1)):
                    raise AppError("Only adjacent bins can be merged.")
                # cut i (0-based) separates groups i+1 and i+2
                drop = set(range(gs[0] - 1, gs[-1] - 1))
                return self.set_cutoffs(name, [c for i, c in enumerate(r.cutoffs) if i not in drop])
            assignments = {}
            for i, grp in enumerate(r.cat_groups or [], start=1):
                for c in grp:
                    assignments[c] = gs[0] if i in gs else i
            return self.set_categories(name, assignments)

    def set_categories(self, name: str, assignments: Dict[str, int]) -> Dict[str, Any]:
        """Regroup a categorical variable: ``{category: group id}`` for every category."""
        with self.lock:
            r = self._require_dtype(name, "categorical")
            known = {c for g in (r.cat_groups or []) for c in g}
            got = {str(k): int(v) for k, v in assignments.items()}
            missing = sorted(known - set(got))
            if missing:
                raise AppError(f"Assign every category to a group; unassigned: {missing[:10]}")
            unknown = sorted(set(got) - known)
            if unknown:
                raise AppError(f"Unknown categories: {unknown[:10]}")
            self.engine.merge_categories(name, got)
            self._var(name).edited = True
            return self._changed(name)

    def set_included(self, name: str, included: bool) -> Dict[str, Any]:
        with self.lock:
            info = self._var(name)
            if included and not info.addable:
                raise AppError(
                    f"'{name}' is '{info.status}' at screening. Decide it in the config cell "
                    "(FORCE_INCLUDE or EXCLUDE) so the decision is documented, then re-run.")
            if included:
                self.result(name)  # must be binnable
            info.included = bool(included)
            self.save()
            return {"state": self.state()}

    # ══════════════════════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════════════════════

    def fingerprint(self) -> Dict[str, Any]:
        cols = [c for c in (self.id_col, self.target) if c]
        h = int(pd.util.hash_pandas_object(self.train[cols], index=False).sum() % (2 ** 61))
        return {"rows": int(len(self.train)), "events": int(self.train[self.target].sum()),
                "target": self.target, "hash": str(h)}

    def to_config(self) -> Dict[str, Any]:
        cfg = self.engine.export_config()
        cfg["variables"] = {v: c for v, c in cfg["variables"].items() if v in self.vars}
        return {
            "app_schema_version": APP_SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": self.fingerprint(),
            "included": self.included,
            "edited": sorted(v.name for v in self.vars.values() if v.edited),
            "fit_defaults": self.fit_params,
            "engine": cfg,
        }

    def save(self) -> Optional[str]:
        """Atomic write of ``binning_config.json`` to ``save_dir`` (if set)."""
        if not self.save_dir:
            return None
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, CONFIG_FILE)
        fd, tmp = tempfile.mkstemp(dir=self.save_dir, prefix=".binning_", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(_json_safe(self.to_config()), fh, indent=2)
        os.replace(tmp, path)
        self.last_saved = datetime.now().strftime("%H:%M:%S")
        return path

    def load(self, config: Dict[str, Any], force: bool = False) -> List[str]:
        """Restore bins and choices from :meth:`to_config` output.

        Refuses a config saved for a different Train sample unless ``force``:
        bins edited on other rows are not the bins you think they are.
        Returns the variables that could not be restored.
        """
        with self.lock:
            if config.get("app_schema_version") != APP_SCHEMA_VERSION:
                raise AppError(f"Unsupported binning config version {config.get('app_schema_version')!r}.")
            if not force and config.get("fingerprint") != self.fingerprint():
                raise AppError(
                    "This binning config was saved for a different Train sample "
                    f"({config.get('fingerprint')} vs {self.fingerprint()}). "
                    "Use force=True only if you are sure.")
            eng = dict(config["engine"])
            wanted = {v: c for v, c in eng["variables"].items() if v in self.vars}
            skipped = sorted(set(eng["variables"]) - set(wanted))
            self.engine.import_config({**eng, "variables": wanted})
            for v in self.vars.values():
                v.edited = v.name in set(config.get("edited", []))
            inc = set(config.get("included", []))
            for v in self.vars.values():
                v.included = v.name in inc and (v.addable or v.included)
            return skipped

    def load_file(self, path: str, force: bool = False) -> List[str]:
        with open(path) as fh:
            return self.load(json.load(fh), force=force)

    # ══════════════════════════════════════════════════════════════════
    # Output dataset
    # ══════════════════════════════════════════════════════════════════

    def build_output(self, variables: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Model-ready table for every sample.

        Columns: ID, ``sample``, target, date (if any), then per variable the
        original column, ``opt_<var>`` (bin label, an ordered categorical in bin
        order: regular bins, Special, Missing) and ``woe_<var>``.
        """
        with self.lock:
            names = list(variables) if variables is not None else self.included
            if not names:
                raise AppError("No variables selected for the output dataset.")
            base = self.full if self.full is not None else self.train
            sample = self.sample if self.sample is not None else pd.Series("train", index=base.index)
            clash = [c for v in names for c in (f"opt_{v}", f"woe_{v}") if c in base.columns]
            if clash:
                raise AppError(f"Output column names already exist in the data: {clash[:5]}")

            cols: Dict[str, Any] = {}
            if self.id_col:
                cols[self.id_col] = base[self.id_col]
            cols["sample"] = sample.astype(str)
            cols[self.target] = base[self.target]
            if self.date_col:
                cols[self.date_col] = base[self.date_col]
            for v in names:
                self.result(v)
                art = self.engine.get_scoring_artifact(v)
                res = art.transform_series(base[v])
                order, names_by_group = _ordered_labels(art)
                cols[v] = base[v]
                cols[f"opt_{v}"] = pd.Categorical([names_by_group[g] for g in res["group"]],
                                                  categories=order, ordered=True)
                cols[f"woe_{v}"] = res["woe"].astype("float64")
            return pd.DataFrame(cols, index=base.index)

    def save_output(self, variables: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Build the output dataset and write it, the scoring bundle and the
        binning tables to ``save_dir``."""
        with self.lock:
            if not self.save_dir:
                raise AppError("No save folder set for this session.")
            names = list(variables) if variables is not None else self.included
            out = self.build_output(names)
            os.makedirs(self.save_dir, exist_ok=True)
            paths = {
                "dataset": os.path.join(self.save_dir, "model_dataset.parquet"),
                "bundle": os.path.join(self.save_dir, "bundle.json"),
                "tables": os.path.join(self.save_dir, "binning_tables.csv"),
            }
            out.to_parquet(paths["dataset"], index=False)
            self.engine.build_scoring_bundle(names, name="binning_bundle").save_json(paths["bundle"])
            self.binning_tables(names).to_csv(paths["tables"], index=False)
            self.save()
            counts = out["sample"].value_counts().to_dict()
            self.last_output = {"rows": int(len(out)), "columns": int(out.shape[1]),
                                "variables": names, "samples": {k: int(v) for k, v in counts.items()},
                                "paths": paths, "at": datetime.now().strftime("%H:%M:%S")}
            return self.last_output

    def binning_tables(self, variables: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """All bins of the chosen variables in one table (for model documentation)."""
        names = list(variables) if variables is not None else self.included
        frames = []
        for v in names:
            t = self.result(v).summary()
            t.insert(0, "Variable", v)
            frames.append(t)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════
    # Internals
    # ══════════════════════════════════════════════════════════════════

    def _var(self, name: str) -> VarInfo:
        if name not in self.vars:
            raise AppError(f"Unknown variable '{name}'.")
        return self.vars[name]

    def _require_dtype(self, name: str, dtype: str) -> BinningResult:
        r = self.result(name)
        if r.dtype != dtype:
            raise AppError(f"'{name}' is {r.dtype}; this action is for {dtype} variables.")
        return r

    def _changed(self, name: str) -> Dict[str, Any]:
        self.save()
        return {"variable": self.variable_payload(name), "state": self.state()}

    def _min_share(self, r: Optional[BinningResult]) -> float:
        if r is None:
            return 0.0
        return float(r.settings.get("min_bin_size") or self.fit_params.get("min_bin_size") or 0.0)

    def _clean_values(self, name: str, r: BinningResult) -> np.ndarray:
        x = self.train[name]
        miss = x.isna().to_numpy()
        vals = pd.to_numeric(x, errors="coerce").astype("float64").to_numpy()
        special = np.isin(vals, [float(s) for s in r.special_codes]) if r.special_codes else np.zeros(len(vals), bool)
        return vals[~miss & ~special]

    def _median_cut(self, name: str, r: BinningResult, b) -> float:
        vals = self._clean_values(name, r)
        lo = -np.inf if b.lower is None else b.lower
        hi = np.inf if b.upper is None else b.upper
        v = np.sort(vals[(vals > lo) & (vals <= hi)])
        u = np.unique(v)
        if len(u) < 2:
            raise AppError(f"Bin '{b.label}' has a single distinct value; it can't be split.")
        k = int(np.searchsorted(u, np.median(v), side="left"))
        k = min(max(k, 0), len(u) - 2)          # keep both sides non-empty
        return float(u[k])                       # (lower, upper]: values <= u[k] go left

    def _histogram(self, name: str, r: BinningResult) -> Dict[str, Any]:
        """Fine-grained distribution of the clean Train values, for drawing and
        dragging cutoffs in value space. One bucket per value for small-range
        integers; otherwise 40 equal-width buckets over the 1st-99th percentile."""
        vals = self._clean_values(name, r)
        y = self.train[self.target].to_numpy()
        x = self.train[name]
        miss = x.isna().to_numpy()
        allv = pd.to_numeric(x, errors="coerce").astype("float64").to_numpy()
        special = np.isin(allv, [float(s) for s in r.special_codes]) if r.special_codes else np.zeros(len(allv), bool)
        yc = y[~miss & ~special]
        if len(vals) == 0:
            return {"edges": [], "counts": [], "events": [], "integer": False, "resolution": 1.0}
        integer = bool(np.all(np.mod(vals, 1) == 0))
        vmin, vmax = float(vals.min()), float(vals.max())
        lo, hi = (float(np.quantile(vals, 0.01)), float(np.quantile(vals, 0.99)))
        if integer and hi - lo <= 60:
            edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
        else:
            if hi <= lo:
                lo, hi = vmin, vmax if vmax > vmin else vmin + 1
            edges = np.linspace(lo, hi, 41)
        inside = (vals >= edges[0]) & (vals <= edges[-1])
        idx = np.clip(np.searchsorted(edges, vals[inside], side="right") - 1, 0, len(edges) - 2)
        counts = np.bincount(idx, minlength=len(edges) - 1)
        events = np.bincount(idx, weights=yc[inside], minlength=len(edges) - 1)
        span = max(hi - lo, 1e-12)
        resolution = 1.0 if integer else 10 ** (math.floor(math.log10(span)) - 2)
        return {
            "edges": [float(e) for e in edges], "counts": counts.astype(int).tolist(),
            "events": events.astype(int).tolist(), "integer": integer, "resolution": resolution,
            "min": vmin, "max": vmax,
            "below": int((vals < edges[0]).sum()), "above": int((vals > edges[-1]).sum()),
        }

    def _category_table(self, name: str, r: BinningResult) -> List[Dict[str, Any]]:
        """Per category (canonical string): count, events, event rate, group."""
        cats = series_to_category_str(self.train[name])
        special = set(str(s) for s in series_to_category_str(pd.Series(r.special_codes, dtype=object)).dropna())
        df = pd.DataFrame({"c": cats, "y": self.train[self.target].to_numpy()})
        df = df[df["c"].notna() & ~df["c"].isin(special)]
        g = df.groupby("c")["y"].agg(["size", "sum"])
        group_of = {c: i for i, grp in enumerate(r.cat_groups or [], start=1) for c in grp}
        out = [{"value": c, "count": int(row["size"]), "events": int(row["sum"]),
                "event_rate": float(row["sum"] / row["size"]), "group": group_of.get(c)}
               for c, row in g.iterrows()]
        return sorted(out, key=lambda d: d["event_rate"])


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def serialize_result(r: BinningResult) -> Dict[str, Any]:
    total = sum(b.count for b in r.bins) or 1
    return {
        "variable": r.variable, "dtype": r.dtype, "iv": _num(r.iv),
        "iv_band": interpret_iv(r.iv), "gini": _num(r.gini), "ks": _num(r.ks),
        "is_monotonic": r.is_monotonic, "monotonic_direction": r.monotonic_direction,
        "cutoffs": [float(c) for c in r.cutoffs], "cat_groups": r.cat_groups,
        "special_codes": r.special_codes, "settings": r.settings, "warnings": r.warnings,
        "bins": [{
            "label": b.label, "group": b.group, "kind": b.kind,
            "lower": _num(b.lower), "upper": _num(b.upper), "categories": b.categories,
            "count": b.count, "share": b.count / total, "event_count": b.event_count,
            "non_event_count": b.non_event_count, "event_rate": _num(b.event_rate),
            "woe": _num(b.woe), "iv_contribution": _num(b.iv_contribution),
        } for b in r.bins],
    }


def _warnings(r: BinningResult, min_share: float) -> List[Dict[str, Any]]:
    """Per-bin flags the UI shows next to the table."""
    flags = []
    total = sum(b.count for b in r.bins) or 1
    for b in r.bins:
        if b.kind == "regular" and b.count == 0:
            flags.append({"group": b.group, "level": "error", "text": f"Bin '{b.label}' is empty."})
            continue
        if b.count and (b.event_count == 0 or b.non_event_count == 0):
            what = "no events" if b.event_count == 0 else "no non-events"
            fix = ("Merge it with a neighbour." if b.kind == "regular" else
                   f"Its WOE is smoothed from {b.count} rows; Missing/Special bins can't be merged, "
                   "so consider routing these values elsewhere in the config (e.g. "
                   "IMPLAUSIBLE_ACTION='missing').")
            flags.append({"group": b.group, "level": "error" if b.kind == "regular" else "warn",
                          "text": f"Bin '{b.label}' has {what}. {fix}"})
        if b.kind == "regular" and min_share and b.count / total < min_share:
            flags.append({"group": b.group, "level": "warn",
                          "text": f"Bin '{b.label}' holds {b.count / total:.1%} of rows (< {min_share:.0%})."})
    if r.dtype == "numerical" and not r.is_monotonic:
        flags.append({"group": None, "level": "warn",
                      "text": "WOE is not monotonic. Keep it only if the shape has a business explanation."})
    return flags


def _regular_bin(r: BinningResult, group: int):
    for b in r.bins:
        if b.group == int(group):
            if b.kind != "regular":
                raise AppError(f"'{b.label}' is a fixed bin; Missing and Special can't be merged or split.")
            return b
    raise AppError(f"No bin with group {group}.")


def _ordered_labels(art):
    """Unique display labels in bin order: regular groups, Special, Missing."""
    table = art.regular + ([art.special] if art.special else []) + [art.missing]
    seen, order, by_group = set(), [], {}
    for b in table:
        label = b["label"]
        if label in seen:
            label = f"{label} [group {b['group']}]"
        seen.add(label)
        order.append(label)
        by_group[b["group"]] = label
    return order, by_group


def _clean_fit_params(p: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(p)
    if "max_bins" in out and out["max_bins"] is not None:
        out["max_bins"] = int(out["max_bins"])
    if "min_bin_size" in out and out["min_bin_size"] is not None:
        out["min_bin_size"] = float(out["min_bin_size"])
    if out.get("min_bin_n_event") in ("", 0, "0"):
        out["min_bin_n_event"] = None
    elif out.get("min_bin_n_event") is not None:
        out["min_bin_n_event"] = int(out["min_bin_n_event"])
    if out.get("cat_cutoff") in ("",):
        out["cat_cutoff"] = None
    elif out.get("cat_cutoff") is not None:
        out["cat_cutoff"] = float(out["cat_cutoff"])
    return out


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if math.isfinite(f) else None


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _num(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


__all__ = ["BinningSession", "AppError", "serialize_result", "MISSING_GROUP", "SPECIAL_GROUP"]
