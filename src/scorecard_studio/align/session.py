"""
State behind the scorecard alignment app.

Works from a :class:`~scorecard_studio.scorecard.Scorecard` and the scored
rows (all samples). Bins never change here, so every row's bin is computed
once and scores are recomputed from the points table after each edit: what
the KPIs, charts, final table and exported code show is the same scorecard.

* settings: good class, PDO / base score / base odds, base points mode,
  cost of a bad, benefit of a good, the sample used to set the cutoff
* cutoff: max profit (suggested), target approval rate, target bad rate, manual
* points: manual overrides with a reason, kept through rescaling (flagged)
* finalize: the final scored table and ``final_scorecard.json``
* export: standalone Python or SQL scoring code

Every change is saved to ``alignment_state.json`` (atomic write).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..scaling import ScalingParams
from ..scorecard import Scorecard
from ..strategy import (
    break_even_score,
    choose_cutoff,
    cutoff_curve,
    discrimination,
    gains_table,
    kpis,
    realized_scale,
    roc_points,
    score_histogram,
    score_psi,
)

STATE_FILE = "alignment_state.json"
MODES = ("profit", "approval", "bad_rate", "manual")


class AlignError(ValueError):
    """A user action that can't be applied; the message is shown in the UI."""


class AlignSession:
    """``data`` is the scored dataset (``dataset_name``); ``datasets`` adds
    other tables with the same raw columns (``{name: df}`` or
    ``{name: (df, sample)}``) that the app can switch to, e.g. a newer vintage
    or a through-the-door file. Any 0/1 column of the chosen table can be the
    target (e.g. an alternative default definition)."""

    def __init__(self, card: Scorecard, data: pd.DataFrame, target: str, *,
                 sample: Optional[pd.Series] = None, id_col: Optional[str] = None,
                 save_dir: Optional[str] = None, cost_bad: float = 1000.0, benefit_good: float = 100.0,
                 strategy_sample: Optional[str] = None, dataset_name: str = "model dataset",
                 datasets: Optional[Dict[str, Any]] = None):
        self.card = card.copy()
        self.id_col = id_col
        self.save_dir = save_dir
        self.lock = threading.RLock()
        name = dataset_name or "model dataset"
        self.datasets: Dict[str, Any] = {name: (data, sample)}
        for k, v in (datasets or {}).items():
            df, smp = v if isinstance(v, tuple) else (v, None)
            if smp is None and "sample" in df.columns:
                smp = df["sample"]
            self.datasets[str(k)] = (df, smp)
        self.settings: Dict[str, Any] = {
            "cost_bad": float(cost_bad), "benefit_good": float(benefit_good),
            "strategy_sample": strategy_sample, "mode": "profit", "value": None,
        }
        self.last_saved: Optional[str] = None
        self.final: Optional[Dict[str, Any]] = None
        self._use(name, target)

    # ══════════════════════════════════════════════════════════════════
    # Dataset and target
    # ══════════════════════════════════════════════════════════════════

    def _use(self, name: str, target: str) -> None:
        if name not in self.datasets:
            raise AlignError(f"Unknown dataset '{name}'.")
        data, sample = self.datasets[name]
        if target not in data.columns:
            raise AlignError(f"Target '{target}' is not a column of '{name}'.")
        y = _binary_or_none(data[target])
        if y is None:
            raise AlignError(f"'{target}' is not a 0/1 column (missing values allowed for unknown outcomes).")
        try:
            groups = self.card.bin_groups(data)            # bins never change in this app
        except (KeyError, ValueError) as exc:
            raise AlignError(f"'{name}' can't be scored: {exc}") from exc
        self.data, self.dataset_name, self.target = data, name, target
        self.sample = (sample.astype(str).reindex(data.index) if sample is not None
                       else pd.Series("all", index=data.index))
        self.groups, self.y = groups, y
        samples = [s for s in ("train", "test", "oot") if (self.sample == s).any()]
        samples += sorted(set(self.sample) - set(samples))
        self.samples = samples
        default = "oot" if "oot" in samples else "test" if "test" in samples else samples[0]
        if self.settings.get("strategy_sample") not in samples + ["all"]:
            self.settings["strategy_sample"] = default
        self.final = None
        self._recompute()

    def target_options(self, name: Optional[str] = None) -> List[str]:
        """0/1 columns of a dataset that can serve as the target."""
        data = self.datasets[name or self.dataset_name][0]
        out = []
        for c in data.columns:
            if c in self.card.variables or c == self.id_col or c == "sample":
                continue
            s = data[c]
            if s.notna().sum() == 0 or s.dropna().nunique() != 2:
                continue
            if _binary_or_none(s) is not None:
                out.append(c)
        return out

    def set_data(self, dataset: Optional[str] = None, target: Optional[str] = None) -> None:
        """Switch to another dataset and/or target; the scorecard and its
        overrides stay as they are."""
        with self.lock:
            name = dataset or self.dataset_name
            if name not in self.datasets:
                raise AlignError(f"Unknown dataset '{name}'.")
            tgt = target or (self.target if self.target in self.datasets.get(name, (pd.DataFrame(),))[0]
                             else None)
            if tgt is None:
                opts = self.target_options(name) if name in self.datasets else []
                if not opts:
                    raise AlignError(f"'{name}' has no 0/1 column to use as the target.")
                tgt = opts[0]
            old = (self.dataset_name, self.target)
            try:
                self._use(name, tgt)
                self.strategy()
            except AlignError:
                self._use(*old)
                raise
            self.save()

    # ══════════════════════════════════════════════════════════════════
    # Scores
    # ══════════════════════════════════════════════════════════════════

    def _recompute(self) -> None:
        pts = self.card.points_from_groups(self.groups)
        base = float(self.card.base_points if self.card.base_points_mode == "separate" else 0.0)
        total = np.full(len(pts), base)
        for c in pts.columns:
            total = total + pts[c].to_numpy()
        self.points = pts
        self.score = total

    def _mask(self, sample: Optional[str]) -> np.ndarray:
        if sample in (None, "all"):
            return np.ones(len(self.score), bool)
        return (self.sample == sample).to_numpy()

    # ══════════════════════════════════════════════════════════════════
    # Settings and cutoff
    # ══════════════════════════════════════════════════════════════════

    def set_settings(self, **kw) -> None:
        with self.lock:
            sc = self.card.scaling
            scaling_keys = {"pdo", "base_score", "base_odds", "base_points", "good_class"}
            if scaling_keys & set(kw):
                try:
                    new = ScalingParams(pdo=float(kw.get("pdo", sc.pdo)),
                                        base_score=float(kw.get("base_score", sc.base_score)),
                                        base_odds=float(kw.get("base_odds", sc.base_odds)))
                    self.card.rescale(new, base_points=kw.get("base_points", self.card.base_points_mode),
                                      good_class=int(kw.get("good_class", self.card.good_class)))
                except ValueError as exc:
                    raise AlignError(str(exc)) from exc
                self._recompute()
            for k in ("cost_bad", "benefit_good"):
                if k in kw:
                    v = float(kw[k])
                    if not math.isfinite(v) or v < 0:
                        raise AlignError(f"{k} must be a non-negative number.")
                    self.settings[k] = v
            if "strategy_sample" in kw:
                if kw["strategy_sample"] not in self.samples + ["all"]:
                    raise AlignError(f"Unknown sample '{kw['strategy_sample']}'.")
                self.settings["strategy_sample"] = kw["strategy_sample"]
            if "mode" in kw:
                if kw["mode"] not in MODES:
                    raise AlignError(f"mode must be one of {MODES}")
                self.settings["mode"] = kw["mode"]
                self.settings["value"] = kw.get("value")
            elif "value" in kw:
                self.settings["value"] = kw["value"]
            self.strategy()           # validate the cutoff choice
            self.save()

    def _curve(self, sample: Optional[str] = None) -> pd.DataFrame:
        m = self._mask(sample or self.settings["strategy_sample"])
        return cutoff_curve(self.score[m], self.y[m], self.card.good_class,
                            self.settings["cost_bad"], self.settings["benefit_good"])

    def chosen_cutoff(self) -> Dict[str, Any]:
        curve = self._curve()
        mode, value = self.settings["mode"], self.settings["value"]
        try:
            value = None if value in (None, "") else float(value)
            return choose_cutoff(curve, mode, value)
        except ValueError as exc:
            raise AlignError(str(exc)) from exc

    def strategy(self) -> Dict[str, Any]:
        """Everything the Strategy tab shows."""
        with self.lock:
            st = self.settings
            m = self._mask(st["strategy_sample"])
            s, y = self.score[m], self.y[m]
            curve = cutoff_curve(s, y, self.card.good_class, st["cost_bad"], st["benefit_good"])
            suggested = choose_cutoff(curve, "profit")
            chosen = self.chosen_cutoff()
            k = kpis(s, y, chosen["cutoff"], self.card.good_class, st["cost_bad"], st["benefit_good"])
            fin = curve[np.isfinite(curve["cutoff"])]
            if len(fin) > 400:
                fin = fin.iloc[np.unique(np.linspace(0, len(fin) - 1, 400).round().astype(int))]
            width = max(1.0, round(self.card.scaling.pdo / 4))
            be = break_even_score(self.card.scaling, st["cost_bad"], st["benefit_good"])
            return {
                "sample": st["strategy_sample"], "mode": st["mode"], "value": st["value"],
                "cutoff": chosen["cutoff"], "suggested": suggested["cutoff"], "kpis": k,
                "suggested_kpis": kpis(s, y, suggested["cutoff"], self.card.good_class,
                                       st["cost_bad"], st["benefit_good"]),
                "break_even_score": be,
                "curve": fin[["cutoff", "approval_rate", "bad_rate_approved", "bad_capture",
                              "profit_per_applicant"]].to_dict(orient="list"),
                "histogram": score_histogram(s, y, self.card.good_class, width).to_dict(orient="list"),
                "roc": roc_points(s, y, self.card.good_class).to_dict(orient="list"),
                "discrimination": discrimination(s, y, self.card.good_class),
                "score_range": [float(np.min(s)), float(np.max(s))] if len(s) else [0, 0],
            }

    # ══════════════════════════════════════════════════════════════════
    # Points
    # ══════════════════════════════════════════════════════════════════

    def set_points(self, variable: str, group: int, points: float, reason: str = "") -> None:
        with self.lock:
            try:
                self.card.set_points(variable, int(group), float(points), reason)
            except (KeyError, ValueError, TypeError) as exc:
                raise AlignError(str(exc)) from exc
            self._recompute()
            self.save()

    def reset_points(self, variable: Optional[str] = None, group: Optional[int] = None) -> None:
        with self.lock:
            self.card.reset_points(variable, None if group is None else int(group))
            self._recompute()
            self.save()

    def scorecard_rows(self) -> Dict[str, Any]:
        """Points table with order checks and each variable's points range."""
        with self.lock:
            p = self.card.points.copy()
            m = self._mask(self.settings["strategy_sample"])
            share = {}
            for v, rep in self.card.representations.items():
                if rep == "raw":
                    continue
                g = pd.Series(self.groups[v][m])
                share[v] = (g.value_counts(normalize=True)).to_dict()
            p["Share"] = [share.get(r.Variable, {}).get(int(r.Group), 0.0) if pd.notna(r.Group) else None
                          for r in p.itertuples()]
            p["Order flag"] = False
            ranges = {}
            for v, rows in p.groupby("Variable", sort=False):
                rep = self.card.representations[v]
                if rep == "raw":
                    col = self.points[f"pts_{v}"].to_numpy()
                    ranges[v] = [float(col.min()), float(col.max())]
                    continue
                ranges[v] = [float(rows["Points"].min()), float(rows["Points"].max())]
                # Points must follow risk: with good = 0 a higher WOE (more target=1) is riskier
                # and must not get more points; with good = 1 the opposite.
                reg = rows[rows["Kind"] == "regular"].dropna(subset=["WOE"])
                if len(reg) >= 2:
                    order = reg.sort_values("WOE")
                    pts = order["Points"].to_numpy() * (1 if self.card.good_class == 0 else -1)
                    bad = np.r_[False, pts[1:] > pts[:-1] + 1e-9]
                    p.loc[order.index[bad], "Order flag"] = True
            total_range = sum(hi - lo for lo, hi in ranges.values()) or 1.0
            return {
                "rows": p.to_dict(orient="records"),
                "ranges": {v: {"min": lo, "max": hi, "share": (hi - lo) / total_range} for v, (lo, hi) in ranges.items()},
                "representations": self.card.representations,
                "base_points": self.card.base_points if self.card.base_points_mode == "separate" else None,
                "n_manual": int(p["Manual"].astype(bool).sum()),
                "n_stale": int(p["Stale"].astype(bool).sum()),
                "n_order_flags": int(p["Order flag"].sum()),
            }

    # ══════════════════════════════════════════════════════════════════
    # Statistics
    # ══════════════════════════════════════════════════════════════════

    def stats(self, sample: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            sample = sample or self.settings["strategy_sample"]
            sc, gc = self.card.scaling, self.card.good_class
            per = []
            for smp in self.samples + (["all"] if len(self.samples) > 1 else []):
                m = self._mask(smp)
                d = discrimination(self.score[m], self.y[m], gc)
                g = gains_table(self.score[m], self.y[m], gc, width=sc.pdo, anchor=sc.base_score, scaling=sc)
                r = realized_scale(g, sc.base_score)
                yk = pd.to_numeric(pd.Series(self.y[m]), errors="coerce")
                per.append({"sample": smp, "rows": int(m.sum()), "bad_rate": float((yk.dropna() != gc).mean()),
                            "mean_score": float(np.mean(self.score[m])), **d,
                            "realized_pdo": r["realized_pdo"], "realized_odds_at_base": r["realized_odds_at_base"]})
            m = self._mask(sample)
            gains = gains_table(self.score[m], self.y[m], gc, width=sc.pdo, anchor=sc.base_score, scaling=sc)
            psi = {}
            if "train" in self.samples:
                for smp in self.samples:
                    if smp != "train":
                        psi[smp] = score_psi(self.score[self._mask("train")], self.score[self._mask(smp)])
            return {"sample": sample, "per_sample": per, "gains": gains.to_dict(orient="records"),
                    "design": {"pdo": sc.pdo, "base_score": sc.base_score, "base_odds": sc.base_odds},
                    "psi": psi, "contribution": self.scorecard_rows()["ranges"]}

    # ══════════════════════════════════════════════════════════════════
    # State, final table, code
    # ══════════════════════════════════════════════════════════════════

    def state(self) -> Dict[str, Any]:
        sc = self.card.scaling
        return {
            "dataset": self.dataset_name, "target": self.target, "rows": int(len(self.score)),
            "datasets": list(self.datasets), "target_options": self.target_options(),
            "outcome_rows": int(np.sum(~np.isnan(self.y))),
            "samples": self.samples, "good_class": self.card.good_class,
            "pdo": sc.pdo, "base_score": sc.base_score, "base_odds": sc.base_odds,
            "factor": sc.factor, "offset": sc.offset, "base_points": self.card.base_points_mode,
            "variables": self.card.variables, **self.settings,
            "last_saved": self.last_saved, "final": self.final,
        }

    def signature(self) -> str:
        """Identifies the model behind the scorecard (coefficients and bins)."""
        p = self.card.points
        key = json.dumps({"intercept": round(self.card.intercept, 10),
                          "coef": [round(float(c), 10) for c in p["Coefficient"].fillna(0)],
                          "variables": list(self.card.variables)})
        return _stable_hash(key)

    def to_state(self) -> Dict[str, Any]:
        return {"saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "signature": self.signature(), "settings": self.settings,
                "data": {"dataset": self.dataset_name, "target": self.target},
                "scorecard": self.card.to_dict()}

    def save(self) -> Optional[str]:
        if not self.save_dir:
            return None
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, STATE_FILE)
        fd, tmp = tempfile.mkstemp(dir=self.save_dir, prefix=".align_", suffix=".json")
        from ..scorecard import _json_safe
        with os.fdopen(fd, "w") as fh:
            json.dump(_json_safe(self.to_state()), fh, indent=2)
        os.replace(tmp, path)
        self.last_saved = datetime.now().strftime("%H:%M:%S")
        return path

    def load(self, state: Dict[str, Any]) -> None:
        """Restore settings and points saved for the same model."""
        with self.lock:
            if state.get("signature") != self.signature():
                raise AlignError("This alignment state was saved for a different scorecard.")
            card = Scorecard.from_dict(state["scorecard"])
            self.card = card
            self.settings.update(state.get("settings") or {})
            d = state.get("data") or {}
            try:
                if d.get("dataset") in self.datasets:
                    self._use(d["dataset"], d.get("target") or self.target)
            except AlignError:
                pass
            if self.settings.get("strategy_sample") not in self.samples + ["all"]:
                self.settings["strategy_sample"] = self.samples[0]
            self._recompute()

    def finalize(self) -> Dict[str, Any]:
        """Final scored table (every sample) and final scorecard, with the
        chosen cutoff, saved to ``save_dir``."""
        with self.lock:
            chosen = self.chosen_cutoff()
            self.card.cutoff = float(chosen["cutoff"])
            table = self.final_table()
            info = {"rows": int(len(table)), "columns": int(table.shape[1]), "cutoff": self.card.cutoff,
                    "approval_rate": float((table["decision"] == "approve").mean()),
                    "at": datetime.now().strftime("%H:%M:%S"), "paths": {}}
            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                P = lambda n: os.path.join(self.save_dir, n)
                table.to_parquet(P("final_scored.parquet"), index=False)
                table.to_csv(P("final_scored.csv"), index=False)
                self.card.save(P("final_scorecard.json"))
                self.card.table().to_csv(P("final_scorecard_table.csv"), index=False)
                info["paths"] = {"table": P("final_scored.parquet"), "csv": P("final_scored.csv"),
                                 "scorecard": P("final_scorecard.json"),
                                 "points_table": P("final_scorecard_table.csv")}
                self.save()
            self.final = info
            return info

    def final_table(self) -> pd.DataFrame:
        cols: Dict[str, Any] = {}
        if self.id_col and self.id_col in self.data:
            cols[self.id_col] = self.data[self.id_col].to_numpy()
        cols["sample"] = self.sample.to_numpy()
        cols[self.target] = self.y
        for v in self.card.variables:
            cols[v] = self.data[v].to_numpy()
        out = pd.DataFrame(cols)
        pts = self.points.reset_index(drop=True)
        out = pd.concat([out, pts], axis=1)
        out["score"] = self.score
        out["pd"] = self.card.pd_from_score(self.score)
        cutoff = self.card.cutoff if self.card.cutoff is not None else self.chosen_cutoff()["cutoff"]
        out["decision"] = np.where(self.score >= cutoff, "approve", "decline")
        return out

    def code(self, lang: str = "python", dialect: str = "standard", table: str = "input_table") -> str:
        with self.lock:
            card = self.card.copy()
            card.cutoff = float(self.chosen_cutoff()["cutoff"])
            if lang == "python":
                return card.to_python()
            if lang == "sql":
                try:
                    return card.to_sql(table or "input_table", dialect=dialect)
                except ValueError as exc:
                    raise AlignError(str(exc)) from exc
            raise AlignError("lang must be 'python' or 'sql'")

    def save_code(self, lang: str = "python", dialect: str = "standard", table: str = "input_table") -> str:
        if not self.save_dir:
            raise AlignError("No save folder for this session.")
        text = self.code(lang, dialect, table)
        name = "scorer.py" if lang == "python" else f"scorer_{dialect}.sql"
        path = os.path.join(self.save_dir, name)
        os.makedirs(self.save_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path


def _binary_or_none(s: pd.Series) -> Optional[np.ndarray]:
    """0/1 as float with NaN for unknown outcomes, or None if not binary."""
    from ..intake import _as_binary
    known = s.notna()
    b = _as_binary(s[known]) if known.any() else None
    if b is None:
        return None
    out = np.full(len(s), np.nan)
    out[known.to_numpy()] = b.to_numpy(dtype=float)
    return out


def _stable_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]
