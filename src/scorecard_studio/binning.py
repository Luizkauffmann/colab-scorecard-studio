"""
Optimal WOE binning engine (numerical + categorical) with explicit Missing
and Special bins.

Ported from the Dataiku ``binning_engine.py`` v3 with no Dataiku dependency.

Bin conventions
---------------
* Numerical bins are ``(lower, upper]``. The first bin is ``(-inf, c1]`` and
  the last is ``(c_k, +inf)``. A value equal to a cutoff falls in the lower bin.
* Regular bins are numbered ``1..k``.
* ``MISSING_GROUP`` (0) holds NaN/None. It always exists; if training data had
  no missing values its WOE is 0 (neutral) and a warning is raised.
* ``SPECIAL_GROUP`` (-1) pools all configured special codes (e.g. -999 =
  "no bureau record"). It exists only when special codes are configured.
* Unseen categories at scoring time are mapped to the Missing bin.
* WOE = ln(%events / %non-events): positive WOE = riskier than average.

Bins are always fitted on the data passed to :class:`BinningEngine` (your
TRAIN sample). Apply them to Test/OOT with :meth:`BinningEngine.transform`;
never refit on validation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .dtypes import (
    DEFAULT_MAX_CATEGORIES,
    detect_dtype,
    infer_variable_types,
    series_to_category_str,
    to_category_str,
    value_type_of,
)
from .metrics import (
    auc_from_bins,
    gini_from_auc,
    interpret_iv,
    ks_from_bins,
    woe_iv,
)

MISSING_GROUP = 0
SPECIAL_GROUP = -1
CONFIG_SCHEMA_VERSION = 1

Monotonic = Literal["none", "auto", "increasing", "decreasing"]
Divergence = Literal["iv", "js", "hellinger", "triangular"]

#: UI monotonic option -> optbinning ``monotonic_trend``.
#: "auto" forces a monotonic trend and lets the solver pick the direction
#: (optbinning's own "auto" may return peak/valley shapes, which we avoid).
#: "increasing" means the event rate (and WOE) increases with the variable.
MONOTONIC_MAP = {
    "none": None,
    "auto": "auto_asc_desc",
    "increasing": "ascending",
    "decreasing": "descending",
}
DIVERGENCES = ("iv", "js", "hellinger", "triangular")

SpecialCodes = Union[None, Sequence[Any], Dict[str, Sequence[Any]]]


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BinStats:
    label: str
    group: int
    kind: str                         # "regular" | "special" | "missing"
    lower: Optional[float]            # numerical regular bins only; None = unbounded
    upper: Optional[float]
    categories: Optional[List[str]]   # categorical regular bins only
    count: int
    event_count: int
    non_event_count: int
    event_rate: float
    woe: float
    iv_contribution: float


@dataclass
class BinningResult:
    variable: str
    dtype: str                         # "numerical" | "categorical"
    value_type: str                    # "numeric" | "string" | "boolean"
    bins: List[BinStats]
    cutoffs: List[float]
    cat_groups: Optional[List[List[str]]]
    special_codes: List[Any]
    iv: float
    gini: float
    ks: float
    is_monotonic: bool
    monotonic_direction: Optional[str]
    settings: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def divergence(self) -> str:
        return self.settings.get("divergence", "iv")

    @property
    def regular_bins(self) -> List[BinStats]:
        return [b for b in self.bins if b.kind == "regular"]

    def bin_by_group(self, group: int) -> BinStats:
        for b in self.bins:
            if b.group == group:
                return b
        raise KeyError(group)

    def summary(self) -> pd.DataFrame:
        """Binning table as a DataFrame (one row per bin, incl. Special/Missing)."""
        total = sum(b.count for b in self.bins) or 1
        return pd.DataFrame([{
            "Group": b.group,
            "Label": b.label,
            "Kind": b.kind,
            "Count": b.count,
            "Share": round(b.count / total, 4),
            "Events": b.event_count,
            "Non-events": b.non_event_count,
            "Event rate": round(b.event_rate, 4),
            "WOE": round(b.woe, 4),
            "IV contribution": round(b.iv_contribution, 4),
        } for b in self.bins])


# ═══════════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════════

class BinningEngine:
    """Fit, refine and apply WOE bins.

    Parameters
    ----------
    df : training data (TRAIN sample only).
    target_col : binary 0/1 target; 1 = event.
    special_codes : codes that get their own pooled Special bin. Either one
        list applied to every variable, or ``{variable: [codes]}``.
    exclude : columns never to bin (IDs, dates, leakage fields...).
    max_categories : numeric columns with at most this many distinct values
        are binned as categorical.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        special_codes: SpecialCodes = None,
        exclude: Optional[Iterable[str]] = None,
        max_categories: int = DEFAULT_MAX_CATEGORIES,
    ):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        y = df[target_col]
        if y.isna().any():
            raise ValueError(
                f"Target '{target_col}' has {int(y.isna().sum())} missing values. "
                "Remove indeterminate/unobserved rows before binning.")
        values = set(pd.unique(y))
        if not values <= {0, 1}:
            raise ValueError(
                f"Target '{target_col}' must be binary 0/1, found values {sorted(values)[:10]}.")
        if len(values) < 2:
            raise ValueError(f"Target '{target_col}' has a single class; nothing to model.")

        self.df = df
        self.target_col = target_col
        self.y = y.to_numpy().astype(int)
        self.total_events = int(self.y.sum())
        self.total_non_events = int(len(self.y) - self.total_events)
        self.special_codes = special_codes
        self.exclude = set(exclude or [])
        self.max_categories = max_categories
        self._results: Dict[str, BinningResult] = {}
        self.fit_errors: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Variable types (single source of truth for UI + notebook)
    # ------------------------------------------------------------------

    def variable_types(self) -> Dict[str, str]:
        """``{column: "numerical" | "categorical" | "datetime"}`` for all predictors."""
        return infer_variable_types(self.df, self.target_col, self.exclude, self.max_categories)

    def detect_dtype(self, variable: str) -> str:
        return detect_dtype(self.df[variable], self.max_categories)

    def special_codes_for(self, variable: str) -> List[Any]:
        sc = self.special_codes
        if sc is None:
            return []
        if isinstance(sc, dict):
            return list(sc.get(variable, []))
        return list(sc)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        variable: str,
        dtype: Optional[str] = None,
        max_bins: int = 6,
        monotonic: Monotonic = "auto",
        divergence: Divergence = "iv",
        min_bin_size: float = 0.05,
        min_bin_n_event: Optional[int] = None,
        cat_cutoff: Optional[float] = 0.05,
        special_codes: Optional[Sequence[Any]] = None,
    ) -> BinningResult:
        """Find optimal bins for one variable with optbinning.

        ``divergence`` is the objective optbinning maximises. ``min_bin_n_event``
        sets a minimum number of events per bin, which matters for low-event-rate
        problems (fraud, AML), where a 5% bin may hold only a handful of events.
        """
        self._check_variable(variable)
        if divergence not in DIVERGENCES:
            raise ValueError(f"divergence must be one of {DIVERGENCES}, got {divergence!r}")
        if monotonic not in MONOTONIC_MAP:
            raise ValueError(f"monotonic must be one of {tuple(MONOTONIC_MAP)}, got {monotonic!r}")
        dtype = dtype or self.detect_dtype(variable)
        if dtype not in ("numerical", "categorical"):
            raise ValueError(f"'{variable}' has dtype {dtype!r}; only numerical/categorical can be binned.")
        specials = list(special_codes) if special_codes is not None else self.special_codes_for(variable)

        settings = dict(max_bins=int(max_bins), monotonic=monotonic, divergence=divergence,
                        min_bin_size=float(min_bin_size), min_bin_n_event=min_bin_n_event,
                        cat_cutoff=cat_cutoff)
        x_raw = self.df[variable]
        missing, special, clean = self._masks(x_raw, dtype, specials)
        y_clean = self.y[clean]

        if clean.sum() == 0:
            raise ValueError(f"'{variable}' has no non-missing, non-special values.")

        from optbinning import OptimalBinning  # imported lazily: heavy import

        # min_bin_size is a share of ALL rows; optbinning only sees the rows that
        # are neither missing nor special, so rescale it to that subset.
        share_of_clean = float(min_bin_size) * len(self.y) / int(clean.sum())
        params: Dict[str, Any] = dict(
            name=variable, dtype=dtype, solver="mip", divergence=divergence,
            max_n_bins=int(max_bins), min_bin_size=min(share_of_clean, 0.5),
        )
        if min_bin_n_event:
            params["min_bin_n_event"] = int(min_bin_n_event)

        if dtype == "numerical":
            x_clean = x_raw[clean].astype("float64").to_numpy()
            params["monotonic_trend"] = MONOTONIC_MAP[monotonic]
            optb = OptimalBinning(**params)
            optb.fit(x_clean, y_clean)
            cutoffs = sorted(float(c) for c in np.asarray(optb.splits).ravel())
            result = self._build(variable, dtype, cutoffs=cutoffs, specials=specials, settings=settings)
            if monotonic != "none" and not result.is_monotonic:
                result.warnings.append(
                    f"Monotonic trend '{monotonic}' requested but the WOE is not monotonic "
                    "(the solver may have relaxed the constraint).")
        else:
            x_clean = series_to_category_str(x_raw[clean]).to_numpy()
            if cat_cutoff is not None:
                params["cat_cutoff"] = float(cat_cutoff)
            optb = OptimalBinning(**params)
            optb.fit(x_clean, y_clean)
            groups = [[str(c) for c in g] for g in (optb.splits or []) if len(g) > 0]
            seen = {c for g in groups for c in g}
            leftovers = sorted(set(x_clean) - seen)
            if leftovers:  # never silently drop a training category
                groups.append(leftovers)
            result = self._build(variable, dtype, cat_groups=groups, specials=specials, settings=settings)

        self._results[variable] = result
        self.fit_errors.pop(variable, None)
        return result

    def fit_all(
        self,
        variables: Optional[Iterable[str]] = None,
        categorical_variables: Optional[Iterable[str]] = None,
        **fit_kwargs,
    ) -> pd.DataFrame:
        """Fit every predictor. Types come from :meth:`variable_types`;
        ``categorical_variables`` forces columns to categorical. Failures are
        collected in ``self.fit_errors`` instead of stopping the loop."""
        types = self.variable_types()
        if variables is None:
            variables = [v for v, t in types.items() if t != "datetime"]
        forced = set(categorical_variables or [])
        for v in variables:
            dtype = "categorical" if v in forced else types.get(v) or self.detect_dtype(v)
            try:
                self.fit(v, dtype=dtype, **fit_kwargs)
            except Exception as exc:  # noqa: BLE001 - report and continue
                self.fit_errors[v] = f"{type(exc).__name__}: {exc}"
        return self.get_iv_summary()

    def adjust_cutoffs(self, variable: str, new_cutoffs: Sequence[float]) -> BinningResult:
        """Replace the cutoffs of a numerical variable (manual refinement)."""
        prev = self._require(variable)
        if prev.dtype != "numerical":
            raise ValueError(f"'{variable}' is categorical; use merge_categories().")
        cuts = [float(c) for c in new_cutoffs]
        if any(not np.isfinite(c) for c in cuts):
            raise ValueError("Cutoffs must be finite numbers.")
        cuts = sorted(set(cuts))
        result = self._build(variable, "numerical", cutoffs=cuts,
                             specials=prev.special_codes, settings=prev.settings)
        self._results[variable] = result
        return result

    def merge_categories(self, variable: str, group_assignments: Dict[str, int]) -> BinningResult:
        """Regroup a categorical variable. ``group_assignments`` maps every
        training category (canonical string) to a group id."""
        prev = self._require(variable)
        if prev.dtype != "categorical":
            raise ValueError(f"'{variable}' is numerical; use adjust_cutoffs().")
        known = {c for g in prev.cat_groups or [] for c in g}
        assigned = {str(k) for k in group_assignments}
        missing_cats = sorted(known - assigned)
        if missing_cats:
            raise ValueError(f"Categories without a group: {missing_cats[:20]}")
        by_gid: Dict[int, List[str]] = {}
        for cat, gid in group_assignments.items():
            by_gid.setdefault(int(gid), []).append(str(cat))
        groups = [sorted(by_gid[k]) for k in sorted(by_gid)]
        result = self._build(variable, "categorical", cat_groups=groups,
                             specials=prev.special_codes, settings=prev.settings)
        self._results[variable] = result
        return result

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        variables: Optional[Iterable[str]] = None,
        metrics: Sequence[str] = ("woe", "group", "label"),
        prefix: str = "opt_",
    ) -> pd.DataFrame:
        """Apply fitted bins to any sample (Train, Test, OOT, new data).
        Adds ``{prefix}{var}_{metric}`` columns."""
        return self.build_scoring_bundle(variables).score_dataframe(df, metrics=metrics, prefix=prefix)

    def build_scoring_bundle(self, variables: Optional[Iterable[str]] = None,
                             name: str = "binning_bundle", description: str = ""):
        from .artifacts import ScoringArtifact, ScoringBundle
        names = list(variables) if variables is not None else list(self._results)
        artifacts = {v: ScoringArtifact.from_result(self._require(v)) for v in names}
        return ScoringBundle(artifacts=artifacts, name=name, description=description)

    def get_scoring_artifact(self, variable: str):
        from .artifacts import ScoringArtifact
        return ScoringArtifact.from_result(self._require(variable))

    # ------------------------------------------------------------------
    # Results / persistence
    # ------------------------------------------------------------------

    @property
    def results(self) -> Dict[str, BinningResult]:
        return dict(self._results)

    def get_result(self, variable: str) -> Optional[BinningResult]:
        return self._results.get(variable)

    def get_iv_summary(self) -> pd.DataFrame:
        cols = ["Variable", "Type", "IV", "IV interpretation", "Gini", "KS", "Bins",
                "Monotonic", "Warnings"]
        rows = [{
            "Variable": v, "Type": r.dtype, "IV": round(r.iv, 4),
            "IV interpretation": interpret_iv(r.iv), "Gini": round(r.gini, 4),
            "KS": round(r.ks, 4), "Bins": len(r.regular_bins),
            "Monotonic": r.is_monotonic, "Warnings": len(r.warnings),
        } for v, r in self._results.items()]
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols).sort_values("IV", ascending=False).reset_index(drop=True)

    def export_config(self) -> dict:
        """JSON-serialisable description of all bins (saved to Drive by the app)."""
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "target_col": self.target_col,
            "variables": {
                v: {
                    "dtype": r.dtype,
                    "cutoffs": list(r.cutoffs),
                    "cat_groups": r.cat_groups,
                    "special_codes": _json_safe_list(r.special_codes),
                    "settings": r.settings,
                } for v, r in self._results.items()
            },
        }

    def import_config(self, config: dict) -> List[str]:
        """Rebuild bins from :meth:`export_config` output on this engine's data.
        Returns the variables that were skipped because they are not in ``df``."""
        if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported config schema_version {config.get('schema_version')!r}")
        skipped = []
        for v, cfg in config["variables"].items():
            if v not in self.df.columns:
                skipped.append(v)
                continue
            self._results[v] = self._build(
                v, cfg["dtype"], cutoffs=cfg.get("cutoffs") or [],
                cat_groups=cfg.get("cat_groups"), specials=cfg.get("special_codes") or [],
                settings=cfg.get("settings") or {})
        return skipped

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_variable(self, variable: str):
        if variable not in self.df.columns:
            raise ValueError(f"'{variable}' not in dataframe.")
        if variable == self.target_col:
            raise ValueError("Cannot bin the target column.")

    def _require(self, variable: str) -> BinningResult:
        if variable not in self._results:
            raise ValueError(f"'{variable}' not fitted yet.")
        return self._results[variable]

    @staticmethod
    def _masks(x_raw: pd.Series, dtype: str, specials: Sequence[Any]):
        missing = x_raw.isna().to_numpy()
        if specials:
            if dtype == "numerical":
                codes = [float(s) for s in specials]
                special = x_raw.astype("float64").isin(codes).to_numpy() & ~missing
            else:
                codes = {to_category_str(s) for s in specials}
                special = series_to_category_str(x_raw).isin(codes).to_numpy() & ~missing
        else:
            special = np.zeros(len(x_raw), dtype=bool)
        return missing, special, ~missing & ~special

    def _build(self, variable: str, dtype: str, *, cutoffs: Sequence[float] = (),
               cat_groups: Optional[List[List[str]]] = None,
               specials: Sequence[Any] = (), settings: Optional[dict] = None) -> BinningResult:
        """Compute every bin's statistics from scratch on the training data."""
        settings = dict(settings or {})
        x_raw = self.df[variable]
        y = self.y
        missing, special, clean = self._masks(x_raw, dtype, specials)
        te, tne = self.total_events, self.total_non_events
        warnings: List[str] = []
        bins: List[BinStats] = []

        def make(label, group, kind, mask, lower=None, upper=None, categories=None):
            n = int(mask.sum())
            ev = int(y[mask].sum())
            ne = n - ev
            w, ivc, adjusted = woe_iv(ev, ne, te, tne)
            if adjusted:
                warnings.append(
                    f"Bin '{label}' has {'no events' if ev == 0 else 'no non-events'}; "
                    "WOE smoothed with +0.5 counts. Consider merging it.")
            return BinStats(label=label, group=group, kind=kind, lower=lower, upper=upper,
                            categories=categories, count=n, event_count=ev,
                            non_event_count=ne, event_rate=(ev / n if n else 0.0),
                            woe=w, iv_contribution=ivc)

        if dtype == "numerical":
            cuts = sorted(float(c) for c in cutoffs)
            x = np.where(clean, pd.to_numeric(x_raw, errors="coerce").astype("float64").to_numpy(), np.nan)
            idx = np.searchsorted(np.asarray(cuts, dtype=float), x, side="left")  # (lo, hi]
            bounds = [None] + cuts + [None]
            for i in range(len(cuts) + 1):
                lo, hi = bounds[i], bounds[i + 1]
                bins.append(make(_numeric_label(lo, hi), i + 1, "regular", clean & (idx == i),
                                 lower=lo, upper=hi))
            groups_out, cuts_out = None, cuts
        elif dtype == "categorical":
            if not cat_groups:
                raise ValueError(f"No category groups for '{variable}'.")
            cats = series_to_category_str(x_raw).to_numpy()
            groups_out = [[str(c) for c in g] for g in cat_groups]
            for i, g in enumerate(groups_out):
                mask = clean & np.isin(cats, g)
                bins.append(make(" | ".join(g), i + 1, "regular", mask, categories=list(g)))
            covered = clean & np.isin(cats, [c for g in groups_out for c in g])
            n_unc = int((clean & ~covered).sum())
            if n_unc:
                warnings.append(f"{n_unc} training rows have categories outside every group; "
                                "they will score as Missing/Unknown.")
            cuts_out = []
        else:
            raise ValueError(f"Unknown dtype {dtype!r}")

        if specials:
            bins.append(make("Special", SPECIAL_GROUP, "special", special))
        bins.append(make("Missing", MISSING_GROUP, "missing", missing))
        if missing.sum() == 0:
            warnings.append("No missing values in training data: missing values at scoring "
                            "time get WOE 0 (neutral).")

        min_share = float(settings.get("min_bin_size") or 0.0)
        n_total = len(y)
        for b in bins:
            if b.kind == "regular" and b.count / n_total < min_share:
                warnings.append(f"Bin '{b.label}' holds {b.count / n_total:.1%} of rows "
                                f"(< {min_share:.0%}).")

        events = [b.event_count for b in bins]
        non_events = [b.non_event_count for b in bins]
        woes = [b.woe for b in bins]
        auc = auc_from_bins(events, non_events, woes)
        ks = ks_from_bins(events, non_events, woes)
        iv = float(sum(b.iv_contribution for b in bins))

        reg_woes = [b.woe for b in bins if b.kind == "regular" and b.count > 0]
        inc = all(reg_woes[i] >= reg_woes[i - 1] - 1e-9 for i in range(1, len(reg_woes)))
        dec = all(reg_woes[i] <= reg_woes[i - 1] + 1e-9 for i in range(1, len(reg_woes)))
        if dtype == "numerical":
            is_mono = inc or dec
            direction = ("flat" if inc and dec else "increasing" if inc
                         else "decreasing" if dec else None)
        else:
            is_mono, direction = True, None  # order of categorical groups is arbitrary

        return BinningResult(
            variable=variable, dtype=dtype, value_type=value_type_of(x_raw), bins=bins,
            cutoffs=list(cuts_out), cat_groups=groups_out,
            special_codes=_json_safe_list(specials), iv=iv, gini=gini_from_auc(auc), ks=ks,
            is_monotonic=is_mono, monotonic_direction=direction,
            settings=settings, warnings=warnings,
        )


def _fmt(v: float) -> str:
    if float(v).is_integer() and abs(v) < 1e15:
        return f"{int(v):,}"
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    return f"{v:.4g}"


def _numeric_label(lo: Optional[float], hi: Optional[float]) -> str:
    """Labels follow the (lower, upper] convention exactly."""
    if lo is None and hi is None:
        return "All values"
    if lo is None:
        return f"<= {_fmt(hi)}"
    if hi is None:
        return f"> {_fmt(lo)}"
    return f"({_fmt(lo)}, {_fmt(hi)}]"


def _json_safe_list(values: Iterable[Any]) -> List[Any]:
    out = []
    for v in values or []:
        if isinstance(v, (np.integer,)):
            out.append(int(v))
        elif isinstance(v, (np.floating,)):
            out.append(float(v))
        elif isinstance(v, (np.bool_,)):
            out.append(bool(v))
        else:
            out.append(v)
    return out
