"""
Logistic regression on the model-ready dataset, with a per-variable choice of
representation:

* ``"woe"``  - one column, ``woe_<var>``. Expected coefficient > 0 (positive
  WOE = riskier). The classic scorecard set-up.
* ``"bins"`` - one dummy per bin of ``opt_<var>``, the most populous Train bin
  as reference. A separate Missing bin becomes its own dummy, i.e. a missing
  flag. Costs one parameter per bin.
* ``"raw"``  - the original value, linear. Only for variables without missing
  values or special codes; gives a linear points formula, not a points table.

The model is always fit on the **Train** rows. Test/OOT are only scored.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .artifacts import ScoringBundle
from .scorecard import REPRESENTATIONS, ordered_labels, performance


class ModelError(ValueError):
    """The requested model can't be fit as specified; the message says why."""


@dataclass
class LogisticModel:
    target: str
    representations: Dict[str, str]
    intercept: float
    coef: Dict[str, float]                      # woe / raw variables
    bin_coefs: Dict[str, Dict[int, float]]      # bins variables: group -> beta (reference = 0)
    reference: Dict[str, int]                   # bins variables: reference group
    coefficients: pd.DataFrame                  # one row per design column
    summary: pd.DataFrame                       # one row per variable
    flags: List[str]
    performance: pd.DataFrame
    selection_log: List[str] = field(default_factory=list)
    result: Any = None                          # statsmodels results object

    def bin_coef(self, variable: str, group: int) -> float:
        return float(self.bin_coefs[variable].get(int(group), 0.0))

    def predict_logit(self, model_df: pd.DataFrame, bundle: ScoringBundle) -> np.ndarray:
        X, _, _ = design_matrix(model_df, self.representations, bundle, reference=self.reference)
        logit = np.full(len(model_df), self.intercept)
        for v, rep in self.representations.items():
            if rep == "bins":
                for g, b in self.bin_coefs[v].items():
                    col = _dummy_name(v, g)
                    if col in X:
                        logit = logit + b * X[col].to_numpy()
            else:
                logit = logit + self.coef[v] * X[_col_name(v, rep)].to_numpy()
        return logit

    def predict_proba(self, model_df: pd.DataFrame, bundle: ScoringBundle) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.predict_logit(model_df, bundle)))


# ═══════════════════════════════════════════════════════════════════════════
# Design matrix
# ═══════════════════════════════════════════════════════════════════════════

def _col_name(v: str, rep: str) -> str:
    return f"woe_{v}" if rep == "woe" else v


def _dummy_name(v: str, group: int) -> str:
    return f"{v}[G{group}]" if group > 0 else f"{v}[{'Missing' if group == 0 else 'Special'}]"


def _groups(model_df: pd.DataFrame, v: str, bundle: ScoringBundle) -> np.ndarray:
    """Bin group of every row, from the ``opt_<var>`` labels (or the raw value)."""
    art = bundle.artifacts[v]
    if f"opt_{v}" in model_df:
        _, by_group = ordered_labels(art)
        to_group = {lbl: g for g, lbl in by_group.items()}
        return model_df[f"opt_{v}"].astype(str).map(to_group).to_numpy()
    return art.transform_series(model_df[v])["group"]


def design_matrix(model_df: pd.DataFrame, representations: Dict[str, str], bundle: ScoringBundle,
                  reference: Optional[Dict[str, int]] = None, train_mask: Optional[np.ndarray] = None,
                  y_train: Optional[np.ndarray] = None
                  ) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, int]]:
    """``(X, columns per variable, reference group per bins-variable)``."""
    bad = {v: r for v, r in representations.items() if r not in REPRESENTATIONS}
    if bad:
        raise ModelError(f"Representation must be one of {REPRESENTATIONS}: {bad}")
    train_mask = np.ones(len(model_df), bool) if train_mask is None else np.asarray(train_mask)
    reference = dict(reference or {})
    cols: Dict[str, np.ndarray] = {}
    by_var: Dict[str, List[str]] = {}
    for v, rep in representations.items():
        if rep == "woe":
            if f"woe_{v}" not in model_df:
                raise ModelError(f"'woe_{v}' is not in the model dataset; bin '{v}' first.")
            cols[f"woe_{v}"] = model_df[f"woe_{v}"].to_numpy(float)
            by_var[v] = [f"woe_{v}"]
        elif rep == "raw":
            x = pd.to_numeric(model_df[v], errors="coerce")
            if not pd.api.types.is_numeric_dtype(model_df[v]):
                raise ModelError(f"'{v}' is not numeric; use 'bins' or 'woe'.")
            if x[train_mask].isna().any():
                raise ModelError(f"'{v}' has missing values; a raw variable can't hold them. Use 'bins' or 'woe'.")
            art = bundle.artifacts.get(v)
            codes = [float(c) for c in (art.special_codes if art else [])]
            if codes and x[train_mask].isin(codes).any():
                raise ModelError(f"'{v}' contains special codes {codes}; a raw (linear) term would treat "
                                 "them as numbers. Use 'bins' or 'woe'.")
            cols[v] = x.to_numpy(float)
            by_var[v] = [v]
        else:
            g = _groups(model_df, v, bundle)
            present = pd.Series(g[train_mask]).value_counts()
            if y_train is not None:
                ev = pd.Series(y_train).groupby(g[train_mask]).agg(["size", "sum"])
                bad_bins = [int(k) for k, r in ev.iterrows() if r["sum"] in (0, r["size"])]
                if bad_bins:
                    names = ", ".join(f"{_dummy_name(v, k)} ({int(ev.loc[k, 'size'])} rows)" for k in bad_bins)
                    raise ModelError(
                        f"'{v}' as bins: {names} has only events or only non-events in Train, so its "
                        "coefficient is infinite. Merge that bin in the binning app (Missing/Special can be "
                        "merged too) or use 'woe' for this variable.")
            if v not in reference:
                reference[v] = int(present.idxmax())
            names = []
            for grp in sorted(present.index):
                if int(grp) == reference[v]:
                    continue
                name = _dummy_name(v, int(grp))
                cols[name] = (g == grp).astype(float)
                names.append(name)
            if not names:
                raise ModelError(f"'{v}' has a single bin with Train rows; nothing to estimate.")
            by_var[v] = names
    return pd.DataFrame(cols, index=model_df.index), by_var, reference


# ═══════════════════════════════════════════════════════════════════════════
# Fit
# ═══════════════════════════════════════════════════════════════════════════

def fit_logistic(model_df: pd.DataFrame, target: str, representations: Dict[str, str],
                 bundle: ScoringBundle, *, sample_col: str = "sample", selection: str = "none",
                 p_max: float = 0.05, keep: Iterable[str] = (), drop_wrong_sign: bool = True,
                 vif_max: float = 5.0, corr_max: float = 0.7) -> LogisticModel:
    """Fit on the Train rows of ``model_df`` (the model-ready dataset).

    ``selection="backward"`` removes one variable at a time: first any WOE
    variable with a non-positive coefficient (if ``drop_wrong_sign``), then the
    variable with the highest p-value above ``p_max`` (joint Wald test for
    bins). Variables in ``keep`` are never removed.
    """
    if selection not in ("none", "backward"):
        raise ModelError("selection must be 'none' or 'backward'")
    reps = dict(representations)
    if not reps:
        raise ModelError("No variables to model.")
    train = (model_df[sample_col] == "train").to_numpy() if sample_col in model_df else np.ones(len(model_df), bool)
    y = model_df[target].to_numpy()
    if np.isnan(y.astype(float)).any():
        raise ModelError(f"'{target}' has missing values in the model dataset.")
    keep = set(keep)
    log: List[str] = []

    while True:
        res, X, by_var, ref = _fit(model_df, y, train, reps, bundle)
        if selection == "none":
            break
        cands = []
        for v, cs in by_var.items():
            if v in keep or len(reps) == 1:
                continue
            p = _joint_p(res, cs)
            wrong = reps[v] == "woe" and res.params[cs[0]] <= 0
            if wrong and drop_wrong_sign:
                cands.append((2, 1.0, v, f"removed {v}: WOE coefficient {res.params[cs[0]]:.3f} <= 0"))
            elif p > p_max:
                cands.append((1, p, v, f"removed {v}: p-value {p:.3g} > {p_max}"))
        if not cands:
            break
        _, _, v, msg = max(cands)
        log.append(msg)
        del reps[v]

    coef_tbl = pd.DataFrame({
        "Column": res.params.index, "Coefficient": res.params.to_numpy(),
        "Std. error": res.bse.to_numpy(), "z": res.tvalues.to_numpy(), "p-value": res.pvalues.to_numpy(),
    })
    vif = _vif(X[train])
    coef_tbl["VIF"] = coef_tbl["Column"].map(vif)

    coef, bin_coefs = {}, {}
    for v, rep in reps.items():
        cs = by_var[v]
        if rep == "bins":
            bin_coefs[v] = {_group_of_dummy(c): float(res.params[c]) for c in cs}
        else:
            coef[v] = float(res.params[cs[0]])

    flags = _flags(res, X[train], y[train], reps, by_var, bin_coefs, ref, bundle, vif, vif_max, corr_max)
    summary = _summary(res, reps, by_var, vif, bundle, bin_coefs)

    logit = res.params["const"] + X.to_numpy() @ res.params.drop("const").reindex(X.columns).to_numpy()
    perf = performance(y, -logit, model_df[sample_col] if sample_col in model_df else None)
    return LogisticModel(target=target, representations=reps, intercept=float(res.params["const"]),
                         coef=coef, bin_coefs=bin_coefs, reference=ref, coefficients=coef_tbl,
                         summary=summary, flags=flags, performance=perf, selection_log=log, result=res)


def _fit(model_df, y, train, reps, bundle):
    import statsmodels.api as sm
    X, by_var, ref = design_matrix(model_df, reps, bundle, train_mask=train, y_train=y[train])
    Xc = sm.add_constant(X[train], has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.Logit(y[train].astype(float), Xc).fit(disp=0, maxiter=200)
        except np.linalg.LinAlgError as exc:
            raise ModelError("The design matrix is singular: two columns carry the same information "
                             "(e.g. a bins variable whose dummies duplicate another variable). "
                             "Remove one of them.") from exc
        except Exception as exc:  # noqa: BLE001 - PerfectSeparation etc.
            raise ModelError(f"Logistic regression failed: {type(exc).__name__}: {exc}. "
                             "A bin that is all events or all non-events causes this: merge it.") from exc
    if not res.mle_retvals.get("converged", True):
        raise ModelError("Logistic regression did not converge. Check for bins with no events or "
                         "non-events, or highly collinear variables.")
    return res, X, by_var, ref


def _group_of_dummy(col: str) -> int:
    tag = col[col.rindex("[") + 1:-1]
    return 0 if tag == "Missing" else -1 if tag == "Special" else int(tag[1:])


def _joint_p(res, cols: Sequence[str]) -> float:
    if len(cols) == 1:
        return float(res.pvalues[cols[0]])
    idx = [list(res.params.index).index(c) for c in cols]
    R = np.zeros((len(cols), len(res.params)))
    for i, j in enumerate(idx):
        R[i, j] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(res.wald_test(R, scalar=True).pvalue)


def _vif(X: pd.DataFrame) -> Dict[str, float]:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    if X.shape[1] < 2:
        return {c: 1.0 for c in X.columns}
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, c in enumerate(X.columns, start=1):
            out[c] = float(variance_inflation_factor(Xc, i))
    return out


def _flags(res, Xt, yt, reps, by_var, bin_coefs, ref, bundle, vif, vif_max, corr_max) -> List[str]:
    flags = []
    for v, rep in reps.items():
        cs = by_var[v]
        p = _joint_p(res, cs)
        if p > 0.05:
            flags.append(f"{v}: p-value {p:.3g} > 0.05 (not significant).")
        if rep == "woe" and res.params[cs[0]] <= 0:
            flags.append(f"{v}: WOE coefficient {res.params[cs[0]]:.3f} is not positive; the variable's effect "
                         "is reversed in the model (collinearity or suppression). Remove it or rebin.")
        if rep == "raw":
            r = pd.Series(Xt[cs[0]].to_numpy()).corr(pd.Series(yt), method="spearman")
            if np.sign(r) != np.sign(res.params[cs[0]]) and abs(r) > 0.01:
                flags.append(f"{v}: coefficient sign disagrees with the variable's univariate relationship.")
        if rep == "bins":
            art = bundle.artifacts[v]
            woe = {b["group"]: b["woe"] for b in art.bins}
            betas = {**bin_coefs[v], ref[v]: 0.0}
            if len(betas) >= 3:
                gs = sorted(betas)
                r = pd.Series([betas[g] for g in gs]).corr(pd.Series([woe[g] for g in gs]), method="spearman")
                if r < 0.5:
                    flags.append(f"{v}: bin coefficients don't follow the bins' WOE order "
                                 f"(rank correlation {r:.2f}); some bins may be unstable or redundant.")
            own = {b["group"] for b in art.bins if b["kind"] == "regular" or b["group"] <= 0}
            empty = sorted(own - set(betas))
            if empty:
                flags.append(f"{v}: bin(s) {', '.join(_dummy_name(v, g) for g in empty)} have no Train rows "
                             "and are scored like the reference bin.")
        worst = max((vif.get(c, 0.0) for c in cs), default=0.0)
        if worst > vif_max:
            flags.append(f"{v}: VIF {worst:.1f} > {vif_max} (collinear with other variables).")
    single = [c for v, cs in by_var.items() if len(cs) == 1 for c in cs]
    if len(single) >= 2:
        corr = Xt[single].corr()
        for i, a in enumerate(single):
            for b in single[i + 1:]:
                r = corr.at[a, b]
                if abs(r) >= corr_max:
                    flags.append(f"{a} and {b} are correlated (r = {r:.2f}); consider keeping one.")
    return flags


def _summary(res, reps, by_var, vif, bundle, bin_coefs) -> pd.DataFrame:
    rows = []
    for v, rep in reps.items():
        cs = by_var[v]
        art = bundle.artifacts.get(v)
        rows.append({
            "Variable": v, "Representation": rep, "Parameters": len(cs),
            "Coefficient": float(res.params[cs[0]]) if len(cs) == 1 else None,
            "p-value": _joint_p(res, cs),
            "Max VIF": max(vif.get(c, np.nan) for c in cs),
            "IV": art.iv if art is not None else None,
        })
    return pd.DataFrame(rows)
