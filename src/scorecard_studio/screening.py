"""
Stage 3 - univariate screening, on the Train sample only.

For every candidate predictor: missing rate, cardinality, top-value share,
an automatic optimal-binning fit with the preset's constraints (the same
engine and settings the binning app will use), and from it IV, Gini, KS and
the event rate per bin.

Status rules, in order of precedence:

========================  =====================================================
``excluded``              listed in EXCLUDE by the user (still profiled)
``id_like``               looks like an identifier (near-unique values)
``constant``              a single distinct value
``fit_error``             the binning fit failed (reason in ``reason``)
``forced``                in FORCE_INCLUDE: kept whatever its IV
``review``                IV > IV_MAX: probably leakage, or a variable that
                          needs a decision. Not kept until the user puts it
                          in FORCE_INCLUDE (reviewed) or EXCLUDE (leakage).
``low_iv``                IV < IV_MIN
``selected``              passes
========================  =====================================================

Correlated numerical pairs are flagged, not dropped: redundancy is decided
later, on WOE, with the model in view.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .binning import BinningEngine, BinningResult
from .dtypes import DEFAULT_MAX_CATEGORIES, detect_dtype
from .metrics import interpret_iv

SELECTED, FORCED, REVIEW, LOW_IV = "selected", "forced", "review", "low_iv"
EXCLUDED, ID_LIKE, CONSTANT, FIT_ERROR = "excluded", "id_like", "constant", "fit_error"
KEPT = (SELECTED, FORCED)
STATUS_ORDER = [SELECTED, FORCED, REVIEW, LOW_IV, EXCLUDED, ID_LIKE, CONSTANT, FIT_ERROR]

#: Engine warnings that matter in the binning app but are noise at screening.
_NOISE = ("No missing values in training data",)

_COLUMNS = ["variable", "type", "missing_rate", "n_unique", "top_share", "iv", "iv_band",
            "gini", "ks", "bins", "monotonic", "status", "reason", "warnings"]


def is_id_like(series: pd.Series, dtype: str) -> bool:
    """Near-unique values: an identifier, a free-text field or a timestamp in
    disguise. Binning it would memorise rows."""
    s = series.dropna()
    if len(s) < 20:
        return False
    ratio = s.nunique() / len(s)
    if dtype == "categorical":
        return s.nunique() > 50 and ratio > 0.5
    # Numerical: only integer-valued columns where almost every value is unique.
    values = pd.to_numeric(s, errors="coerce").dropna()
    return ratio > 0.95 and bool(np.all(np.mod(values, 1) == 0))


@dataclass
class ScreeningResult:
    table: pd.DataFrame
    engine: BinningEngine
    correlated_pairs: pd.DataFrame
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def selected(self) -> List[str]:
        """Variables passed on to the binning app (``selected`` + ``forced``)."""
        return self.table.loc[self.table["status"].isin(KEPT), "variable"].tolist()

    @property
    def needs_review(self) -> List[str]:
        return self.table.loc[self.table["status"] == REVIEW, "variable"].tolist()

    def status_counts(self) -> pd.Series:
        counts = self.table["status"].value_counts()
        return counts.reindex([s for s in STATUS_ORDER if s in counts.index])

    def result(self, variable: str) -> Optional[BinningResult]:
        return self.engine.get_result(variable)

    def shortlist(self) -> Dict[str, Any]:
        """What ``shortlist.json`` holds: the kept variables and why every
        other candidate left."""
        dropped = {r.variable: {"status": r.status, "reason": r.reason,
                                "iv": None if pd.isna(r.iv) else round(float(r.iv), 4)}
                   for r in self.table.itertuples() if r.status not in KEPT}
        return {"selected": self.selected, "needs_review": self.needs_review,
                "dropped": dropped, "params": self.params}

    # ------------------------------------------------------------------

    def plot(self, variable: str, ax=None):
        """Bin share (bars) and event rate (line) for one variable."""
        return plot_bins(self.result(variable), ax=ax)

    def plot_grid(self, variables: Optional[Iterable[str]] = None, ncols: int = 3):
        import matplotlib.pyplot as plt
        if variables is None:
            variables = [v for v in self.table["variable"]
                         if self.result(v) is not None and
                         self.table.set_index("variable").at[v, "status"] in (*KEPT, REVIEW)]
        variables = list(variables)
        if not variables:
            return None
        nrows = int(np.ceil(len(variables) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows), squeeze=False)
        for ax, v in zip(axes.ravel(), variables):
            plot_bins(self.result(v), ax=ax)
        for ax in axes.ravel()[len(variables):]:
            ax.axis("off")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------

    def save(self, folder: str, extra_html: str = "") -> Dict[str, str]:
        os.makedirs(folder, exist_ok=True)
        paths = {
            "shortlist": os.path.join(folder, "shortlist.json"),
            "table": os.path.join(folder, "screening_table.csv"),
            "report": os.path.join(folder, "screening_report.html"),
        }
        with open(paths["shortlist"], "w") as fh:
            json.dump(self.shortlist(), fh, indent=2, default=str)
        self.table.to_csv(paths["table"], index=False)
        with open(paths["report"], "w", encoding="utf-8") as fh:
            fh.write(self.to_html(extra_html=extra_html))
        return paths

    def to_html(self, extra_html: str = "", max_plots: int = 40) -> str:
        return _render_html(self, extra_html=extra_html, max_plots=max_plots)


# ═══════════════════════════════════════════════════════════════════════════
# Screening
# ═══════════════════════════════════════════════════════════════════════════


def screen_variables(
    train: pd.DataFrame,
    target: str,
    *,
    iv_min: float,
    iv_max: float,
    fit_params: Optional[Dict[str, Any]] = None,
    exclude: Iterable[str] = (),
    force_include: Iterable[str] = (),
    special_codes: Optional[Dict[str, List[Any]]] = None,
    id_col: Optional[str] = None,
    date_col: Optional[str] = None,
    corr_threshold: float = 0.70,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> ScreeningResult:
    """Screen every column of ``train`` except target, ID and date.

    ``train`` must be the Train sample only.
    """
    if not 0 <= iv_min < iv_max:
        raise ValueError(f"Need 0 <= iv_min < iv_max, got {iv_min}, {iv_max}")
    exclude, force = set(exclude), set(force_include)
    roles = {target, id_col, date_col} - {None}
    candidates = [c for c in train.columns if c not in roles]
    unknown = sorted((exclude | force) - set(candidates))
    if unknown:
        raise ValueError(f"EXCLUDE/FORCE_INCLUDE name columns that are not candidates: {unknown}")
    special_codes = dict(special_codes or {})
    fit_params = dict(fit_params or {})

    engine = BinningEngine(train, target, special_codes=special_codes,
                           exclude=roles - {target}, max_categories=max_categories)
    n = max(len(train), 1)
    rows = []
    for v in candidates:
        s = train[v]
        dtype = detect_dtype(s, max_categories)
        top = s.value_counts(dropna=True)
        row: Dict[str, Any] = {
            "variable": v, "type": dtype,
            "missing_rate": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
            "top_share": float(top.iloc[0] / n) if len(top) else 0.0,
            "iv": np.nan, "iv_band": "", "gini": np.nan, "ks": np.nan, "bins": np.nan,
            "monotonic": None, "warnings": "",
        }
        status = reason = None
        if dtype == "datetime":
            status, reason = EXCLUDED, "datetime column; set it as DATE_COL or derive features from it"
        elif row["n_unique"] <= 1:
            status, reason = CONSTANT, "a single distinct value"
        elif v not in force and is_id_like(s, dtype):
            status, reason = ID_LIKE, f"{row['n_unique']} distinct values in {s.notna().sum()} rows"

        if status is None:
            try:
                r = engine.fit(v, dtype=dtype, **fit_params)
                row.update(iv=r.iv, iv_band=interpret_iv(r.iv), gini=r.gini, ks=r.ks,
                           bins=len(r.regular_bins), monotonic=r.is_monotonic,
                           warnings=" | ".join(w for w in r.warnings if not w.startswith(_NOISE)))
            except Exception as exc:  # noqa: BLE001 - report, don't stop
                status, reason = FIT_ERROR, f"{type(exc).__name__}: {exc}"

        if v in exclude:
            note = "" if status is None else f" ({status}: {reason})"
            status, reason = EXCLUDED, "listed in EXCLUDE" + note
        elif status is None:
            iv = row["iv"]
            if v in force:
                status = FORCED
                reason = (f"FORCE_INCLUDE (IV {iv:.3f} > IV_MAX {iv_max})" if iv > iv_max else
                          f"FORCE_INCLUDE (IV {iv:.3f} < IV_MIN {iv_min})" if iv < iv_min else
                          "FORCE_INCLUDE")
            elif iv > iv_max:
                status = REVIEW
                reason = (f"IV {iv:.3f} > IV_MAX {iv_max}: check for leakage, then add to "
                          "FORCE_INCLUDE (known at decision time) or EXCLUDE")
            elif iv < iv_min:
                status, reason = LOW_IV, f"IV {iv:.3f} < IV_MIN {iv_min}"
            else:
                status, reason = SELECTED, f"IV_MIN {iv_min} <= IV {iv:.3f} <= IV_MAX {iv_max}"
        if status in (SELECTED, LOW_IV, REVIEW) and _borderline(row["iv"], iv_min, iv_max):
            reason += (" [borderline: within 10% of a threshold, the status can change with "
                       "the split; decide on business grounds]")
        row.update(status=status, reason=reason)
        rows.append(row)

    table = pd.DataFrame(rows, columns=_COLUMNS)
    order = {s: i for i, s in enumerate(STATUS_ORDER)}
    table = (table.assign(_o=table["status"].map(order), _iv=-table["iv"].fillna(-1))
             .sort_values(["_o", "_iv"]).drop(columns=["_o", "_iv"]).reset_index(drop=True))

    pairs = correlated_pairs(train, table, special_codes, corr_threshold)
    params = {"iv_min": iv_min, "iv_max": iv_max, "fit_params": fit_params,
              "exclude": sorted(exclude), "force_include": sorted(force),
              "special_codes": {k: list(v) for k, v in special_codes.items()},
              "corr_threshold": corr_threshold, "train_rows": int(len(train)),
              "train_events": int(train[target].sum())}
    return ScreeningResult(table=table, engine=engine, correlated_pairs=pairs, params=params)


def _borderline(iv: float, iv_min: float, iv_max: float, rel: float = 0.10) -> bool:
    """IV within ``rel`` (relative) of either threshold. IV is a sample
    estimate: a variable at 0.098 vs a 0.10 cutoff is not meaningfully weaker
    than one at 0.102."""
    return any(t > 0 and abs(iv - t) <= rel * t for t in (iv_min, iv_max))


def correlated_pairs(train: pd.DataFrame, table: pd.DataFrame,
                     special_codes: Dict[str, List[Any]], threshold: float) -> pd.DataFrame:
    """Spearman correlation between numerical candidates (special codes and
    missing values removed pairwise). Returns pairs with |rho| >= threshold."""
    cols = table.loc[(table["type"] == "numerical") &
                     ~table["status"].isin([ID_LIKE, CONSTANT, FIT_ERROR]), "variable"].tolist()
    out_cols = ["var_1", "var_2", "spearman", "status_1", "status_2", "higher_iv"]
    if len(cols) < 2:
        return pd.DataFrame(columns=out_cols)
    x = train[cols].apply(pd.to_numeric, errors="coerce")
    for c, codes in special_codes.items():
        if c in x.columns and codes:
            x[c] = x[c].mask(x[c].isin([float(v) for v in codes]))
    rho = x.corr(method="spearman", min_periods=30)
    info = table.set_index("variable")
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = rho.at[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                ia, ib = info.at[a, "iv"], info.at[b, "iv"]
                rows.append({"var_1": a, "var_2": b, "spearman": round(float(r), 3),
                             "status_1": info.at[a, "status"], "status_2": info.at[b, "status"],
                             "higher_iv": a if (ia if pd.notna(ia) else -1) >= (ib if pd.notna(ib) else -1) else b})
    df = pd.DataFrame(rows, columns=out_cols)
    return df.reindex(df["spearman"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Plots and report
# ═══════════════════════════════════════════════════════════════════════════


def plot_bins(result: Optional[BinningResult], ax=None):
    """Share of rows per bin (bars, left axis) and event rate (line, right axis).
    Missing and Special bins are drawn in grey at the end."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.4))
    if result is None:
        ax.axis("off")
        return ax
    t = result.summary()
    t = t[t["Count"] > 0].reset_index(drop=True)
    xs = np.arange(len(t))
    colors = ["#4C78A8" if k == "regular" else "#B0B0B0" for k in t["Kind"]]
    ax.bar(xs, t["Share"], color=colors, width=0.7)
    ax.set_ylabel("share of rows")
    ax.set_xticks(xs)
    ax.set_xticklabels([_short(lbl) for lbl in t["Label"]], rotation=35, ha="right", fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(xs, t["Event rate"], color="#E45756", marker="o", lw=1.6)
    ax2.set_ylabel("event rate", color="#E45756")
    ax2.set_ylim(bottom=0)
    ax.set_title(f"{result.variable}  (IV {result.iv:.3f})", fontsize=9)
    return ax


def _short(label: str, n: int = 18) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


def _fig_to_base64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


_STATUS_COLOR = {SELECTED: "#d9f0d9", FORCED: "#d6e6f5", REVIEW: "#fde2b8", LOW_IV: "#eeeeee",
                 EXCLUDED: "#eeeeee", ID_LIKE: "#f6d5d5", CONSTANT: "#f6d5d5", FIT_ERROR: "#f6d5d5"}


def _render_html(res: ScreeningResult, extra_html: str, max_plots: int) -> str:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    t = res.table.copy()
    for c in ("missing_rate", "top_share"):
        t[c] = t[c].map(lambda v: f"{v:.1%}")
    for c in ("iv", "gini", "ks"):
        t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    head = "".join(f"<th>{html.escape(c)}</th>" for c in t.columns)
    body = []
    for _, r in t.iterrows():
        color = _STATUS_COLOR.get(r["status"], "#fff")
        cells = "".join(f"<td>{html.escape(str(v if v is not None else ''))}</td>" for v in r)
        body.append(f'<tr style="background:{color}">{cells}</tr>')

    pairs = res.correlated_pairs
    pairs_html = ("<p>None above the threshold.</p>" if pairs.empty else
                  pairs.to_html(index=False, border=0, classes="t"))

    plots = []
    for v in [v for v in t["variable"] if res.result(v) is not None][:max_plots]:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        plot_bins(res.result(v), ax=ax)
        plots.append(f'<img alt="{html.escape(v)}" src="data:image/png;base64,{_fig_to_base64(fig)}">')

    p = res.params
    counts = " · ".join(f"{k}: {v}" for k, v in res.status_counts().items())
    review = res.needs_review
    review_html = (f'<p class="warn">Held for review (IV above {p["iv_max"]}): '
                   f'{html.escape(", ".join(review))}. Put each in FORCE_INCLUDE or EXCLUDE.</p>'
                   if review else "")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Screening report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0 20px}}
th,td{{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f4f4f4}} .warn{{background:#fde2b8;padding:8px;border-radius:4px}}
img{{width:420px;margin:4px}} .muted{{color:#666;font-size:12px}}
</style></head><body>
<h1>Univariate screening (Train only)</h1>
<p class="muted">Train rows: {p["train_rows"]:,} · events: {p["train_events"]:,} ·
IV_MIN {p["iv_min"]} · IV_MAX {p["iv_max"]} · binning: {html.escape(json.dumps(p["fit_params"]))}</p>
{extra_html}
<p><b>Status:</b> {html.escape(counts)}</p>
{review_html}
<h2>Variables</h2>
<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>
<h2>Correlated numerical pairs (|Spearman| ≥ {p["corr_threshold"]})</h2>
{pairs_html}
<p class="muted">Flagged only. Decide on redundancy at variable selection, on WOE.</p>
<h2>Event rate by bin</h2>
<p class="muted">Bars: share of Train rows per bin (grey = Missing / Special). Line: event rate.</p>
{''.join(plots)}
</body></html>"""
