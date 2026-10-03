"""
Portable scoring artifacts: apply fitted bins anywhere.

A :class:`ScoringArtifact` holds the rules for one variable; a
:class:`ScoringBundle` packs all variables and exports them as

* ``bundle.json``  - machine-readable rules (reload with ``ScoringBundle.load_json``)
* ``scorer.py``    - dependency-free Python module (stdlib only)
* SQL              - ``SELECT *, CASE WHEN ...`` for standard SQL, Spark or BigQuery

All of them implement the same rules as :meth:`ScoringArtifact.transform_series`;
``tests/test_exports_parity.py`` checks that they produce identical output.

Scoring order for every value: Missing -> Special -> regular bins.
Unseen categories go to the Missing bin.
"""

from __future__ import annotations

import datetime as _dt
import json
import keyword
import re
import textwrap
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .binning import MISSING_GROUP, SPECIAL_GROUP, BinningResult
from .dtypes import is_missing, series_to_category_str, to_category_str

ARTIFACT_SCHEMA_VERSION = 1
Dialect = Literal["standard", "spark", "bigquery"]
SQL_DIALECTS = ("standard", "spark", "bigquery")


# ═══════════════════════════════════════════════════════════════════════════
# ScoringArtifact
# ═══════════════════════════════════════════════════════════════════════════

class ScoringArtifact:
    """Self-contained scoring rules for one variable."""

    def __init__(self, variable: str, dtype: str, value_type: str, bins: List[dict],
                 special_codes: Sequence[Any] = (), iv: float = 0.0, gini: float = 0.0,
                 ks: float = 0.0):
        if dtype not in ("numerical", "categorical"):
            raise ValueError(f"Unknown dtype {dtype!r}")
        self.variable = variable
        self.dtype = dtype
        self.value_type = value_type
        self.special_codes = list(special_codes)
        self.iv, self.gini, self.ks = float(iv), float(gini), float(ks)
        self.bins = [dict(b) for b in bins]

        self.regular = [b for b in self.bins if b["kind"] == "regular"]
        self.missing = next(b for b in self.bins if b["kind"] == "missing")
        self.special = next((b for b in self.bins if b["kind"] == "special"), None)
        if self.special_codes and self.special is None:
            raise ValueError(f"{variable}: special codes configured but no Special bin.")

        if dtype == "numerical":
            self.cutoffs = [b["upper"] for b in self.regular[:-1]]
            self._special_num = {float(s) for s in self.special_codes}
        else:
            self._cat_index = {c: i for i, b in enumerate(self.regular) for c in b["categories"]}
            self._special_str = {to_category_str(s) for s in self.special_codes}

    # ---------------------------------------------------------------- build / io

    @classmethod
    def from_result(cls, result: BinningResult) -> "ScoringArtifact":
        bins = [{
            "group": b.group, "kind": b.kind, "label": b.label,
            "lower": b.lower, "upper": b.upper,
            "categories": list(b.categories) if b.categories is not None else None,
            "woe": float(b.woe), "event_rate": float(b.event_rate), "count": int(b.count),
        } for b in result.bins]
        return cls(result.variable, result.dtype, result.value_type, bins,
                   special_codes=result.special_codes, iv=result.iv,
                   gini=result.gini, ks=result.ks)

    def to_dict(self) -> dict:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "variable": self.variable, "dtype": self.dtype, "value_type": self.value_type,
            "iv": self.iv, "gini": self.gini, "ks": self.ks,
            "special_codes": self.special_codes, "bins": self.bins,
        }

    # backwards-compatible name used by the Dataiku version
    to_json = to_dict

    @classmethod
    def from_dict(cls, d: dict) -> "ScoringArtifact":
        if d.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact schema_version {d.get('schema_version')!r}")
        return cls(d["variable"], d["dtype"], d["value_type"], d["bins"],
                   special_codes=d.get("special_codes", []), iv=d.get("iv", 0.0),
                   gini=d.get("gini", 0.0), ks=d.get("ks", 0.0))

    # ---------------------------------------------------------------- scoring

    @staticmethod
    def _out(b: dict) -> dict:
        return {"group": b["group"], "woe": b["woe"], "label": b["label"]}

    def score_value(self, value) -> dict:
        """Score one raw value -> ``{"group", "woe", "label"}``."""
        if is_missing(value):
            return self._out(self.missing)
        if self.dtype == "numerical":
            v = float(value)
            if v in self._special_num:
                return self._out(self.special)
            for b in self.regular:
                if b["upper"] is None or v <= b["upper"]:
                    return self._out(b)
            return self._out(self.regular[-1])  # unreachable: last bin is unbounded
        s = to_category_str(value)
        if s in self._special_str:
            return self._out(self.special)
        i = self._cat_index.get(s)
        return self._out(self.missing if i is None else self.regular[i])

    def transform_series(self, series: pd.Series) -> Dict[str, np.ndarray]:
        """Vectorised scoring of a column -> arrays ``group``, ``woe``, ``label``."""
        n_reg = len(self.regular)
        table = self.regular + ([self.special] if self.special else []) + [self.missing]
        i_special, i_missing = n_reg, len(table) - 1
        if self.dtype == "numerical":
            miss = series.isna().to_numpy()
            vals = pd.to_numeric(series, errors="raise").astype("float64").to_numpy()
            idx = np.searchsorted(np.asarray(self.cutoffs, dtype=float), vals, side="left")
            if self.special:
                idx = np.where(np.isin(vals, list(self._special_num)) & ~miss, i_special, idx)
            idx = np.where(miss, i_missing, idx)
        else:
            strs = series_to_category_str(series)
            miss = strs.isna().to_numpy()
            idx = strs.map(self._cat_index).fillna(i_missing).to_numpy(dtype=int)
            if self.special:
                idx = np.where(strs.isin(self._special_str).to_numpy() & ~miss, i_special, idx)
            idx = np.where(miss, i_missing, idx)
        groups = np.array([b["group"] for b in table], dtype=int)
        woes = np.array([b["woe"] for b in table], dtype=float)
        labels = np.array([b["label"] for b in table], dtype=object)
        return {"group": groups[idx], "woe": woes[idx], "label": labels[idx]}

    # ---------------------------------------------------------------- python export

    def to_python(self, func_name: Optional[str] = None) -> str:
        """Standalone function; relies on ``_is_missing``/``_cat_str`` from the
        module header produced by :meth:`ScoringBundle.to_python_module`."""
        fn = func_name or f"score_{_py_ident(self.variable)}"
        ret = lambda b: (f'return {{"group": {b["group"]}, "woe": {b["woe"]!r}, '
                         f'"label": {b["label"]!r}}}')
        L = [f"def {fn}(value):",
             f'    """{_doc_safe(self.variable)} ({self.dtype}). IV={self.iv:.4f} '
             f'Gini={self.gini:.4f} KS={self.ks:.4f}"""',
             "    if _is_missing(value):",
             f"        {ret(self.missing)}"]
        if self.dtype == "numerical":
            L.append("    v = float(value)")
            if self.special:
                codes = ", ".join(repr(float(s)) for s in self.special_codes)
                L += [f"    if v in ({codes},):", f"        {ret(self.special)}"]
            for b in self.regular[:-1]:
                L += [f"    if v <= {b['upper']!r}:", f"        {ret(b)}"]
            L.append(f"    {ret(self.regular[-1])}")
        else:
            L.append("    s = _cat_str(value)")
            if self.special:
                codes = ", ".join(repr(c) for c in sorted(self._special_str))
                L += [f"    if s in ({codes},):", f"        {ret(self.special)}"]
            for b in self.regular:
                cats = ", ".join(repr(c) for c in b["categories"])
                L += [f"    if s in ({cats},):", f"        {ret(b)}"]
            L.append(f"    {ret(self.missing)}  # unseen category")
        return "\n".join(L)

    # ---------------------------------------------------------------- sql export

    def _sql_value(self, raw: Any, dialect: Dialect) -> str:
        """Literal for a category / special code, typed like the source column."""
        if self.value_type == "boolean":
            return "TRUE" if to_category_str(raw) == "True" else "FALSE"
        if self.value_type == "numeric" or self.dtype == "numerical":
            f = float(raw)
            return str(int(f)) if f.is_integer() else repr(f)
        return quote_literal(str(raw), dialect)

    def to_sql(self, dialect: Dialect = "standard", prefix: str = "opt_",
               metrics: Sequence[str] = ("group", "woe")) -> str:
        """CASE WHEN expressions (one per metric) for this variable."""
        _check_dialect(dialect)
        col = quote_ident(self.variable, dialect)

        def num(x):
            return repr(float(x))

        def case(metric: str) -> str:
            val = (lambda b: str(b["group"])) if metric == "group" else (lambda b: num(b["woe"]))
            L = ["CASE", f"    WHEN {col} IS NULL THEN {val(self.missing)}"]
            if self.special:
                codes = ", ".join(self._sql_value(s, dialect) for s in self.special_codes)
                L.append(f"    WHEN {col} IN ({codes}) THEN {val(self.special)}")
            if self.dtype == "numerical":
                for b in self.regular[:-1]:
                    L.append(f"    WHEN {col} <= {num(b['upper'])} THEN {val(b)}")
                L.append(f"    ELSE {val(self.regular[-1])}")
            else:
                for b in self.regular:
                    cats = ", ".join(self._sql_value(c, dialect) for c in b["categories"])
                    L.append(f"    WHEN {col} IN ({cats}) THEN {val(b)}")
                L.append(f"    ELSE {val(self.missing)}  -- unseen category")
            L.append(f"END AS {quote_ident(f'{prefix}{self.variable}_{metric}', dialect)}")
            return "\n".join(L)

        bad = set(metrics) - {"group", "woe"}
        if bad:
            raise ValueError(f"SQL export supports metrics 'group' and 'woe', not {sorted(bad)}")
        header = f"-- {_sql_comment_safe(self.variable)}  IV={self.iv:.4f}"
        return header + "\n" + ",\n".join(case(m) for m in metrics)


# ═══════════════════════════════════════════════════════════════════════════
# ScoringBundle
# ═══════════════════════════════════════════════════════════════════════════

class ScoringBundle:
    """All variables' scoring rules, packed for deployment."""

    def __init__(self, artifacts: Dict[str, ScoringArtifact], name: str = "binning_bundle",
                 description: str = "", created_at: Optional[str] = None):
        self.artifacts = dict(artifacts)
        self.name = name
        self.description = description
        self.created_at = created_at or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    # ---------------------------------------------------------------- scoring

    def score_record(self, record: dict, prefix: str = "opt_") -> dict:
        out = {}
        for var, art in self.artifacts.items():
            r = art.score_value(record.get(var))
            for m in ("group", "woe", "label"):
                out[f"{prefix}{var}_{m}"] = r[m]
        return out

    def score_dataframe(self, df: pd.DataFrame, metrics: Sequence[str] = ("woe", "group"),
                        prefix: str = "opt_") -> pd.DataFrame:
        """Return a copy of ``df`` with ``{prefix}{var}_{metric}`` columns added.
        Variables absent from ``df`` raise, so a scoring job never runs silently
        on a partial input."""
        absent = [v for v in self.artifacts if v not in df.columns]
        if absent:
            raise KeyError(f"Columns missing from input: {absent}")
        new = {}
        for var, art in self.artifacts.items():
            res = art.transform_series(df[var])
            for m in metrics:
                new[f"{prefix}{var}_{m}"] = res[m]
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # ---------------------------------------------------------------- json

    def to_dict(self) -> dict:
        from . import __version__
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "package_version": __version__,
            "name": self.name, "description": self.description, "created_at": self.created_at,
            "variables": {v: a.to_dict() for v, a in self.artifacts.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, d: dict) -> "ScoringBundle":
        if d.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported bundle schema_version {d.get('schema_version')!r}")
        arts = {v: ScoringArtifact.from_dict(a) for v, a in d["variables"].items()}
        return cls(arts, name=d.get("name", ""), description=d.get("description", ""),
                   created_at=d.get("created_at"))

    @classmethod
    def from_json(cls, text: str) -> "ScoringBundle":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load_json(cls, path: str) -> "ScoringBundle":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())

    def save_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path

    # ---------------------------------------------------------------- python module

    def to_python_module(self, prefix: str = "opt_") -> str:
        """Self-contained ``scorer.py`` (stdlib only)."""
        names, used = {}, set()
        for var in self.artifacts:
            base = f"score_{_py_ident(var)}"
            name, k = base, 2
            while name in used:
                name, k = f"{base}_{k}", k + 1
            used.add(name)
            names[var] = name

        L = ['"""',
             "scorer.py - generated by scorecard_studio. Do not edit by hand.",
             f"Bundle: {_doc_safe(self.name)}",
             f"Generated: {self.created_at}",
             "",
             "Rules: Missing -> Special -> bins. Numerical bins are (lower, upper].",
             "Unseen categories score as Missing.",
             '"""',
             "",
             f"PREFIX = {prefix!r}",
             "",
             _PY_HELPERS,
             ""]
        for var, art in self.artifacts.items():
            L += [art.to_python(names[var]), "", ""]
        L.append("SCORERS = {")
        for var in self.artifacts:
            L.append(f"    {var!r}: {names[var]},")
        L += ["}", "", _PY_RUNNERS]
        return "\n".join(L)

    def save_python(self, path: str = "scorer.py", prefix: str = "opt_") -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_python_module(prefix=prefix))
        return path

    # ---------------------------------------------------------------- sql

    def to_sql(self, source_table: str = "input_table", dialect: Dialect = "standard",
               prefix: str = "opt_") -> str:
        """``SELECT *, <CASE WHEN ...> FROM source_table``.
        ``source_table`` is inserted verbatim so it can be ``schema.table``."""
        _check_dialect(dialect)
        blocks = [textwrap.indent(a.to_sql(dialect=dialect, prefix=prefix), "    ")
                  for a in self.artifacts.values()]
        return "\n".join([
            f"-- Generated by scorecard_studio ({self.created_at}), dialect={dialect}",
            f"-- Bundle: {_sql_comment_safe(self.name)}",
            "SELECT",
            "    *,",
            ",\n".join(blocks),
            f"FROM {source_table}",
        ])

    def save_sql(self, path: str = "transform.sql", source_table: str = "input_table",
                 dialect: Dialect = "standard", prefix: str = "opt_") -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_sql(source_table=source_table, dialect=dialect, prefix=prefix))
        return path

    # ---------------------------------------------------------------- scorecard

    def to_scorecard_table(self, coefficients: Dict[str, float], intercept: float,
                           pdo: float = 20.0, base_score: float = 600.0,
                           base_odds: float = 50.0, round_points: Optional[int] = None) -> pd.DataFrame:
        """Points per bin. Needs the fitted logistic-regression coefficients:
        points depend on beta * WOE, not on WOE alone. See :mod:`scaling`."""
        from .scaling import ScalingParams, scorecard_table
        return scorecard_table(self, coefficients, intercept,
                               ScalingParams(pdo=pdo, base_score=base_score, base_odds=base_odds),
                               round_points=round_points)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_PY_HELPERS = '''\
def _is_missing(v):
    """None, NaN, NaT or pandas.NA."""
    if v is None:
        return True
    try:
        return bool(v != v)
    except Exception:  # pandas.NA cannot be coerced to bool
        return True


def _cat_str(v):
    """Canonical category string; must match scorecard_studio.dtypes.to_category_str."""
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            v = v.item()  # numpy scalar -> python scalar
        except Exception:
            pass
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == v and v not in (float("inf"), float("-inf")) and v.is_integer():
            return str(int(v))
        return repr(v)
    return str(v)'''

_PY_RUNNERS = '''\
def score_record(record):
    """{"AGE": 34, ...} -> {"opt_AGE_group": 2, "opt_AGE_woe": -0.51, "opt_AGE_label": ...}"""
    out = {}
    for var, fn in SCORERS.items():
        r = fn(record.get(var))
        for m in ("group", "woe", "label"):
            out[PREFIX + var + "_" + m] = r[m]
    return out


def score_dataframe(df, metrics=("woe", "group")):
    """Return a copy of a pandas DataFrame with the transformed columns added."""
    df = df.copy()
    for var, fn in SCORERS.items():
        res = [fn(v) for v in df[var].tolist()]
        for m in metrics:
            df[PREFIX + var + "_" + m] = [r[m] for r in res]
    return df
'''


def _check_dialect(dialect: str):
    if dialect not in SQL_DIALECTS:
        raise ValueError(f"dialect must be one of {SQL_DIALECTS}, got {dialect!r}")


def quote_ident(name: str, dialect: Dialect = "standard") -> str:
    """Quote a column name so spaces, quotes and reserved words are safe."""
    name = str(name)
    if dialect == "standard":
        return '"' + name.replace('"', '""') + '"'
    if dialect == "spark":
        return "`" + name.replace("`", "``") + "`"
    return "`" + name.replace("\\", "\\\\").replace("`", "\\`") + "`"  # bigquery


def quote_literal(value: str, dialect: Dialect = "standard") -> str:
    """Quote a string literal (escapes embedded quotes)."""
    if dialect == "standard":
        return "'" + value.replace("'", "''") + "'"
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"  # spark, bigquery


def _py_ident(name: str) -> str:
    s = re.sub(r"\W", "_", str(name)).lower().strip("_") or "var"
    if s[0].isdigit():
        s = "v_" + s
    if keyword.iskeyword(s):
        s += "_"
    return s


def _doc_safe(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"""', "'''")


def _sql_comment_safe(text: str) -> str:
    return str(text).replace("\n", " ").replace("\r", " ")
