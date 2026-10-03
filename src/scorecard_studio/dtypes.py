"""
Variable-type detection and value canonicalisation.

This module is the single source of truth for deciding whether a column is
binned as *numerical* or *categorical*. The engine, the web app backend and
the notebook all call :func:`detect_dtype` so they can never disagree.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd

DType = Literal["numerical", "categorical", "datetime"]
ValueType = Literal["numeric", "string", "boolean"]

#: Numeric columns with at most this many distinct values are treated as
#: categorical (e.g. coded fields such as EDUCATION_LEVEL = 1..5).
DEFAULT_MAX_CATEGORIES = 10


def detect_dtype(series: pd.Series, max_categories: int = DEFAULT_MAX_CATEGORIES) -> DType:
    """Decide how a column should be binned.

    * booleans                                   -> categorical
    * datetimes                                  -> datetime (not binned)
    * numeric with > ``max_categories`` distinct -> numerical
    * everything else (strings, low-cardinality) -> categorical
    """
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical" if series.nunique(dropna=True) > max_categories else "categorical"
    return "categorical"


def infer_variable_types(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    exclude: Optional[Iterable[str]] = None,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> Dict[str, DType]:
    """Return ``{column: dtype}`` for every candidate predictor in ``df``."""
    skip = set(exclude or [])
    if target_col is not None:
        skip.add(target_col)
    return {
        c: detect_dtype(df[c], max_categories=max_categories)
        for c in df.columns
        if c not in skip
    }


def value_type_of(series: pd.Series) -> ValueType:
    """Physical type of a column's values; drives literal formatting in SQL."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "string"


def is_missing(value) -> bool:
    """True for None, NaN, NaT and pandas.NA."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def to_category_str(value) -> Optional[str]:
    """Canonical string form of a categorical value.

    Integral floats lose their ``.0`` so that a coded column read as float
    (because it contains NaN) maps to the same categories as the int column:
    ``3.0 -> "3"``. Missing values return ``None``. The generated
    ``scorer.py`` embeds an identical function, which the parity tests check.
    """
    if is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isfinite(f) and f.is_integer():
            return str(int(f))
        return repr(f)
    return str(value)


def series_to_category_str(series: pd.Series) -> pd.Series:
    """Vectorised :func:`to_category_str` (returns object dtype, ``None`` for missing)."""
    return series.map(to_category_str).astype(object)
