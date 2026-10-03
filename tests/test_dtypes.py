"""Issues #7 and #8: one dtype detector used everywhere; forced categoricals honoured."""

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import BinningEngine, detect_dtype, infer_variable_types
from scorecard_studio.dtypes import to_category_str


@pytest.mark.parametrize("series,expected", [
    (pd.Series(np.arange(100, dtype=float)), "numerical"),
    (pd.Series(np.arange(100) % 5), "categorical"),           # coded numeric
    (pd.Series([1.0, 2.0, np.nan] * 10), "categorical"),
    (pd.Series(["a", "b"] * 50), "categorical"),
    (pd.Series([True, False] * 50), "categorical"),
    (pd.Series(pd.date_range("2024-01-01", periods=50)), "datetime"),
])
def test_detect_dtype(series, expected):
    assert detect_dtype(series) == expected


def test_threshold_is_configurable():
    s = pd.Series(np.arange(100) % 12)
    assert detect_dtype(s) == "numerical"
    assert detect_dtype(s, max_categories=15) == "categorical"


def test_engine_uses_the_same_detector(demo_df):
    eng = BinningEngine(demo_df, "DEFAULT_12M", exclude=["APPLICATION_ID"])
    assert eng.variable_types() == infer_variable_types(
        demo_df, "DEFAULT_12M", exclude=["APPLICATION_ID"])
    assert "DEFAULT_12M" not in eng.variable_types()


def test_fit_all_skips_datetimes_and_honours_forced_categoricals(demo_df):
    eng = BinningEngine(demo_df, "DEFAULT_12M", exclude=["APPLICATION_ID"])
    eng.fit_all(variables=["NUM_INQUIRIES_6M", "AGE", "APPLICATION_DATE"],
                categorical_variables=["NUM_INQUIRIES_6M"], max_bins=4)
    assert eng.get_result("NUM_INQUIRIES_6M").dtype == "categorical"
    assert eng.get_result("AGE").dtype == "numerical"
    assert "APPLICATION_DATE" in eng.fit_errors  # reported, not raised


@pytest.mark.parametrize("value,expected", [
    (3, "3"), (3.0, "3"), (np.int64(3), "3"), (np.float32(3.0), "3"),
    (2.5, "2.5"), ("x", "x"), (True, "True"), (np.bool_(False), "False"),
    (None, None), (np.nan, None), (pd.NA, None),
])
def test_to_category_str(value, expected):
    assert to_category_str(value) == expected
