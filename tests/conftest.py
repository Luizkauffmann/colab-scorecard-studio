import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", module="cvxpy")

from scorecard_studio import BinningEngine, make_credit_application_data  # noqa: E402
from scorecard_studio.datasets import (  # noqa: E402
    CREDIT_DEMO_DATE,
    CREDIT_DEMO_ID,
    CREDIT_DEMO_SPECIAL_CODES,
    CREDIT_DEMO_TARGET,
)


@pytest.fixture(scope="session")
def demo_df() -> pd.DataFrame:
    return make_credit_application_data(n=8000, seed=7)


@pytest.fixture(scope="session")
def fitted_engine(demo_df) -> BinningEngine:
    eng = BinningEngine(demo_df, CREDIT_DEMO_TARGET, special_codes=CREDIT_DEMO_SPECIAL_CODES,
                        exclude=[CREDIT_DEMO_ID, CREDIT_DEMO_DATE])
    eng.fit_all()
    assert not eng.fit_errors, eng.fit_errors
    return eng


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Hand-checkable data: x in 1..10, events at x >= 8, plus missing rows."""
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, np.nan]
    y = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0]
    return pd.DataFrame({"x": x, "y": y})
