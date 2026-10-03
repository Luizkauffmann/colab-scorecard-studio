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


def make_credit_risk_like(n: int = 6000, seed: int = 3) -> pd.DataFrame:
    """Same schema as the Kaggle Credit Risk Dataset, generated, so CI needs
    neither network nor the real file. Includes the real file's quirks:
    missing emp_length / int_rate and a few impossible ages and tenures."""
    rng = np.random.default_rng(seed)
    grade = rng.choice(list("ABCDEFG"), n, p=[0.33, 0.32, 0.2, 0.11, 0.03, 0.007, 0.003])
    g = np.array([list("ABCDEFG").index(v) for v in grade])
    income = np.round(np.exp(rng.normal(10.9, 0.5, n)), -2).astype(int)
    amount = np.clip(np.round(rng.gamma(2.2, 4300, n), -2), 500, 35000).astype(int)
    pct = np.round(amount / income, 2)
    home = rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[0.5, 0.41, 0.08, 0.01])
    age = np.clip(rng.normal(28, 6, n).round(), 20, 80).astype(int)
    emp = np.clip(rng.exponential(4.5, n).round(), 0, 40)
    lin = (-1.6 + 0.55 * g + 6.0 * pct - 0.6 * (np.log(income) - 10.9)
           + np.vectorize({"RENT": 0.5, "MORTGAGE": -0.3, "OWN": -1.0, "OTHER": 0.3}.get)(home)
           + rng.normal(0, 0.4, n))
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    df = pd.DataFrame({
        "person_age": age, "person_income": income, "person_home_ownership": home,
        "person_emp_length": np.where(rng.random(n) < 0.03, np.nan, emp),
        "loan_intent": rng.choice(["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
                                   "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], n),
        "loan_grade": grade, "loan_amnt": amount,
        "loan_int_rate": np.where(rng.random(n) < 0.1, np.nan, np.round(7 + 2.4 * g + rng.normal(0, 1, n), 2)),
        "loan_status": y, "loan_percent_income": pct,
        "cb_person_default_on_file": np.where(rng.random(n) < 0.18 + 0.1 * (g >= 3), "Y", "N"),
        "cb_person_cred_hist_length": np.clip(age - 20 + rng.integers(-2, 3, n), 2, 30),
    })
    df.loc[[5, 17, 99], "person_age"] = [123, 144, 144]
    df.loc[[7, 18], "person_emp_length"] = 123.0
    return df


@pytest.fixture
def credit_risk_like() -> pd.DataFrame:
    return make_credit_risk_like()
