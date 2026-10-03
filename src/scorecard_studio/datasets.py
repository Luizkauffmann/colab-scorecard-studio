"""
Synthetic demo data, so the template runs end to end without real data.

The credit application dataset deliberately contains the things real data
has and toy data usually lacks: missing values that carry risk, special
codes, rare categories, coded categoricals, a boolean, an application date
and a mild drift in the default rate over time (useful for OOT checks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Special codes used in MONTHS_SINCE_LAST_DELINQ:
#: -999 = never delinquent, -998 = no bureau file.
CREDIT_DEMO_SPECIAL_CODES = {"MONTHS_SINCE_LAST_DELINQ": [-999, -998]}
CREDIT_DEMO_TARGET = "DEFAULT_12M"
CREDIT_DEMO_ID = "APPLICATION_ID"
CREDIT_DEMO_DATE = "APPLICATION_DATE"


def _calibrate_intercept(linear: np.ndarray, target_rate: float) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(mid + linear)))) < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def make_credit_application_data(n: int = 20_000, default_rate: float = 0.08,
                                 seed: int = 42) -> pd.DataFrame:
    """Synthetic application-scorecard dataset with target ``DEFAULT_12M``.

    Columns: APPLICATION_ID, APPLICATION_DATE (24 months), AGE, MONTHLY_INCOME
    (~6% missing), DEBT_TO_INCOME, CREDIT_UTILIZATION, NUM_INQUIRIES_6M,
    MONTHS_SINCE_LAST_DELINQ (special codes -999/-998), EMPLOYMENT_TYPE (with a
    rare category), EDUCATION_LEVEL (codes 1-5, some missing), HOME_OWNERSHIP,
    HAS_COSIGNER (bool), DEFAULT_12M (0/1).
    """
    rng = np.random.default_rng(seed)

    months = rng.integers(0, 24, n)
    dates = (pd.Timestamp("2023-01-01") + pd.to_timedelta(months * 30.4 + rng.integers(0, 28, n), unit="D")).normalize()

    age = np.clip(rng.normal(41, 12, n), 18, 80).round()
    income = np.round(np.exp(rng.normal(8.4, 0.55, n)), -1)
    dti = np.clip(rng.gamma(2.2, 0.16, n), 0, 1.8).round(3)
    util = np.clip(rng.beta(1.6, 2.4, n) * 1.15, 0, 1.2).round(3)
    inquiries = np.clip(rng.poisson(1.6, n) + (rng.random(n) < 0.08) * rng.integers(3, 9, n), 0, 20)

    emp_levels = ["Employed", "Self-employed", "Retired", "Unemployed", "Student", "Contractor"]
    emp = rng.choice(emp_levels, n, p=[0.52, 0.15, 0.13, 0.08, 0.105, 0.015])
    edu = rng.choice([1, 2, 3, 4, 5], n, p=[0.08, 0.32, 0.30, 0.22, 0.08]).astype(float)
    home = rng.choice(["Mortgage", "Rent", "Own", "Other"], n, p=[0.40, 0.38, 0.19, 0.03])
    cosigner = rng.random(n) < 0.12

    delinq_state = rng.choice(["never", "nofile", "some"], n, p=[0.55, 0.04, 0.41])
    months_delinq = np.where(delinq_state == "never", -999,
                             np.where(delinq_state == "nofile", -998,
                                      rng.integers(1, 85, n))).astype(float)

    emp_eff = {"Employed": -0.35, "Self-employed": 0.05, "Retired": -0.25,
               "Unemployed": 0.95, "Student": 0.45, "Contractor": 0.30}
    home_eff = {"Mortgage": -0.25, "Rent": 0.25, "Own": -0.35, "Other": 0.30}
    delinq_eff = np.where(delinq_state == "never", -0.55,
                          np.where(delinq_state == "nofile", 0.45,
                                   0.85 * np.exp(-months_delinq.clip(1) / 24)))

    linear = (
        -0.022 * (age - 41)
        - 0.75 * (np.log(income) - 8.4)
        + 1.35 * dti
        + 1.50 * util
        + 0.17 * inquiries
        + delinq_eff
        + np.vectorize(emp_eff.get)(emp)
        - 0.11 * (edu - 3)
        + np.vectorize(home_eff.get)(home)
        - 0.35 * cosigner
        + 0.012 * months          # mild deterioration over time
        + rng.normal(0, 0.35, n)
    )

    # Missing income is not random: thin-file and self-declared applicants
    # who skip it are riskier, so the Missing bin should carry information.
    p_income_missing = 0.03 + 0.06 * (linear > np.quantile(linear, 0.7))
    income_missing = rng.random(n) < p_income_missing
    edu_missing = rng.random(n) < 0.03

    b0 = _calibrate_intercept(linear, default_rate)
    p = 1 / (1 + np.exp(-(b0 + linear)))
    target = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        CREDIT_DEMO_ID: [f"APP{i:07d}" for i in range(1, n + 1)],
        CREDIT_DEMO_DATE: dates,
        "AGE": age,
        "MONTHLY_INCOME": np.where(income_missing, np.nan, income),
        "DEBT_TO_INCOME": dti,
        "CREDIT_UTILIZATION": util,
        "NUM_INQUIRIES_6M": inquiries.astype(int),
        "MONTHS_SINCE_LAST_DELINQ": months_delinq,
        "EMPLOYMENT_TYPE": emp,
        "EDUCATION_LEVEL": np.where(edu_missing, np.nan, edu),
        "HOME_OWNERSHIP": home,
        "HAS_COSIGNER": cosigner,
        CREDIT_DEMO_TARGET: target,
    })
    return df.sort_values(CREDIT_DEMO_DATE, kind="stable").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Default demo dataset: Kaggle "Credit Risk Dataset" (laotse/credit-risk-dataset)
# ═══════════════════════════════════════════════════════════════════════════

CREDIT_RISK_TARGET = "loan_status"
CREDIT_RISK_COLUMNS = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]
CREDIT_RISK_N_ROWS = 32_581
CREDIT_RISK_OPENML_ID = 43454
CREDIT_RISK_KAGGLE = "laotse/credit-risk-dataset"

#: Lender-assigned outputs of a previous risk assessment. Using them in an
#: application PD model means re-learning the old scorecard.
CREDIT_RISK_EXCLUDE = ["loan_grade", "loan_int_rate"]

#: Values outside these ranges are data errors (ages of 123 and 144, 123 years
#: of employment), not extreme applicants.
CREDIT_RISK_PLAUSIBILITY = {"person_age": (18, 100), "person_emp_length": (0, 60)}

_PACKAGED_FILE = "credit_risk_dataset.csv.gz"


def load_credit_risk_dataset(source: str = "auto") -> pd.DataFrame:
    """Load the Kaggle Credit Risk Dataset (target ``loan_status``, 1 = default).

    ``source``: ``"packaged"`` (copy shipped with this package), ``"openml"``
    (OpenML dataset 43454, no login, needs scikit-learn and internet) or
    ``"auto"`` (packaged if present, else OpenML).
    """
    if source not in ("auto", "packaged", "openml"):
        raise ValueError("source must be 'auto', 'packaged' or 'openml'")
    if source in ("auto", "packaged"):
        df = _read_packaged()
        if df is not None:
            return _normalise_credit_risk(df)
        if source == "packaged":
            raise FileNotFoundError(f"{_PACKAGED_FILE} is not shipped with this install.")
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as exc:  # pragma: no cover - Colab ships scikit-learn
        raise ImportError("Loading from OpenML needs scikit-learn: pip install scikit-learn") from exc
    frame = fetch_openml(data_id=CREDIT_RISK_OPENML_ID, as_frame=True, parser="auto").frame
    return _normalise_credit_risk(frame)


def _read_packaged():
    from importlib import resources
    try:
        ref = resources.files("scorecard_studio").joinpath("data", _PACKAGED_FILE)
        if not ref.is_file():
            return None
        with resources.as_file(ref) as path:
            return pd.read_csv(path)
    except (FileNotFoundError, ModuleNotFoundError):
        return None


def _normalise_credit_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Same column order and dtypes whatever the source (CSV or OpenML ARFF)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in CREDIT_RISK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Not the Credit Risk Dataset: columns {missing} are missing.")
    df = df[CREDIT_RISK_COLUMNS]
    for c in ("person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"):
        s = df[c].astype(object)
        df[c] = s.where(s.notna(), None).map(lambda v: None if v is None else str(v)).astype(object)
    for c in ("person_age", "person_income", "loan_amnt", "cb_person_cred_hist_length"):
        df[c] = pd.to_numeric(df[c]).astype("int64")
    for c in ("person_emp_length", "loan_int_rate", "loan_percent_income"):
        df[c] = pd.to_numeric(df[c]).astype("float64")
    df[CREDIT_RISK_TARGET] = pd.to_numeric(df[CREDIT_RISK_TARGET]).astype("int64")
    return df.reset_index(drop=True)
