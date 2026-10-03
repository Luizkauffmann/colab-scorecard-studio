"""Stage 2: Train/Test stratified, OOT strictly later in time."""

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import make_split, split_summary, split_warnings
from scorecard_studio.datasets import CREDIT_DEMO_DATE, CREDIT_DEMO_TARGET


def test_stratified_split_keeps_event_rate_and_is_reproducible(credit_risk_like):
    s1 = make_split(credit_risk_like, "loan_status", test_size=0.3, seed=1)
    s2 = make_split(credit_risk_like, "loan_status", test_size=0.3, seed=1)
    s3 = make_split(credit_risk_like, "loan_status", test_size=0.3, seed=2)
    assert s1.equals(s2) and not s1.equals(s3)
    assert set(s1) == {"train", "test"}
    y = credit_risk_like["loan_status"]
    for cls in (0, 1):
        n = int((y == cls).sum())
        assert int(((s1 == "test") & (y == cls)).sum()) == round(n * 0.3)
    rates = y.groupby(s1).mean()
    assert abs(rates["train"] - rates["test"]) < 0.002


def test_time_split_puts_latest_dates_in_oot(demo_df):
    s = make_split(demo_df, CREDIT_DEMO_TARGET, date_col=CREDIT_DEMO_DATE, oot_share=0.2, seed=0)
    d = demo_df[CREDIT_DEMO_DATE]
    assert d[s == "oot"].min() > d[s != "oot"].max()
    # a single date never straddles development and OOT
    per_day = s.eq("oot").groupby(d).nunique()
    assert (per_day == 1).all()
    assert 0.15 < (s == "oot").mean() < 0.25


def test_time_split_with_explicit_start(demo_df):
    s = make_split(demo_df, CREDIT_DEMO_TARGET, date_col=CREDIT_DEMO_DATE, oot_start="2024-07-01")
    assert (demo_df.loc[s == "oot", CREDIT_DEMO_DATE] >= "2024-07-01").all()
    assert (demo_df.loc[s != "oot", CREDIT_DEMO_DATE] < "2024-07-01").all()
    with pytest.raises(ValueError, match="empty"):
        make_split(demo_df, CREDIT_DEMO_TARGET, date_col=CREDIT_DEMO_DATE, oot_start="2030-01-01")
    with pytest.raises(ValueError, match="needs DATE_COL"):
        make_split(demo_df, CREDIT_DEMO_TARGET, oot_start="2024-01-01")


def test_summary_and_warnings(demo_df, credit_risk_like):
    s = make_split(credit_risk_like, "loan_status")
    w = split_warnings(split_summary(credit_risk_like, "loan_status", s))
    assert any("No OOT" in x for x in w)

    s = make_split(demo_df, CREDIT_DEMO_TARGET, date_col=CREDIT_DEMO_DATE)
    summ = split_summary(demo_df, CREDIT_DEMO_TARGET, s, CREDIT_DEMO_DATE)
    assert list(summ["sample"]) == ["train", "test", "oot"]
    assert summ["rows"].sum() == len(demo_df) and "date_from" in summ

    tiny = pd.DataFrame({"y": [1] * 10 + [0] * 90})
    w = split_warnings(split_summary(tiny, "y", make_split(tiny, "y")))
    assert any("only" in x for x in w)
