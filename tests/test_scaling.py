"""Issue #4: scorecard points use beta * WOE plus offset/intercept allocation."""

import math

import numpy as np
import pytest

from scorecard_studio import ScalingParams, check_coefficient_signs, scorecard_table


def test_scaling_anchor_points():
    p = ScalingParams(pdo=20, base_score=600, base_odds=50)
    assert p.score_from_pd(1 / 51) == pytest.approx(600)          # odds 50:1 -> base score
    assert p.score_from_pd(1 / 101) == pytest.approx(620)         # odds doubled -> +PDO
    assert p.pd_from_score(p.score_from_pd(0.07)) == pytest.approx(0.07)
    with pytest.raises(ValueError):
        ScalingParams(pdo=0)


def test_points_sum_to_score(fitted_engine, demo_df):
    bundle = fitted_engine.build_scoring_bundle()
    coefs = {"CREDIT_UTILIZATION": 0.9, "opt_MONTHLY_INCOME_woe": 0.8, "EMPLOYMENT_TYPE": 1.1,
             "MONTHS_SINCE_LAST_DELINQ": 0.7}
    b0 = -2.4
    params = ScalingParams(pdo=20, base_score=600, base_odds=50)
    table = scorecard_table(bundle, coefs, b0, params)
    assert set(table["Variable"]) == {"CREDIT_UTILIZATION", "MONTHLY_INCOME",
                                      "EMPLOYMENT_TYPE", "MONTHS_SINCE_LAST_DELINQ"}
    # every bin incl. Special/Missing gets points
    assert {"special", "missing"} <= set(table["Kind"])

    sample = demo_df.head(500)
    woe = fitted_engine.transform(sample, metrics=("group", "woe"))
    pts = {(r.Variable, r.Group): r.Points for r in table.itertuples()}
    betas = {"CREDIT_UTILIZATION": 0.9, "MONTHLY_INCOME": 0.8, "EMPLOYMENT_TYPE": 1.1,
             "MONTHS_SINCE_LAST_DELINQ": 0.7}
    for i in sample.index:
        total = sum(pts[(v, woe.at[i, f"opt_{v}_group"])] for v in betas)
        lp = b0 + sum(b * woe.at[i, f"opt_{v}_woe"] for v, b in betas.items())
        assert total == pytest.approx(params.score_from_logit(lp), abs=1e-9)


def test_riskier_bins_get_fewer_points(fitted_engine):
    bundle = fitted_engine.build_scoring_bundle()
    t = scorecard_table(bundle, {"CREDIT_UTILIZATION": 1.0}, -2.5)
    reg = t[t["Kind"] == "regular"].sort_values("WOE")
    assert reg["Points"].is_monotonic_decreasing


def test_bundle_method_requires_coefficients(fitted_engine):
    bundle = fitted_engine.build_scoring_bundle()
    with pytest.raises(TypeError):
        bundle.to_scorecard_table()  # the old WOE-only table is gone on purpose
    t = bundle.to_scorecard_table({"AGE": 1.0}, intercept=-2.5, round_points=0)
    assert (t["Points"] == t["Points"].round()).all()
    with pytest.raises(KeyError):
        bundle.to_scorecard_table({"NOT_A_VAR": 1.0}, intercept=0)


def test_check_coefficient_signs():
    assert check_coefficient_signs({"a": 0.8, "b": -0.2, "c": 0.0}) == ["b", "c"]
