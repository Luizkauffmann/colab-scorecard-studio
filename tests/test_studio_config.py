"""The config cell: every problem reported at once, typos caught with a suggestion."""

import json

import pytest

from scorecard_studio import ConfigError, StudioConfig

COLS = ["loan_status", "person_age", "loan_grade", "loan_int_rate", "app_date"]


def test_typo_in_exclude_is_an_error_with_suggestion():
    cfg = StudioConfig(target="loan_status", exclude=["loan_grad"])
    with pytest.raises(ConfigError, match="did you mean 'loan_grade'"):
        cfg.validate(COLS)


def test_case_mismatch_gets_its_own_hint():
    cfg = StudioConfig(target="LOAN_STATUS")
    with pytest.raises(ConfigError, match="case-sensitive"):
        cfg.validate(COLS)


def test_all_problems_reported_together():
    cfg = StudioConfig(target="loan_status", exclude=["nope", "loan_grade"],
                       force_include=["loan_grade", "also_nope"], iv_min=0.6, iv_max=0.5,
                       implausible_action="cap", plausibility={"person_age": (100, 18)})
    with pytest.raises(ConfigError) as err:
        cfg.validate(COLS)
    text = str(err.value)
    for piece in ["'nope'", "'also_nope'", "IV_MIN", "EXCLUDE and FORCE_INCLUDE",
                  "IMPLAUSIBLE_ACTION", "min > max"]:
        assert piece in text, piece
    assert len(err.value.problems) == 6


def test_roles_cannot_be_forced():
    cfg = StudioConfig(target="loan_status", date_col="app_date", force_include=["app_date"])
    with pytest.raises(ConfigError, match="DATE_COL"):
        cfg.validate(COLS)


def test_iv_thresholds_come_from_preset_unless_set():
    assert StudioConfig(preset="credit_pd").resolved_iv_min == 0.10
    assert StudioConfig(preset="fraud").resolved_iv_min == 0.02
    cfg = StudioConfig(preset="credit_pd", iv_min=0.05, iv_max=0.8)
    assert (cfg.resolved_iv_min, cfg.resolved_iv_max) == (0.05, 0.8)
    with pytest.raises(ConfigError, match="PRESET"):
        StudioConfig(preset="mortgage").validate()


def test_json_round_trip():
    cfg = StudioConfig(target="loan_status", exclude=["loan_grade"], special_codes={"x": [-1]},
                       plausibility={"person_age": (18, 100)})
    d = json.loads(cfg.to_json())
    assert d["resolved"]["iv_min"] == 0.10 and d["plausibility"] == {"person_age": [18, 100]}
    back = StudioConfig.from_json(cfg.to_json())
    assert back.exclude == ["loan_grade"] and back.plausibility == {"person_age": (18, 100)}
