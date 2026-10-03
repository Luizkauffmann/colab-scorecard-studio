"""Binning config round trip (what the app will save to Google Drive)."""

import json

import pytest

from scorecard_studio import BinningEngine, get_preset, PRESETS
from scorecard_studio.datasets import CREDIT_DEMO_SPECIAL_CODES


def test_config_round_trip_rebuilds_identical_bins(fitted_engine, demo_df):
    cfg = json.loads(json.dumps(fitted_engine.export_config()))  # must be JSON-serialisable
    fresh = BinningEngine(demo_df, "DEFAULT_12M", special_codes=CREDIT_DEMO_SPECIAL_CODES)
    assert fresh.import_config(cfg) == []
    for v, r in fitted_engine.results.items():
        r2 = fresh.get_result(v)
        assert r2.cutoffs == r.cutoffs and r2.cat_groups == r.cat_groups, v
        assert [b.woe for b in r2.bins] == [b.woe for b in r.bins], v
        assert r2.settings == r.settings and r2.special_codes == r.special_codes


def test_config_skips_unknown_columns_and_checks_version(fitted_engine, demo_df):
    cfg = fitted_engine.export_config()
    fresh = BinningEngine(demo_df.drop(columns=["AGE"]), "DEFAULT_12M")
    assert fresh.import_config(cfg) == ["AGE"]
    with pytest.raises(ValueError):
        fresh.import_config({**cfg, "schema_version": 99})


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_fit(name, demo_df):
    p = get_preset(name)
    p["fit_params"]["min_bin_n_event"] = None if p["fit_params"]["min_bin_n_event"] is None else 10
    eng = BinningEngine(demo_df, "DEFAULT_12M")
    r = eng.fit("DEBT_TO_INCOME", **p["fit_params"])
    assert r.iv > 0


def test_get_preset_returns_a_copy():
    a = get_preset("fraud")
    a["fit_params"]["max_bins"] = 99
    assert get_preset("fraud")["fit_params"]["max_bins"] != 99
    with pytest.raises(KeyError):
        get_preset("nope")
