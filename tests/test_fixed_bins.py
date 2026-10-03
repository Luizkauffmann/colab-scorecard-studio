"""Missing / Special as a modeling choice: separate (default), merged into a
regular bin, or Special combined with Missing. The choice must score
identically in the engine, scorer.py, SQL and after a config round trip."""

import json

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import MISSING_GROUP, SPECIAL_GROUP, BinningEngine
from scorecard_studio.datasets import CREDIT_DEMO_DATE, CREDIT_DEMO_ID, CREDIT_DEMO_SPECIAL_CODES

from test_exports_parity import GROUP_WOE, _edge_rows, _load_module, _sqlite_transform


@pytest.fixture
def engine(demo_df):
    eng = BinningEngine(demo_df, "DEFAULT_12M", special_codes=CREDIT_DEMO_SPECIAL_CODES,
                        exclude=[CREDIT_DEMO_ID, CREDIT_DEMO_DATE])
    for v in ("MONTHS_SINCE_LAST_DELINQ", "MONTHLY_INCOME", "EDUCATION_LEVEL", "EMPLOYMENT_TYPE"):
        eng.fit(v)
    return eng


def _bin(r, kind=None, group=None):
    return next(b for b in r.bins if (kind is None or b.kind == kind) and (group is None or b.group == group))


def test_merge_missing_into_regular_bin(engine, demo_df):
    v = "MONTHLY_INCOME"
    before = engine.get_result(v)
    n_missing = int(demo_df[v].isna().sum())
    g1_before = _bin(before, "regular", 1)
    r = engine.set_fixed_bins(v, missing_to=1)
    g1 = _bin(r, "regular", 1)
    assert g1.count == g1_before.count + n_missing
    assert g1.label.endswith(" | Missing")
    m = _bin(r, "missing")
    assert m.merged_into == 1 and m.count == 0 and m.group == 1 and m.woe == g1.woe
    assert sum(b.count for b in r.bins) == len(demo_df)          # no row lost or double counted
    out = engine.transform(demo_df[[v]], variables=[v], metrics=("group", "woe"))
    miss = demo_df[v].isna().to_numpy()
    assert (out[f"opt_{v}_group"].to_numpy()[miss] == 1).all()
    assert np.allclose(out[f"opt_{v}_woe"].to_numpy()[miss], g1.woe)


def test_special_with_missing_and_into_group(engine, demo_df):
    v = "MONTHS_SINCE_LAST_DELINQ"
    n_sp = int(demo_df[v].isin([-999, -998]).sum())
    n_mi = int(demo_df[v].isna().sum())
    r = engine.set_fixed_bins(v, special_to=MISSING_GROUP)
    m = _bin(r, "missing")
    assert m.count == n_sp + n_mi and m.label == "Missing | Special"
    s = _bin(r, "special")
    assert s.merged_into == MISSING_GROUP and s.group == MISSING_GROUP and s.woe == m.woe

    r = engine.set_fixed_bins(v, special_to=2)
    assert _bin(r, "regular", 2).label.endswith(" | Special")
    assert _bin(r, "special").group == 2

    # Special with Missing, Missing merged into 3 -> both land in 3
    r = engine.set_fixed_bins(v, missing_to=3, special_to=MISSING_GROUP)
    g3 = _bin(r, "regular", 3)
    assert g3.label.endswith(" | Missing | Special")
    assert _bin(r, "special").group == 3 and _bin(r, "missing").group == 3
    assert sum(b.count for b in r.bins) == len(demo_df)

    r = engine.set_fixed_bins(v)                                # back to separate
    assert _bin(r, "special").group == SPECIAL_GROUP and _bin(r, "missing").group == MISSING_GROUP


def test_categorical_merge_and_validation(engine):
    v = "EMPLOYMENT_TYPE"
    r = engine.set_fixed_bins(v, missing_to=2)
    assert _bin(r, "missing").group == 2
    with pytest.raises(ValueError, match="regular groups"):
        engine.set_fixed_bins(v, missing_to=99)
    with pytest.raises(ValueError, match="no special codes"):
        engine.set_fixed_bins(v, special_to=1)


def test_edits_keep_valid_choice_and_drop_invalid(engine):
    v = "MONTHLY_INCOME"
    engine.set_fixed_bins(v, missing_to=2)
    r = engine.adjust_cutoffs(v, [3000, 6000])
    assert r.missing_to == 2                                   # still a valid group
    r = engine.adjust_cutoffs(v, [3000])                       # only groups 1..2 now
    assert r.missing_to == 2
    r = engine.adjust_cutoffs(v, [])                           # single group
    assert r.missing_to is None
    engine.adjust_cutoffs(v, [3000, 6000])
    engine.set_fixed_bins(v, missing_to=1)
    r = engine.fit(v)
    assert r.missing_to == 1                                   # refit keeps it


def test_config_round_trip_keeps_choice(engine, demo_df):
    engine.set_fixed_bins("MONTHLY_INCOME", missing_to=2)
    engine.set_fixed_bins("MONTHS_SINCE_LAST_DELINQ", missing_to=1, special_to=MISSING_GROUP)
    cfg = json.loads(json.dumps(engine.export_config()))
    fresh = BinningEngine(demo_df, "DEFAULT_12M", special_codes=CREDIT_DEMO_SPECIAL_CODES)
    fresh.import_config(cfg)
    for v in ("MONTHLY_INCOME", "MONTHS_SINCE_LAST_DELINQ"):
        a, b = engine.get_result(v), fresh.get_result(v)
        assert (a.missing_to, a.special_to) == (b.missing_to, b.special_to)
        assert [x.woe for x in a.bins] == [x.woe for x in b.bins]


def test_merged_choice_scores_identically_everywhere(engine, demo_df, tmp_path):
    engine.set_fixed_bins("MONTHLY_INCOME", missing_to=2)
    engine.set_fixed_bins("MONTHS_SINCE_LAST_DELINQ", missing_to=1, special_to=MISSING_GROUP)
    engine.set_fixed_bins("EDUCATION_LEVEL", missing_to=3)
    engine.set_fixed_bins("EMPLOYMENT_TYPE", missing_to=1)       # unseen categories follow Missing
    data = pd.concat([demo_df.head(800), _edge_rows(engine, demo_df)], ignore_index=True)
    data["HAS_COSIGNER"] = data["HAS_COSIGNER"].map(
        lambda v: v if isinstance(v, (bool, np.bool_)) else False).astype(bool)
    bundle = engine.build_scoring_bundle(name="merged")
    expected = engine.transform(data, metrics=GROUP_WOE)
    scorer = _load_module(bundle.save_python(tmp_path / "scorer.py"), "merged_scorer")
    got_py = scorer.score_dataframe(data, metrics=GROUP_WOE)
    got_sql = _sqlite_transform(data, bundle.to_sql(source_table="input_table"))
    for v in engine.results:
        for m in GROUP_WOE:
            col = f"opt_{v}_{m}"
            np.testing.assert_array_equal(got_py[col].to_numpy(), expected[col].to_numpy(), err_msg=col)
            np.testing.assert_array_equal(got_sql[col].to_numpy(), expected[col].to_numpy(), err_msg=col)
    # missing values really land in the chosen group
    miss = data["MONTHLY_INCOME"].isna().to_numpy()
    assert (expected["opt_MONTHLY_INCOME_group"].to_numpy()[miss] == 2).all()
