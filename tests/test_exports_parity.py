"""Engine transform vs generated scorer.py vs generated SQL must agree exactly
(issue #6 for SQL quoting/escaping)."""

import importlib.util
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import BinningEngine, ScoringBundle

GROUP_WOE = ("group", "woe")


def _load_module(path, name="generated_scorer"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sqlite_transform(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    con = sqlite3.connect(":memory:")
    data = df.copy()
    for c in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[c]):
            data[c] = data[c].astype(str)
        elif pd.api.types.is_bool_dtype(data[c]):
            data[c] = data[c].astype(int)
    data.insert(0, "__row", np.arange(len(data)))
    data.to_sql("input_table", con, index=False)
    out = pd.read_sql_query(sql + "\nORDER BY __row", con)
    con.close()
    return out


def _edge_rows(engine: BinningEngine, df: pd.DataFrame) -> pd.DataFrame:
    """Rows that stress the rules: exact cutoffs, extremes, missing, specials, unseen."""
    rows = []
    base = df.iloc[0].to_dict()
    for v, r in engine.results.items():
        if r.dtype == "numerical":
            for c in r.cutoffs:
                rows.append({**base, v: c})
            rows.append({**base, v: -1e12})
            rows.append({**base, v: 1e12})
            for s in r.special_codes:
                rows.append({**base, v: s})
        else:
            rows.append({**base, v: "never_seen" if r.value_type == "string" else 99})
        rows.append({**base, v: None})
    edge = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == bool:
            edge[c] = edge[c].astype(object)
        elif pd.api.types.is_integer_dtype(df[c]) and (
                edge[c].isna().any() or (edge[c].dropna() % 1 != 0).any()):
            edge[c] = edge[c].astype("float64")  # fractional cutoffs / NaN in an int column
        else:
            edge[c] = edge[c].astype(df[c].dtype)
    return edge


@pytest.fixture(scope="module")
def scoring_input(fitted_engine, demo_df):
    edge = _edge_rows(fitted_engine, demo_df)
    edge["HAS_COSIGNER"] = edge["HAS_COSIGNER"].map(
        lambda v: v if isinstance(v, (bool, np.bool_)) else False).astype(bool)
    return pd.concat([demo_df.head(1500), edge], ignore_index=True)


def test_python_scorer_matches_engine(fitted_engine, scoring_input, tmp_path):
    bundle = fitted_engine.build_scoring_bundle(name="parity")
    path = bundle.save_python(tmp_path / "scorer.py")
    scorer = _load_module(path)
    expected = fitted_engine.transform(scoring_input, metrics=GROUP_WOE)
    got = scorer.score_dataframe(scoring_input, metrics=GROUP_WOE)
    for v in fitted_engine.results:
        for m in GROUP_WOE:
            col = f"opt_{v}_{m}"
            np.testing.assert_array_equal(got[col].to_numpy(), expected[col].to_numpy(), err_msg=col)
    rec = scoring_input.iloc[-1].to_dict()
    assert scorer.score_record(rec) == bundle.score_record(rec)


def test_sql_matches_engine(fitted_engine, scoring_input):
    bundle = fitted_engine.build_scoring_bundle(name="parity")
    expected = fitted_engine.transform(scoring_input, metrics=GROUP_WOE)
    got = _sqlite_transform(scoring_input, bundle.to_sql(source_table="input_table"))
    for v in fitted_engine.results:
        for m in GROUP_WOE:
            col = f"opt_{v}_{m}"
            np.testing.assert_array_equal(got[col].to_numpy(), expected[col].to_numpy(), err_msg=col)


def test_bundle_json_round_trip_scores_identically(fitted_engine, scoring_input):
    bundle = fitted_engine.build_scoring_bundle()
    text = bundle.to_json()
    json.loads(text)  # strict JSON: no NaN / Infinity
    reloaded = ScoringBundle.from_json(text)
    a = bundle.score_dataframe(scoring_input)
    b = reloaded.score_dataframe(scoring_input)
    pd.testing.assert_frame_equal(a, b)


def test_score_record_matches_vectorised(fitted_engine, scoring_input):
    bundle = fitted_engine.build_scoring_bundle()
    vec = bundle.score_dataframe(scoring_input.tail(40), metrics=("group", "woe", "label"))
    for i, row in scoring_input.tail(40).iterrows():
        rec = bundle.score_record(row.to_dict())
        for k, v in rec.items():
            assert vec.at[i, k] == v, (i, k)


def test_score_dataframe_requires_all_columns(fitted_engine, demo_df):
    with pytest.raises(KeyError):
        fitted_engine.build_scoring_bundle().score_dataframe(demo_df.drop(columns=["AGE"]))


# ── issue #6: awkward names and values ────────────────────────────────────

@pytest.fixture(scope="module")
def awkward_engine():
    rng = np.random.default_rng(11)
    n = 4000
    cats = np.array(["O'Brien", 'say "hi"', "back\\slash", "plain", "semi;colon -- x"])
    c = rng.choice(cats, n)
    x = rng.normal(size=n)
    risk = 0.08 + 0.05 * (c == "O'Brien") + 0.04 * (x > 0.5)
    df = pd.DataFrame({
        "weird col": c,
        'quote"col': x,
        "select": rng.choice(["a", "b"], n),     # reserved word as column name
        "a-b": rng.normal(size=n),
        "a_b": rng.normal(size=n),
        "y": (rng.random(n) < risk).astype(int),
    })
    eng = BinningEngine(df, "y")
    eng.fit_all(max_bins=4)
    assert not eng.fit_errors, eng.fit_errors
    return eng, df


def test_sql_escapes_identifiers_and_literals(awkward_engine):
    eng, df = awkward_engine
    bundle = eng.build_scoring_bundle()
    sql = bundle.to_sql()
    assert '"weird col"' in sql and '"quote""col"' in sql and '"select"' in sql
    assert "'O''Brien'" in sql
    got = _sqlite_transform(df, sql)
    exp = eng.transform(df, metrics=GROUP_WOE)
    for v in eng.results:
        for m in GROUP_WOE:
            col = f"opt_{v}_{m}"
            np.testing.assert_array_equal(got[col].to_numpy(), exp[col].to_numpy(), err_msg=col)


def test_spark_and_bigquery_quoting(awkward_engine):
    eng, _ = awkward_engine
    bundle = eng.build_scoring_bundle()
    spark = bundle.to_sql(dialect="spark")
    bq = bundle.to_sql(dialect="bigquery")
    assert "`weird col`" in spark and "'O\\'Brien'" in spark
    assert "`weird col`" in bq and "'back\\\\slash'" in bq
    with pytest.raises(ValueError):
        bundle.to_sql(dialect="oracle")


def test_python_scorer_handles_awkward_names(awkward_engine, tmp_path):
    eng, df = awkward_engine
    bundle = eng.build_scoring_bundle()
    src = bundle.to_python_module()
    assert "def score_a_b(" in src and "def score_a_b_2(" in src  # de-duplicated
    scorer = _load_module(bundle.save_python(tmp_path / "scorer.py"), "awkward_scorer")
    got = scorer.score_dataframe(df, metrics=GROUP_WOE)
    exp = eng.transform(df, metrics=GROUP_WOE)
    for v in eng.results:
        for m in GROUP_WOE:
            col = f"opt_{v}_{m}"
            np.testing.assert_array_equal(got[col].to_numpy(), exp[col].to_numpy(), err_msg=col)
