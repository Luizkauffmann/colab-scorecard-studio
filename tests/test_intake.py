"""Stage 1: loading, target preparation, plausibility rules, row IDs."""

import numpy as np
import pandas as pd
import pytest

from scorecard_studio import ConfigError, StudioConfig, load_table, run_intake
from scorecard_studio.datasets import (
    CREDIT_RISK_COLUMNS,
    CREDIT_RISK_EXCLUDE,
    CREDIT_RISK_PLAUSIBILITY,
    _normalise_credit_risk,
    _read_packaged,
    load_credit_risk_dataset,
)
from scorecard_studio.intake import add_row_id, apply_plausibility, prepare_target, read_file


# ── loading ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,kwargs", [("t.csv", {}), ("t.csv.gz", {}), ("t.parquet", {}),
                                         ("t.tsv", {}), ("t.jsonl", {})])
def test_read_file_by_extension(tmp_path, name, kwargs):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]})
    p = tmp_path / name
    if name.endswith(".parquet"):
        df.to_parquet(p)
    elif name.endswith(".tsv"):
        df.to_csv(p, sep="\t", index=False)
    elif name.endswith(".jsonl"):
        df.to_json(p, orient="records", lines=True)
    else:
        df.to_csv(p, index=False)
    out = read_file(str(p), **kwargs)
    assert out["a"].tolist() == [1, 2, 3] and out["b"].isna().sum() == 1


def test_load_table_sources(tmp_path):
    df = pd.DataFrame({"a": [1]})
    out, src = load_table(df)
    assert out is not df and src == "in-memory DataFrame"
    syn, _ = load_table("demo:synthetic")
    assert "DEFAULT_12M" in syn.columns
    with pytest.raises(ValueError, match="Unknown demo"):
        load_table("demo:mortgage")
    with pytest.raises(RuntimeError, match="only works in Google Colab"):
        load_table("upload")
    with pytest.raises(FileNotFoundError):
        load_table("drive:nope/file.csv")


def test_credit_risk_normalisation_matches_csv_and_openml_shapes(credit_risk_like):
    # OpenML ARFF gives categories and floats; the CSV gives strings and ints.
    arff = credit_risk_like.copy()
    for c in ("person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"):
        arff[c] = arff[c].astype("category")
    for c in ("person_age", "person_income", "loan_amnt", "loan_status"):
        arff[c] = arff[c].astype(float)
    a = _normalise_credit_risk(credit_risk_like[CREDIT_RISK_COLUMNS[::-1]])
    b = _normalise_credit_risk(arff)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == CREDIT_RISK_COLUMNS
    assert a["loan_grade"].dtype == object and a["loan_status"].dtype == "int64"
    with pytest.raises(ValueError, match="missing"):
        _normalise_credit_risk(credit_risk_like.drop(columns="loan_grade"))


@pytest.mark.skipif(_read_packaged() is not None, reason="dataset is packaged in this install")
def test_packaged_source_raises_when_file_absent():
    with pytest.raises(FileNotFoundError):
        load_credit_risk_dataset("packaged")


@pytest.mark.skipif(_read_packaged() is None, reason="dataset not packaged in this install")
def test_packaged_dataset_is_the_kaggle_file():
    import gzip
    import hashlib
    from importlib import resources
    raw = gzip.decompress(resources.files("scorecard_studio").joinpath(
        "data", "credit_risk_dataset.csv.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == \
        "ce3c6d2167717bf1627d1c0c81cbccd28323cd4aa7b96d542599366d5ff6aac8"
    df = load_credit_risk_dataset("packaged")
    assert df.shape == (32_581, 12) and df["loan_status"].sum() == 7108
    assert df["person_emp_length"].isna().sum() == 895 and df["loan_int_rate"].isna().sum() == 3116


# ── target ───────────────────────────────────────────────────────────────

def test_prepare_target_drops_unknown_outcomes_and_reports():
    df = pd.DataFrame({"y": [0, 1, np.nan, 1, 0], "x": range(5)})
    out, rep = prepare_target(df, "y")
    assert len(out) == 4 and out["y"].dtype == "int64"
    assert rep["rows_dropped_missing_target"] == 1 and rep["events"] == 2


@pytest.mark.parametrize("values,event_value,expected", [
    (["yes", "no", "Yes"], None, [1, 0, 1]),
    ([True, False, True], None, [1, 0, 1]),
    (["0", "1", "1"], None, [0, 1, 1]),
    (["Charged Off", "Fully Paid", "Current"], "Charged Off", [1, 0, 0]),
])
def test_prepare_target_maps_to_01(values, event_value, expected):
    out, _ = prepare_target(pd.DataFrame({"y": values}), "y", event_value)
    assert out["y"].tolist() == expected


def test_prepare_target_errors():
    with pytest.raises(ConfigError, match="EVENT_VALUE"):
        prepare_target(pd.DataFrame({"y": ["a", "b", "c"]}), "y")
    with pytest.raises(ConfigError, match="never occurs"):
        prepare_target(pd.DataFrame({"y": ["a", "b"]}), "y", event_value="z")
    with pytest.raises(ConfigError, match="single class"):
        prepare_target(pd.DataFrame({"y": [1, 1, np.nan]}), "y")


# ── plausibility ─────────────────────────────────────────────────────────

def test_plausibility_special_routes_values_to_a_code(credit_risk_like):
    out, rep, specials = apply_plausibility(credit_risk_like, CREDIT_RISK_PLAUSIBILITY, "special", -99999)
    assert (out["person_age"] == -99999).sum() == 3
    assert (out["person_emp_length"] == -99999).sum() == 2
    assert out["person_emp_length"].isna().sum() == credit_risk_like["person_emp_length"].isna().sum()
    assert specials == {"person_age": [-99999], "person_emp_length": [-99999]}
    r = rep.set_index("column")
    assert r.at["person_age", "n_above"] == 3 and r.at["person_age", "examples"] == [123, 144]


def test_plausibility_missing_and_flag(credit_risk_like):
    out, _, specials = apply_plausibility(credit_risk_like, {"person_age": (18, 100)}, "missing")
    assert out["person_age"].isna().sum() == 3 and specials == {}
    out, rep, _ = apply_plausibility(credit_risk_like, {"person_age": (18, 100)}, "flag")
    assert out["person_age"].max() == 144 and rep["n_above"].iloc[0] == 3


def test_plausibility_code_must_not_collide():
    df = pd.DataFrame({"a": [-99999, 5, 500]})
    with pytest.raises(ConfigError, match="already occurs"):
        apply_plausibility(df, {"a": (0, 100)}, "special", -99999)


# ── IDs and orchestration ────────────────────────────────────────────────

def test_row_id():
    df = pd.DataFrame({"row_id": [5, 6], "k": [1, 1]})
    out, name = add_row_id(df, None)
    assert name == "_row_id" and out[name].tolist() == [0, 1]
    assert add_row_id(df, "row_id")[1] == "row_id"
    with pytest.raises(ConfigError, match="duplicated"):
        add_row_id(df, "k")


def test_run_intake_on_credit_risk_schema(credit_risk_like):
    df = credit_risk_like.copy()
    df.loc[[0, 1], "loan_status"] = np.nan
    cfg = StudioConfig(target="loan_status", exclude=CREDIT_RISK_EXCLUDE,
                       special_codes={"person_emp_length": [-1]},
                       plausibility=CREDIT_RISK_PLAUSIBILITY)
    res = run_intake(cfg, df=df)
    assert res.id_col == "row_id" and res.df["row_id"].is_unique
    assert res.report["rows_dropped_missing_target"] == 2 and res.report["rows"] == len(df) - 2
    assert res.special_codes["person_emp_length"] == [-1, -99999]
    assert res.special_codes["person_age"] == [-99999]
    roles = res.dictionary.set_index("column")["role"]
    assert roles["loan_grade"] == "excluded by user" and roles["loan_status"] == "target"
    assert res.report["implausible_values"] >= 4


def test_run_intake_stops_on_typo_before_doing_anything(credit_risk_like):
    cfg = StudioConfig(target="loan_status", exclude=["loan_grades"])
    with pytest.raises(ConfigError, match="did you mean 'loan_grade'"):
        run_intake(cfg, df=credit_risk_like)


def test_run_intake_checks_dates(demo_df):
    df = demo_df.copy()
    df["APPLICATION_DATE"] = df["APPLICATION_DATE"].astype(str)
    df.loc[3, "APPLICATION_DATE"] = "not a date"
    cfg = StudioConfig(target="DEFAULT_12M", id_col="APPLICATION_ID", date_col="APPLICATION_DATE")
    with pytest.raises(ConfigError, match="not parseable"):
        run_intake(cfg, df=df)
    df.loc[3, "APPLICATION_DATE"] = "2023-05-01"
    res = run_intake(cfg, df=df)
    assert pd.api.types.is_datetime64_any_dtype(res.df["APPLICATION_DATE"])
