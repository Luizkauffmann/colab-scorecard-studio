"""
Stage 1 - data intake.

Load a modeling table from wherever the user has it, check the config
against it, make the target a clean 0/1 integer, apply plausibility rules,
and describe every column. Nothing here looks at the relationship between
predictors and the target: that happens in screening, on Train only.
"""

from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ConfigError, StudioConfig
from .dtypes import DEFAULT_MAX_CATEGORIES, detect_dtype

DRIVE_ROOT = "/content/drive/MyDrive"

# ═══════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════


def load_table(source: Any, **read_kwargs) -> Tuple[pd.DataFrame, str]:
    """Load a table and say where it came from.

    ``source`` can be:

    * a ``pandas.DataFrame`` (used as is, copied)
    * ``"demo:credit_risk"``: the Kaggle Credit Risk Dataset (default demo)
    * ``"demo:synthetic"``: the synthetic application dataset (has dates)
    * ``"upload"``: Colab file picker
    * ``"drive:<path under My Drive>"``, e.g. ``"drive:data/apps.parquet"``
    * ``"openml:<id>"``
    * ``"kaggle:<owner>/<dataset>"`` or ``"kaggle:<owner>/<dataset>/<file>"``
      (needs a Kaggle API token in Colab secrets or ``~/.kaggle/kaggle.json``)
    * an ``http(s)://`` URL or a local path to CSV, Parquet, Excel or JSON

    ``read_kwargs`` go to the pandas reader (e.g. ``sep=";"``).
    """
    if isinstance(source, pd.DataFrame):
        return source.copy(), "in-memory DataFrame"
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"DATA_SOURCE must be a path, URL or one of the prefixes; got {source!r}")
    s = source.strip()

    if s == "demo:credit_risk":
        from .datasets import load_credit_risk_dataset
        return load_credit_risk_dataset(), "Kaggle Credit Risk Dataset (laotse/credit-risk-dataset)"
    if s == "demo:synthetic":
        from .datasets import make_credit_application_data
        return make_credit_application_data(), "synthetic credit application data"
    if s.startswith("demo:"):
        raise ValueError(f"Unknown demo {s!r}; use 'demo:credit_risk' or 'demo:synthetic'.")
    if s == "upload":
        return _load_upload(**read_kwargs)
    if s.startswith("drive:"):
        path = os.path.join(DRIVE_ROOT, s[len("drive:"):].lstrip("/"))
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Is Drive mounted and the path relative to My Drive?")
        return read_file(path, **read_kwargs), path
    if s.startswith("openml:"):
        from sklearn.datasets import fetch_openml
        frame = fetch_openml(data_id=int(s.split(":", 1)[1]), as_frame=True, parser="auto").frame
        return frame, s
    if s.startswith("kaggle:"):
        return _load_kaggle(s[len("kaggle:"):], **read_kwargs), s
    return read_file(s, **read_kwargs), s


def read_file(path: str, **read_kwargs) -> pd.DataFrame:
    """Read by extension: .csv/.txt/.tsv (optionally .gz/.zip), .parquet/.pq,
    .xlsx/.xls, .json/.jsonl, .feather."""
    p = path.lower()
    for comp in (".gz", ".zip", ".bz2", ".xz"):
        if p.endswith(comp):
            p = p[: -len(comp)]
            break
    if p.endswith((".parquet", ".pq")):
        return pd.read_parquet(path, **read_kwargs)
    if p.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path, **read_kwargs)
    if p.endswith(".jsonl"):
        return pd.read_json(path, lines=True, **read_kwargs)
    if p.endswith(".json"):
        return pd.read_json(path, **read_kwargs)
    if p.endswith(".feather"):
        return pd.read_feather(path, **read_kwargs)
    if p.endswith(".tsv"):
        read_kwargs.setdefault("sep", "\t")
    return pd.read_csv(path, **read_kwargs)


def _load_upload(**read_kwargs) -> Tuple[pd.DataFrame, str]:
    try:
        from google.colab import files  # type: ignore
    except ImportError as exc:
        raise RuntimeError("DATA_SOURCE='upload' only works in Google Colab.") from exc
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No file uploaded.")
    name = next(iter(uploaded))
    tmp = os.path.join(tempfile.mkdtemp(), name)
    with open(tmp, "wb") as fh:
        fh.write(uploaded[name])
    return read_file(tmp, **read_kwargs), f"upload:{name}"


def _load_kaggle(ref: str, **read_kwargs) -> pd.DataFrame:
    parts = ref.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Use 'kaggle:<owner>/<dataset>' or 'kaggle:<owner>/<dataset>/<file>'.")
    dataset, file_name = "/".join(parts[:2]), "/".join(parts[2:]) or None
    _kaggle_credentials_from_colab_secrets()
    out = tempfile.mkdtemp()
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", out, "--unzip"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "Kaggle download failed. Add KAGGLE_USERNAME and KAGGLE_KEY in Colab's Secrets panel "
            f"(key icon) or upload kaggle.json.\n{res.stderr.strip() or res.stdout.strip()}")
    if file_name:
        path = os.path.join(out, file_name)
    else:
        tables = [f for f in glob.glob(os.path.join(out, "**", "*"), recursive=True)
                  if f.lower().endswith((".csv", ".parquet", ".xlsx", ".json"))]
        if len(tables) != 1:
            names = [os.path.relpath(f, out) for f in tables]
            raise ValueError(f"Dataset has {len(tables)} tables {names}; name one: kaggle:{dataset}/<file>")
        path = tables[0]
    return read_file(path, **read_kwargs)


def _kaggle_credentials_from_colab_secrets():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    try:
        from google.colab import userdata  # type: ignore
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    except Exception:  # noqa: BLE001 - fall back to ~/.kaggle/kaggle.json
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Checks and preparation
# ═══════════════════════════════════════════════════════════════════════════


def prepare_target(df: pd.DataFrame, target: str, event_value: Any = None
                   ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return ``df`` with ``target`` as int 0/1 and a report.

    Rows with a missing target have no observed outcome (indeterminate,
    still in the performance window...). They are dropped and counted.
    If the target isn't already 0/1, set ``event_value`` to the value
    that means *event* (e.g. ``"Charged Off"``); everything else becomes 0.
    """
    if target not in df.columns:
        raise ConfigError([f"TARGET column {target!r} not found."])
    y = df[target]
    n_missing = int(y.isna().sum())
    df = df.loc[y.notna()].copy()
    y = df[target]

    if event_value is not None:
        if not (y == event_value).any():
            raise ConfigError([f"EVENT_VALUE {event_value!r} never occurs in {target!r}; "
                               f"values are {sorted(map(str, y.unique()))[:10]}."])
        y01 = (y == event_value).astype("int64")
    else:
        y01 = _as_binary(y)
        if y01 is None:
            vals = sorted(map(str, pd.unique(y)))[:10]
            raise ConfigError([
                f"TARGET {target!r} must be 0/1 (1 = event), found values {vals}. "
                "Set EVENT_VALUE to the value that means event."])
    if y01.nunique() < 2:
        raise ConfigError([f"TARGET {target!r} has a single class; nothing to model."])
    df[target] = y01.to_numpy()
    report = {"rows_dropped_missing_target": n_missing, "rows": int(len(df)),
              "events": int(y01.sum()), "event_rate": float(y01.mean())}
    return df, report


def _as_binary(y: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_bool_dtype(y):
        return y.astype("int64")
    num = pd.to_numeric(y, errors="coerce")
    if num.isna().any():
        mapped = y.astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "yes": 1, "no": 0, "y": 1, "n": 0})
        if mapped.isna().any():
            return None
        num = mapped
    if not set(pd.unique(num)) <= {0, 1}:
        return None
    return num.astype("int64")


def apply_plausibility(df: pd.DataFrame, rules: Dict[str, Tuple[Optional[float], Optional[float]]],
                       action: str = "special", code: float = -99999
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[Any]]]:
    """Find values outside ``[min, max]`` and treat them per ``action``.

    Returns ``(df, report, new_special_codes)``. ``new_special_codes`` maps
    each column that got the code to ``[code]``; merge it into the special
    codes passed to binning. The same rule must run before scoring.
    """
    df = df.copy()
    rows, new_specials = [], {}
    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            raise ConfigError([f"PLAUSIBILITY: column {col!r} not found."])
        x = pd.to_numeric(df[col], errors="coerce")
        below = (x < lo) if lo is not None else pd.Series(False, index=df.index)
        above = (x > hi) if hi is not None else pd.Series(False, index=df.index)
        bad = (below | above).fillna(False)
        n_bad = int(bad.sum())
        if n_bad and action == "special":
            if (x == code).any():
                raise ConfigError([f"IMPLAUSIBLE_CODE {code} already occurs in {col!r}; choose another."])
            df.loc[bad, col] = code
            new_specials[col] = [code]
        elif n_bad and action == "missing":
            df[col] = df[col].astype("float64")
            df.loc[bad, col] = np.nan
        rows.append({
            "column": col, "min": lo, "max": hi,
            "n_below": int(below.fillna(False).sum()), "n_above": int(above.fillna(False).sum()),
            "examples": sorted(set(x[bad].tolist()))[:5],
            "action": action if n_bad else "none needed",
            "replaced_by": (code if action == "special" else "missing" if action == "missing" else None)
                           if n_bad else None,
        })
    return df, pd.DataFrame(rows), new_specials


def add_row_id(df: pd.DataFrame, id_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """Use ``id_col`` if given (must be unique), else add ``row_id``."""
    if id_col is not None:
        dup = int(df[id_col].duplicated().sum())
        if dup:
            raise ConfigError([f"ID_COL {id_col!r} has {dup} duplicated values; it must be unique."])
        return df, id_col
    name = "row_id"
    while name in df.columns:
        name = "_" + name
    df = df.copy()
    df.insert(0, name, np.arange(len(df), dtype="int64"))
    return df, name


def data_dictionary(df: pd.DataFrame, cfg: StudioConfig, id_col: str,
                    max_categories: int = DEFAULT_MAX_CATEGORIES) -> pd.DataFrame:
    """One row per column: role, types, missing rate, cardinality, examples."""
    rows = []
    n = max(len(df), 1)
    for c in df.columns:
        s = df[c]
        if c == cfg.target:
            role = "target"
        elif c == id_col:
            role = "id"
        elif c == cfg.date_col:
            role = "date"
        elif c in cfg.exclude:
            role = "excluded by user"
        else:
            role = "candidate"
        top = s.value_counts(dropna=True)
        rows.append({
            "column": c, "role": role, "pandas_dtype": str(s.dtype),
            "binning_type": detect_dtype(s, max_categories) if role == "candidate" or role == "excluded by user" else "",
            "missing_rate": round(float(s.isna().mean()), 4),
            "n_unique": int(s.nunique(dropna=True)),
            "top_value": None if top.empty else top.index[0],
            "top_share": 0.0 if top.empty else round(float(top.iloc[0] / n), 4),
            "examples": [v for v in s.dropna().unique()[:5].tolist()],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class IntakeResult:
    df: pd.DataFrame
    id_col: str
    source: str
    special_codes: Dict[str, List[Any]]
    dictionary: pd.DataFrame
    plausibility: pd.DataFrame
    report: Dict[str, Any] = field(default_factory=dict)


def run_intake(cfg: StudioConfig, df: Optional[pd.DataFrame] = None) -> IntakeResult:
    """Load (unless ``df`` is given), validate the config against the columns,
    prepare the target, apply plausibility rules and describe the data."""
    cfg.validate()
    if df is None:
        df, source = load_table(cfg.data_source)
    else:
        df, source = df.copy(), "in-memory DataFrame"
    df.columns = [str(c) for c in df.columns]
    n_raw = len(df)
    cfg.validate(df.columns)

    df, target_report = prepare_target(df, cfg.target, cfg.event_value)
    df, plaus, new_specials = apply_plausibility(df, cfg.plausibility, cfg.implausible_action,
                                                 cfg.implausible_code)
    special_codes = {k: list(v) for k, v in cfg.special_codes.items()}
    for col, codes in new_specials.items():
        special_codes[col] = list(dict.fromkeys([*special_codes.get(col, []), *codes]))

    df, id_col = add_row_id(df, cfg.id_col)
    if cfg.date_col is not None:
        parsed = pd.to_datetime(df[cfg.date_col], errors="coerce")
        bad = int(parsed.isna().sum())
        if bad:
            raise ConfigError([f"DATE_COL {cfg.date_col!r} has {bad} values that are missing "
                               "or not parseable as dates."])
        df[cfg.date_col] = parsed

    feature_cols = [c for c in df.columns if c != id_col]
    n_dup = int(df.duplicated(subset=feature_cols).sum())
    report = {
        "source": source, "rows_raw": n_raw, **target_report,
        "columns": int(df.shape[1]),
        "exact_duplicate_rows": n_dup,
        "implausible_values": int((plaus["n_below"] + plaus["n_above"]).sum()) if len(plaus) else 0,
    }
    return IntakeResult(df=df, id_col=id_col, source=source, special_codes=special_codes,
                        dictionary=data_dictionary(df, cfg, id_col), plausibility=plaus,
                        report=report)
