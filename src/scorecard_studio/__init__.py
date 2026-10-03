"""
scorecard_studio - optimal binning, WOE and scorecard engine for
binary-target models, designed to run in Google Colab.
"""

__version__ = "0.1.0"

from .artifacts import ScoringArtifact, ScoringBundle  # noqa: E402
from .binning import (  # noqa: E402
    MISSING_GROUP,
    SPECIAL_GROUP,
    BinningEngine,
    BinningResult,
    BinStats,
)
from .config import ConfigError, StudioConfig  # noqa: E402
from .datasets import load_credit_risk_dataset, make_credit_application_data  # noqa: E402
from .dtypes import detect_dtype, infer_variable_types  # noqa: E402
from .metrics import interpret_iv  # noqa: E402
from .presets import PRESETS, get_preset  # noqa: E402
from .intake import IntakeResult, load_table, run_intake  # noqa: E402
from .model import LogisticModel, ModelError, fit_logistic  # noqa: E402
from .scaling import ScalingParams, check_coefficient_signs, scorecard_table  # noqa: E402
from .scorecard import (  # noqa: E402
    Scorecard,
    apply_bins,
    build_scorecard,
    load_scorecard,
    performance,
    score_bands,
)
from .screening import ScreeningResult, screen_all_targets, screen_variables, target_sample  # noqa: E402
from .split import make_split, split_summary, split_warnings  # noqa: E402
from .store import RunStore  # noqa: E402

__all__ = [
    "BinningEngine", "BinningResult", "BinStats", "MISSING_GROUP", "SPECIAL_GROUP",
    "ScoringArtifact", "ScoringBundle",
    "ScalingParams", "scorecard_table", "check_coefficient_signs",
    "detect_dtype", "infer_variable_types", "interpret_iv",
    "PRESETS", "get_preset", "make_credit_application_data", "load_credit_risk_dataset",
    "StudioConfig", "ConfigError", "IntakeResult", "load_table", "run_intake",
    "make_split", "split_summary", "split_warnings",
    "ScreeningResult", "screen_variables", "screen_all_targets", "target_sample", "RunStore",
    "fit_logistic", "LogisticModel", "ModelError",
    "Scorecard", "build_scorecard", "load_scorecard", "apply_bins", "performance", "score_bands",
]
