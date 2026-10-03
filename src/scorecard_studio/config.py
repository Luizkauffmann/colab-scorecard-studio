"""
The notebook's single config cell, as a validated object.

Everything a user can change lives here, so the notebook, the web app and the
saved ``config.json`` can never disagree. Widgets are deliberately not used
for these choices: widget state does not survive a Colab runtime reset, a
config cell does.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .presets import PRESETS, get_preset

IMPLAUSIBLE_ACTIONS = ("special", "missing", "flag")


class ConfigError(ValueError):
    """Raised with *all* problems found in the config cell, one per line."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("Fix the config cell:\n  - " + "\n  - ".join(self.problems))


@dataclass
class StudioConfig:
    """User choices for intake, sample design and screening.

    ``iv_min`` / ``iv_max`` left as ``None`` take the preset's defaults.
    ``plausibility`` maps a column to ``(min, max)`` (either may be ``None``);
    values outside the range get ``implausible_action``:

    * ``"special"``: replaced by ``implausible_code`` and added to that
      column's special codes, so they land in the Special bin.
    * ``"missing"``: set to missing, so they land in the Missing bin.
    * ``"flag"``: only reported.
    """

    data_source: Any = "demo:credit_risk"
    target: str = "loan_status"
    id_col: Optional[str] = None
    date_col: Optional[str] = None
    preset: str = "credit_pd"
    event_value: Any = None
    exclude: List[str] = field(default_factory=list)
    force_include: List[str] = field(default_factory=list)
    iv_min: Optional[float] = None
    iv_max: Optional[float] = None
    special_codes: Dict[str, List[Any]] = field(default_factory=dict)
    plausibility: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    implausible_action: str = "special"
    implausible_code: float = -99999
    test_size: float = 0.30
    oot_start: Optional[str] = None
    oot_share: float = 0.20
    corr_threshold: float = 0.70
    seed: int = 42
    output_dir: str = "/content/drive/MyDrive/scorecard_studio"

    # ------------------------------------------------------------------

    def __post_init__(self):
        self.exclude = list(self.exclude or [])
        self.force_include = list(self.force_include or [])
        self.special_codes = {k: list(v) for k, v in (self.special_codes or {}).items()}
        self.plausibility = {k: tuple(v) for k, v in (self.plausibility or {}).items()}

    @property
    def preset_params(self) -> Dict[str, Any]:
        return get_preset(self.preset)

    @property
    def fit_params(self) -> Dict[str, Any]:
        return self.preset_params["fit_params"]

    @property
    def resolved_iv_min(self) -> float:
        return float(self.iv_min if self.iv_min is not None
                     else self.preset_params["screening"]["iv_min"])

    @property
    def resolved_iv_max(self) -> float:
        return float(self.iv_max if self.iv_max is not None
                     else self.preset_params["screening"]["iv_max"])

    # ------------------------------------------------------------------

    def check_settings(self) -> List[str]:
        """Problems that can be found without the data."""
        problems: List[str] = []
        if self.preset not in PRESETS:
            problems.append(f"PRESET {self.preset!r} is unknown; choose one of {sorted(PRESETS)}.")
            return problems
        lo, hi = self.resolved_iv_min, self.resolved_iv_max
        if not 0 <= lo < hi:
            problems.append(f"Need 0 <= IV_MIN < IV_MAX, got IV_MIN={lo}, IV_MAX={hi}.")
        if not 0 < self.test_size < 1:
            problems.append(f"TEST_SIZE must be between 0 and 1, got {self.test_size}.")
        if not 0 < self.oot_share < 1:
            problems.append(f"OOT_SHARE must be between 0 and 1, got {self.oot_share}.")
        if self.implausible_action not in IMPLAUSIBLE_ACTIONS:
            problems.append(f"IMPLAUSIBLE_ACTION must be one of {IMPLAUSIBLE_ACTIONS}, "
                            f"got {self.implausible_action!r}.")
        for col, rng in self.plausibility.items():
            if len(rng) != 2:
                problems.append(f"PLAUSIBILITY[{col!r}] must be (min, max), got {rng!r}.")
            elif rng[0] is not None and rng[1] is not None and rng[0] > rng[1]:
                problems.append(f"PLAUSIBILITY[{col!r}] has min > max: {rng!r}.")
        both = sorted(set(self.exclude) & set(self.force_include))
        if both:
            problems.append(f"In both EXCLUDE and FORCE_INCLUDE: {both}. Pick one.")
        roles = {"TARGET": self.target, "ID_COL": self.id_col, "DATE_COL": self.date_col}
        for name, col in roles.items():
            if col is not None and col in self.force_include:
                problems.append(f"{name} {col!r} cannot be in FORCE_INCLUDE.")
        if self.target in (self.id_col, self.date_col):
            problems.append("TARGET must differ from ID_COL and DATE_COL.")
        return problems

    def check_columns(self, columns: Iterable[str]) -> List[str]:
        """Problems with column names: every name typed in the config must exist."""
        cols = list(columns)
        known = set(cols)
        problems: List[str] = []

        def check(label: str, names: Iterable[Optional[str]]):
            for n in names:
                if n is not None and n not in known:
                    problems.append(f"{label}: column {n!r} not found{_suggest(n, cols)}.")

        check("TARGET", [self.target])
        check("ID_COL", [self.id_col])
        check("DATE_COL", [self.date_col])
        check("EXCLUDE", self.exclude)
        check("FORCE_INCLUDE", self.force_include)
        check("SPECIAL_CODES", self.special_codes)
        check("PLAUSIBILITY", self.plausibility)
        return problems

    def validate(self, columns: Optional[Iterable[str]] = None) -> "StudioConfig":
        """Raise :class:`ConfigError` listing every problem at once."""
        problems = self.check_settings()
        if columns is not None:
            problems += self.check_columns(columns)
        if problems:
            raise ConfigError(problems)
        return self

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not isinstance(self.data_source, (str, type(None))):
            d["data_source"] = f"<{type(self.data_source).__name__}>"
        d["plausibility"] = {k: list(v) for k, v in self.plausibility.items()}
        d["resolved"] = {"iv_min": self.resolved_iv_min, "iv_max": self.resolved_iv_max,
                         "fit_params": self.fit_params}
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StudioConfig":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)

    @classmethod
    def from_json(cls, text: str) -> "StudioConfig":
        return cls.from_dict(json.loads(text))


def _suggest(name: str, columns: Sequence[str]) -> str:
    by_lower = {c.lower(): c for c in columns}
    if name.lower() in by_lower:
        return f" (did you mean {by_lower[name.lower()]!r}? names are case-sensitive)"
    close = difflib.get_close_matches(name, list(columns), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""
