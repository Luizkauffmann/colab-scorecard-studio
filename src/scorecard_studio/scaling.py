"""
Scorecard scaling: turn a logistic regression on WOE variables into points.

Definitions
-----------
Model (WOE convention: positive WOE = riskier)::

    logit(p_event) = b0 + sum_j  b_j * WOE_j          (b_j expected > 0)

Score scale, with ``base_odds`` = non-event : event odds at ``base_score``::

    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset + factor * ln(odds_non_event) = offset - factor * logit(p_event)

Splitting the intercept and offset evenly over the ``n`` model variables::

    points_j(bin) = -factor * b_j * WOE_bin + (offset - factor * b0) / n

so a record's total points equals its score exactly (before rounding).
Higher score = lower risk. Doubling the odds adds PDO points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class ScalingParams:
    pdo: float = 20.0
    base_score: float = 600.0
    base_odds: float = 50.0   # non-event : event odds at base_score (e.g. 50:1)

    def __post_init__(self):
        if self.pdo <= 0 or self.base_odds <= 0:
            raise ValueError("pdo and base_odds must be positive.")

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    def score_from_logit(self, logit_p_event: float) -> float:
        return self.offset - self.factor * logit_p_event

    def score_from_pd(self, p_event: float) -> float:
        return self.score_from_logit(math.log(p_event / (1 - p_event)))

    def pd_from_score(self, score: float) -> float:
        logit = (self.offset - score) / self.factor
        return 1.0 / (1.0 + math.exp(-logit))


def resolve_coefficients(bundle, coefficients: Dict[str, float]) -> Dict[str, float]:
    """Accept ``{"AGE": b}`` or model column names like ``{"opt_AGE_woe": b}``."""
    out = {}
    for key, beta in coefficients.items():
        var = key
        if var not in bundle.artifacts and key.endswith("_woe"):
            for v in bundle.artifacts:
                if key.endswith(f"{v}_woe"):
                    var = v
                    break
        if var not in bundle.artifacts:
            raise KeyError(f"Coefficient '{key}' does not match any binned variable.")
        out[var] = float(beta)
    if not out:
        raise ValueError("No coefficients given.")
    return out


def check_coefficient_signs(coefficients: Dict[str, float]) -> List[str]:
    """Variables whose coefficient is not positive. With WOE = ln(%ev/%non-ev)
    every coefficient should be positive; a negative one means the variable's
    effect is reversed in the multivariate model (collinearity, suppression)."""
    return [v for v, b in coefficients.items() if not b > 0]


def scorecard_table(bundle, coefficients: Dict[str, float], intercept: float,
                    params: ScalingParams = ScalingParams(),
                    round_points: Optional[int] = None) -> pd.DataFrame:
    """One row per (variable, bin) with its points, incl. Special and Missing bins."""
    betas = resolve_coefficients(bundle, coefficients)
    n = len(betas)
    shift = (params.offset - params.factor * float(intercept)) / n
    rows = []
    for var, beta in betas.items():
        art = bundle.artifacts[var]
        for b in art.bins:
            pts = -params.factor * beta * b["woe"] + shift
            rows.append({
                "Variable": var, "Group": b["group"], "Kind": b["kind"], "Label": b["label"],
                "Categories": " | ".join(b["categories"]) if b.get("categories") else "",
                "WOE": b["woe"], "Coefficient": beta, "Event rate": b["event_rate"],
                "Count": b["count"],
                "Points": round(pts, round_points) if round_points is not None else pts,
            })
    return pd.DataFrame(rows)
