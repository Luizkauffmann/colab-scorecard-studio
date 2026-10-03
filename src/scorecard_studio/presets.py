"""
Domain presets: default binning settings and labels per use case.

Use them as a starting point, not as a substitute for judgement::

    from scorecard_studio import get_preset
    p = get_preset("fraud")
    engine.fit_all(**p["fit_params"])
"""

from __future__ import annotations

import copy
from typing import Any, Dict

PRESETS: Dict[str, Dict[str, Any]] = {
    "credit_pd": {
        "description": "Application / origination PD scorecard.",
        "event_label": "Default", "non_event_label": "Non-default",
        "fit_params": dict(max_bins=6, monotonic="auto", divergence="iv",
                           min_bin_size=0.05, min_bin_n_event=None, cat_cutoff=0.05),
        "split": "time",          # out-of-time validation when a date column exists
        "metrics": ["gini", "ks", "auc", "psi"],
        # Univariate screening on Train. IV above iv_max is held for review, not dropped.
        "screening": dict(iv_min=0.10, iv_max=0.50),
    },
    "fraud": {
        "description": "Transaction or application fraud; event rates often below 1%.",
        "event_label": "Fraud", "non_event_label": "Genuine",
        # Small share per bin is fine, but every bin needs enough frauds
        # for its WOE to mean anything.
        "fit_params": dict(max_bins=5, monotonic="auto", divergence="iv",
                           min_bin_size=0.02, min_bin_n_event=30, cat_cutoff=0.02),
        "split": "time",
        "metrics": ["pr_auc", "recall_at_alert_rate", "gini", "ks"],
        # IV is small and noisy when events are rare; a 0.10 floor drops useful variables.
        "screening": dict(iv_min=0.02, iv_max=0.50),
    },
    "aml": {
        "description": "AML / transaction monitoring; few, noisy (SAR/case) labels.",
        "event_label": "SAR", "non_event_label": "No SAR",
        "fit_params": dict(max_bins=4, monotonic="auto", divergence="iv",
                           min_bin_size=0.03, min_bin_n_event=20, cat_cutoff=0.03),
        "split": "time",
        "metrics": ["pr_auc", "recall_at_alert_rate", "gini", "ks"],
        "screening": dict(iv_min=0.02, iv_max=0.50),
    },
    "kaggle": {
        "description": "Generic binary classification. Fit bins out-of-fold when cross-validating.",
        "event_label": "Positive", "non_event_label": "Negative",
        "fit_params": dict(max_bins=10, monotonic="none", divergence="iv",
                           min_bin_size=0.02, min_bin_n_event=None, cat_cutoff=0.01),
        "split": "random",
        "metrics": ["auc", "gini", "ks"],
        "screening": dict(iv_min=0.02, iv_max=0.50),
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """Deep copy of a preset, safe to modify."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return copy.deepcopy(PRESETS[name])
