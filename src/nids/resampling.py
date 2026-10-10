"""Class-imbalance strategies compared in the benchmark.

* ``none``          - train on the natural distribution (the baseline every IDS paper starts from).
* ``class_weight``  - no new rows; the loss weights each class inversely to its frequency.
* ``smote``         - synthesise minority-class rows (SMOTE) up to a share of the majority class.
* ``hybrid``        - randomly undersample the majority class, then SMOTE the minorities.
                      Faster to train and usually the best precision/recall balance.

Resampling is applied inside an imblearn ``Pipeline``, so it only ever touches training folds:
validation and test sets keep the real-world class distribution.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

log = logging.getLogger(__name__)

STRATEGIES = ("none", "class_weight", "smote", "hybrid")


def undersample_targets(counts: pd.Series, cap_ratio: float) -> dict:
    """Cap every class at ``cap_ratio`` x the second-largest class (only the majority is affected)."""
    if len(counts) < 2:
        return {}
    cap = int(cap_ratio * counts.sort_values(ascending=False).iloc[1])
    return {cls: int(min(n, cap)) for cls, n in counts.items()}


def smote_targets(counts: pd.Series, ratio: float) -> dict:
    """Raise every class smaller than ``ratio`` x the largest class up to that size."""
    target = int(ratio * counts.max())
    return {cls: target for cls, n in counts.items() if 2 <= n < target}


def make_samplers(strategy: str, y: pd.Series, cfg: dict, seed: int) -> list[tuple[str, object]]:
    """Return imblearn pipeline steps implementing ``strategy`` for training labels ``y``."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}")
    if strategy in ("none", "class_weight"):
        return []

    counts = pd.Series(y).value_counts()
    steps: list[tuple[str, object]] = []
    if strategy == "hybrid":
        under = undersample_targets(counts, cfg.get("majority_cap_ratio", 4.0))
        steps.append(("undersample", RandomUnderSampler(sampling_strategy=under, random_state=seed)))
        counts = pd.Series(under)

    targets = smote_targets(counts, cfg.get("smote_target_ratio", 0.15))
    if targets:
        # SMOTE interpolates between a row and its k nearest same-class neighbours,
        # so k must be smaller than the rarest class being oversampled.
        k = int(max(1, min(5, min(counts[c] for c in targets) - 1)))
        steps.append(("smote", SMOTE(sampling_strategy=targets, k_neighbors=k, random_state=seed)))
    log.debug("%s sampler steps: %s", strategy, [name for name, _ in steps])
    return steps


def resampled_size(strategy: str, y: pd.Series, cfg: dict) -> int:
    """Number of training rows after resampling (for reporting)."""
    counts = pd.Series(y).value_counts().to_dict()
    if strategy == "hybrid":
        counts = undersample_targets(pd.Series(counts), cfg.get("majority_cap_ratio", 4.0))
    if strategy in ("smote", "hybrid"):
        counts.update(smote_targets(pd.Series(counts), cfg.get("smote_target_ratio", 0.15)))
    return int(np.sum(list(counts.values())))
