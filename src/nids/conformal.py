"""Class-conditional (Mondrian) conformal prediction: per-family coverage guarantees for alerts.

A classifier's "confidence 0.7" is not a guarantee of anything. Conformal prediction turns probabilities into a
*prediction set* with a finite-sample guarantee: if calibration and new flows are exchangeable, the true class is in
the set with probability >= 1 - alpha.

The standard (marginal) guarantee holds on average over all flows, which under 80% benign traffic says almost nothing
about a family that is 0.2% of traffic. The Mondrian variant calibrates one threshold per class, so the guarantee holds
**for every class separately, including the rarest attack families**: the dimension this problem is about.

How it is used in NetSentinel:
* every alert lists its plausible families ("WebAttack | Benign") instead of a single over-confident label,
* an alert also needs analyst review when its set holds more than one family (a second trigger next to low
  confidence: thresholds differ per family, so the two signals catch different cases),
* "guaranteed-recall alerting" (alert whenever any attack family is in the set) shows what a per-family recall
  guarantee would cost in false alarms, a trade-off the SOC can choose knowingly.

Calibration uses the validation split, which the classifier never trains on.
"""

from __future__ import annotations

import math

import numpy as np


class MondrianConformal:
    """Per-class split-conformal thresholds on the nonconformity score 1 - p(class)."""

    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(self, proba: np.ndarray, y, classes) -> "MondrianConformal":
        self.classes_ = np.asarray([str(c) for c in classes])
        col = {c: j for j, c in enumerate(self.classes_)}
        y = np.asarray(y).astype(str)
        self.thresholds_ = np.ones(len(self.classes_))
        self.n_calibration_ = np.zeros(len(self.classes_), dtype=int)
        # A guarantee at level 1 - alpha needs ceil((n + 1)(1 - alpha)) <= n, i.e. n >= 9 calibration flows at 90%.
        self.guaranteed_ = np.zeros(len(self.classes_), dtype=bool)
        for c, j in col.items():
            scores = np.sort(1.0 - proba[y == c, j])
            n = len(scores)
            self.n_calibration_[j] = n
            k = math.ceil((n + 1) * (1 - self.alpha))  # finite-sample corrected quantile rank
            if n and k <= n:
                self.thresholds_[j] = float(scores[k - 1])
                self.guaranteed_[j] = True
        return self

    def sets(self, proba: np.ndarray) -> np.ndarray:
        """Boolean matrix (flows x classes): True where the class is in the flow's prediction set.

        Sets are never empty: the most probable family is always included. Coverage is a *lower* bound, so for an
        accurate classifier the raw procedure would drop up to alpha of even confident predictions; adding the top
        class can only raise coverage, so the guarantee still holds.

        Families with too few calibration flows for any guarantee are never added on their own (the strict
        procedure would put them in *every* set, which says nothing); they appear only as the top prediction.
        """
        sets = ((1.0 - proba) <= self.thresholds_[None, :] + 1e-12) & self.guaranteed_[None, :]
        sets[np.arange(len(proba)), proba.argmax(axis=1)] = True
        return sets

    def describe(self, sets: np.ndarray) -> list[str]:
        """Human-readable prediction sets, e.g. 'Benign | WebAttack' (classes in model order)."""
        return [" | ".join(self.classes_[row]) for row in sets]

    @property
    def guaranteed_classes(self) -> list[str]:
        """Classes with enough calibration flows for a coverage guarantee at this alpha."""
        return [c for c, g in zip(self.classes_, self.guaranteed_) if g]


def coverage_range(n_calibration: int, alpha: float, level: float = 0.95) -> tuple[float, float]:
    """Range in which realised coverage falls for 95% of calibration draws.

    The guarantee is on average over calibration sets; with n calibration flows the realised coverage follows
    Beta(k, n + 1 - k), k = ceil((n + 1)(1 - alpha)) (Vovk 2012; Angelopoulos & Bates 2021). Small n means a wide range.
    """
    from scipy.stats import beta

    k = math.ceil((n_calibration + 1) * (1 - alpha))
    if n_calibration == 0 or k > n_calibration:
        return 1.0, 1.0  # class is always included
    dist = beta(k, n_calibration + 1 - k)
    return float(dist.ppf((1 - level) / 2)), float(dist.ppf((1 + level) / 2))


def coverage_report(sets: np.ndarray, y_true, classes, benign: str) -> dict:
    """Empirical per-class coverage, set sizes, and what 'alert if any attack is in the set' would do."""
    classes = np.asarray([str(c) for c in classes])
    y_true = np.asarray(y_true).astype(str)
    col = {c: j for j, c in enumerate(classes)}
    attack_cols = np.array([c != benign for c in classes])
    size = sets.sum(axis=1)
    conformal_alert = (sets & attack_cols[None, :]).any(axis=1)
    per_class = {}
    for c in classes:
        rows = y_true == c
        if not rows.any():
            continue
        per_class[c] = {
            "flows": int(rows.sum()),
            "coverage": float(sets[rows, col[c]].mean()),
            "mean_set_size": float(size[rows].mean()),
            "singleton_rate": float((size[rows] == 1).mean()),
            # share of this class raised as an alert under guaranteed-recall alerting
            "alerted": float(conformal_alert[rows].mean()),
        }
    benign_rows = y_true == benign
    return {
        "per_class": per_class,
        "mean_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "guaranteed_alerting_false_alarm_rate": float(conformal_alert[benign_rows].mean()) if benign_rows.any() else None,
        "guaranteed_alerting_detection_rate": float(conformal_alert[~benign_rows].mean()) if (~benign_rows).any() else None,
    }
