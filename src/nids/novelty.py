"""Handling attack patterns the classifier has never seen.

A supervised classifier can only name attacks from its training labels; a zero-day that resembles
nothing it knows tends to be called "benign". NetSentinel pairs it with a novelty detector trained
on benign traffic only:

    final label = classifier label, except
                  classifier says Benign  AND  anomaly score > threshold  ->  "Unknown-Anomaly"

The threshold is the (1 - benign_fpr) quantile of anomaly scores on held-out *benign* validation
traffic, so the extra false-alarm budget is explicit (1% by default).

Detector choice: Local Outlier Factor (default) compares each flow with the local density of its
nearest benign neighbours. At the same 1% false-alarm budget it had the best mean detection of unseen
attack families, about twice Isolation Forest's and ahead of an autoencoder (scripts/compare_novelty_detectors.py,
docs/METHODOLOGY.md). Benign traffic is a mix of very different services, and a single global model blurs them.

Drift monitoring uses the Population Stability Index (PSI): PSI > 0.25 is the conventional
"significant shift" signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from nids.features import build_preprocessor

METHODS = ("lof", "iforest")


class NoveltyDetector:
    """Benign-only novelty detector (LOF or Isolation Forest) with a calibrated alert threshold."""

    def __init__(self, numeric: list[str], categorical: list[str], benign_fpr: float = 0.01, method: str = "lof",
                 n_neighbors: int = 10, max_fit_rows: int = 20_000, n_estimators: int = 200, seed: int = 42):
        if method not in METHODS:
            raise ValueError(f"Unknown novelty method '{method}'. Choose from {METHODS}")
        self.numeric = numeric
        self.categorical = categorical
        self.benign_fpr = benign_fpr
        self.method = method
        self.n_neighbors = n_neighbors
        self.max_fit_rows = max_fit_rows
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X_benign: pd.DataFrame, X_benign_calibration: pd.DataFrame,
            X_always: pd.DataFrame | None = None) -> "NoveltyDetector":
        """Fit on benign flows. ``X_always``: benign flows that must be in the reference set (e.g. false
        positives confirmed by analysts), so that pattern stops being flagged as novel."""
        self.pre_ = build_preprocessor(self.numeric, self.categorical).fit(X_benign)
        Z = self.pre_.transform(X_benign)
        Z_always = self.pre_.transform(X_always) if X_always is not None and len(X_always) else Z[:0]
        budget = max(self.max_fit_rows - len(Z_always), 0)
        if len(Z) > budget:  # neighbour search cost grows with the reference set
            Z = Z[np.random.default_rng(self.seed).choice(len(Z), budget, replace=False)]
        Z = np.vstack([Z, Z_always])
        if self.method == "lof":
            self.model_ = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True, n_jobs=-1).fit(Z)
        else:
            self.model_ = IsolationForest(n_estimators=self.n_estimators, max_samples=min(2048, len(Z)),
                                          random_state=self.seed, n_jobs=-1).fit(Z)
        self.threshold_ = float(np.quantile(self.score(X_benign_calibration), 1 - self.benign_fpr))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Anomaly score: higher = less like benign traffic."""
        return -self.model_.score_samples(self.pre_.transform(X))

    def flag(self, X: pd.DataFrame) -> np.ndarray:
        return self.score(X) > self.threshold_


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index of one feature (quantile bins of the reference distribution)."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    values = np.unique(ref)
    if len(values) <= bins:  # discrete feature: one bin per observed value (+ everything else)
        edges = np.concatenate([[-np.inf], (values[:-1] + values[1:]) / 2, [np.inf]])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_table(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.Series({c: psi(reference[c], current[c]) for c in columns}).sort_values(ascending=False)
