"""Data audit: quality checks and an automatic shortcut / leakage scan, run before any model is trusted.

Public IDS datasets are notorious for *shortcut features*: a column that separates attacks from benign traffic
for reasons unrelated to attack behaviour (a lab IP, a capture-specific TTL or window size, a timestamp). A model
that learns them scores brilliantly offline and fails on a new network.

The scan trains a tiny decision tree on **each feature alone** (class-balanced, 3-fold CV) and reports how much of
the full model's macro-F1 that single feature reaches, and which family it isolates best. A single feature that
nearly matches the full model, or perfectly isolates a family, is flagged for a human to judge: legitimate signal
(a port scan *is* many destination ports) or an artefact of the capture.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from nids.data import PreparedData, stratified_subsample

log = logging.getLogger(__name__)


def quality_table(data: PreparedData) -> pd.DataFrame:
    X = data.X
    rows = []
    for c in data.feature_columns:
        col = X[c]
        rows.append({
            "feature": c,
            "type": "categorical" if c in data.categorical else "numeric",
            "missing_rate": float(col.isna().mean()),
            "distinct": int(col.nunique(dropna=True)),
            "distinct_ratio": float(col.nunique(dropna=True) / max(len(col), 1)),
        })
    return pd.DataFrame(rows).set_index("feature")


def shortcut_scan(data: PreparedData, full_macro_f1: float | None = None, max_rows: int = 30_000,
                  seed: int = 42) -> pd.DataFrame:
    """Single-feature macro-F1 (3-fold CV, depth-4 balanced tree) and best per-family F1 for every feature."""
    pos = stratified_subsample(data.y, max_rows, seed, min_per_class=300)
    X, y = data.X.iloc[pos], data.y.iloc[pos].to_numpy()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    classes = np.unique(y)
    rows = []
    for c in data.feature_columns:
        col = X[[c]]
        if c in data.categorical:
            values = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(col.astype(str))
        else:
            values = col.to_numpy(dtype=float)
            values = np.where(np.isnan(values), np.nanmedian(values) if np.isfinite(values).any() else 0.0, values)
        tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = cross_val_predict(tree, values, y, cv=cv)
        per_class = f1_score(y, pred, labels=classes, average=None, zero_division=0)
        best = int(np.argmax(per_class))
        rows.append({
            "feature": c,
            "single_feature_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "best_isolated_class": str(classes[best]),
            "best_class_f1": float(per_class[best]),
        })
    out = pd.DataFrame(rows).set_index("feature").sort_values("single_feature_macro_f1", ascending=False)
    if full_macro_f1:
        out["share_of_full_model"] = out["single_feature_macro_f1"] / full_macro_f1
    out["flag"] = ""
    near_full = out.get("share_of_full_model", pd.Series(0, index=out.index)) >= 0.8
    out.loc[near_full, "flag"] = "near full-model score from one feature"
    isolates = (out["best_class_f1"] >= 0.98) & (out["flag"] == "")
    out.loc[isolates, "flag"] = "alone isolates a family (F1 >= 0.98)"
    return out


def identifier_suspects(data: PreparedData, threshold: float = 0.9) -> dict[str, str]:
    """Numeric features that track row order or time: row IDs, counters and clocks, not flow measurements."""
    orders = {"row order": np.arange(len(data.X))}
    if data.time is not None:
        orders["time"] = np.argsort(np.argsort(data.time.to_numpy(), kind="stable"), kind="stable")
    suspects = {}
    for c in data.numeric:
        values = pd.Series(data.X[c].to_numpy(dtype=float))
        if values.notna().sum() < 10 or values.nunique() < 10:
            continue
        for name, rank in orders.items():
            rho = values.corr(pd.Series(rank, dtype=float), method="spearman")
            if abs(rho) > threshold:
                suspects[c] = f"tracks {name} (Spearman {rho:+.2f}): likely an ID, counter or clock"
                break
    return suspects
