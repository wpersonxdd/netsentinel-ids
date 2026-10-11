"""Metrics that stay honest under class imbalance.

Accuracy is useless here: predicting "benign" for everything scores ~80%. We report
* macro-F1 / macro-recall  - every class counts equally, so rare attacks matter as much as floods,
* rare-class macro-F1      - the same, restricted to attack families below 1% prevalence,
* per-class precision/recall/F1 and PR-AUC (average precision; ROC-AUC flatters rare classes),
* MCC                      - a single correlation-style score robust to imbalance,
* SOC view: detection rate (attacks raised as *any* alert) and false-alarm rate (benign flows alerted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_fscore_support

from nids import BENIGN
from nids.data import class_order


def binary_view(y_true, y_pred, benign: str = BENIGN) -> dict:
    """Attack-vs-benign metrics: any non-benign prediction is an alert."""
    true_attack = np.asarray(y_true) != benign
    pred_attack = np.asarray(y_pred) != benign
    tp = int((true_attack & pred_attack).sum())
    fp = int((~true_attack & pred_attack).sum())
    fn = int((true_attack & ~pred_attack).sum())
    tn = int((~true_attack & ~pred_attack).sum())
    return {
        "detection_rate": tp / max(tp + fn, 1),
        "false_alarm_rate": fp / max(fp + tn, 1),
        "alert_precision": tp / max(tp + fp, 1),
        "alerts": tp + fp,
    }


def evaluate(y_true, y_pred, proba: np.ndarray | None = None, proba_classes=None, benign: str = BENIGN,
             rare_threshold: float = 0.01, min_support: int = 10) -> dict:
    """Full metric set for one set of predictions."""
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    classes = class_order(np.unique(y_true))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)

    pr_auc = {}
    if proba is not None and proba_classes is not None:
        col = {str(c): j for j, c in enumerate(proba_classes)}
        for c in classes:
            if c in col:
                pr_auc[c] = float(average_precision_score(y_true == c, proba[:, col[c]]))

    total = len(y_true)
    per_class = {
        c: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "prevalence": float(support[i] / total),
            "pr_auc": pr_auc.get(c),
        }
        for i, c in enumerate(classes)
    }
    rare = [c for c in classes if c != benign and per_class[c]["prevalence"] < rare_threshold]
    attacks = [c for c in classes if c != benign]
    # Families with a handful of test flows: their F1 is decided by one or two flows, so macro-F1 is also reported
    # over the families that can actually be measured, and the tiny ones as plain counts.
    supported = [c for c in classes if per_class[c]["support"] >= min_support]
    tiny = {c: {"support": per_class[c]["support"],
                "found": int(round(per_class[c]["recall"] * per_class[c]["support"]))}
            for c in classes if per_class[c]["support"] < min_support}
    out = {
        "n": total,
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean(f1)),
        "supported_macro_f1": float(np.mean([per_class[c]["f1"] for c in supported])) if supported else None,
        "supported_classes": supported,
        "tiny_classes": tiny,
        "macro_recall": float(np.mean(recall)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "attack_macro_f1": float(np.mean([per_class[c]["f1"] for c in attacks])) if attacks else None,
        "rare_classes": rare,
        "rare_macro_f1": float(np.mean([per_class[c]["f1"] for c in rare])) if rare else None,
        "rare_macro_recall": float(np.mean([per_class[c]["recall"] for c in rare])) if rare else None,
        "macro_pr_auc": float(np.mean(list(pr_auc.values()))) if pr_auc else None,
        "per_class": per_class,
    }
    out.update(binary_view(y_true, y_pred, benign))
    return out


def per_class_frame(metrics: dict) -> pd.DataFrame:
    df = pd.DataFrame(metrics["per_class"]).T
    df.index.name = "class"
    return df[["support", "prevalence", "precision", "recall", "f1", "pr_auc"]]


HEADLINE = ["macro_f1", "rare_macro_f1", "macro_recall", "mcc", "macro_pr_auc", "detection_rate", "false_alarm_rate"]


def headline(metrics: dict) -> dict:
    return {k: metrics.get(k) for k in HEADLINE}


# --------------------------------------------------------------------------------------------------
# Uncertainty: stratified bootstrap confidence intervals and paired comparisons
# --------------------------------------------------------------------------------------------------

BOOTSTRAP_METRICS = ("macro_f1", "supported_macro_f1", "rare_macro_f1", "macro_recall", "detection_rate",
                     "false_alarm_rate")


def _metrics_from_codes(t: np.ndarray, p: np.ndarray, k: int, rare: np.ndarray, benign: int,
                        supported: np.ndarray | None = None) -> dict:
    """Metrics from integer-coded labels; prediction code k = 'not a true class' (e.g. Unknown-Anomaly)."""
    cm = np.bincount(t * (k + 1) + p, minlength=k * (k + 1)).reshape(k, k + 1)
    tp = np.diag(cm[:, :k]).astype(float)
    support = cm.sum(axis=1)
    predicted = cm[:, :k].sum(axis=0)
    recall = tp / np.maximum(support, 1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    attack_rows = np.arange(k) != benign
    benign_alerted = cm[benign].sum() - cm[benign, benign] if benign >= 0 else 0
    attacks_alerted = (cm[attack_rows].sum() - cm[attack_rows, benign].sum()) if benign >= 0 else cm.sum()
    return {
        "macro_f1": f1.mean(),
        "supported_macro_f1": f1[supported].mean() if supported is not None and supported.any() else np.nan,
        "rare_macro_f1": f1[rare].mean() if rare.any() else np.nan,
        "macro_recall": recall.mean(),
        "detection_rate": attacks_alerted / max(cm[attack_rows].sum(), 1),
        "false_alarm_rate": benign_alerted / max(cm[benign].sum(), 1) if benign >= 0 else np.nan,
    }


def bootstrap_metrics(y_true, predictions: dict, pairs: list[tuple[str, str]] = (), n_boot: int = 1000,
                      seed: int = 42, benign: str = BENIGN, rare_threshold: float = 0.01,
                      confidence: float = 0.95, min_support: int = 10) -> dict:
    """95% confidence intervals for each model's metrics, and paired differences between models.

    The bootstrap is *stratified*: every resample keeps each class's size, so a family with 29 test flows never
    vanishes from a resample. Pairs ``(a, b)`` are evaluated on the *same* resamples, giving a CI for ``b - a`` and the
    share of resamples in which b is not better than a (a one-sided bootstrap p-value).
    """
    y_true = np.asarray(y_true).astype(str)
    classes = class_order(np.unique(y_true))
    k = len(classes)
    code = {c: i for i, c in enumerate(classes)}
    t = np.array([code[c] for c in y_true])
    preds = {name: np.array([code.get(str(c), k) for c in p]) for name, p in predictions.items()}
    prevalence = np.bincount(t, minlength=k) / len(t)
    rare = np.array([(c != benign) and (prevalence[i] < rare_threshold) for i, c in enumerate(classes)])
    supported = np.bincount(t, minlength=k) >= min_support
    b_idx = code.get(benign, -1)

    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(t == i) for i in range(k)]
    samples = {name: {m: np.empty(n_boot) for m in BOOTSTRAP_METRICS} for name in preds}
    for b in range(n_boot):
        rows = np.concatenate([idx[rng.integers(0, len(idx), len(idx))] for idx in by_class])
        for name, p in preds.items():
            for m, v in _metrics_from_codes(t[rows], p[rows], k, rare, b_idx, supported).items():
                samples[name][m][b] = v

    lo, hi = 100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2
    out = {"n_boot": n_boot, "confidence": confidence, "models": {}, "pairs": {}}
    for name, p in preds.items():
        point = _metrics_from_codes(t, p, k, rare, b_idx, supported)
        out["models"][name] = {m: {"point": float(point[m]), "low": float(np.nanpercentile(samples[name][m], lo)),
                                   "high": float(np.nanpercentile(samples[name][m], hi))}
                               for m in BOOTSTRAP_METRICS if not np.isnan(point[m])}
    for a, b in pairs:
        diffs = {}
        for m in BOOTSTRAP_METRICS:
            d = samples[b][m] - samples[a][m]
            if np.isnan(d).all():
                continue
            better = -1 if m == "false_alarm_rate" else 1  # for false alarms, lower is better
            diffs[m] = {
                "delta": float(out["models"][b][m]["point"] - out["models"][a][m]["point"]),
                "low": float(np.nanpercentile(d, lo)),
                "high": float(np.nanpercentile(d, hi)),
                "p_not_better": float(np.nanmean(better * d <= 0)),
            }
        out["pairs"][f"{a}->{b}"] = diffs
    return out
