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
             rare_threshold: float = 0.01) -> dict:
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
    out = {
        "n": total,
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean(f1)),
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
