import numpy as np
import pytest

from nids import BENIGN, UNKNOWN
from nids.evaluate import binary_view, bootstrap_metrics, evaluate


def test_binary_view_counts_any_alert_as_detection():
    y_true = [BENIGN, BENIGN, BENIGN, "DoS", "Bot"]
    y_pred = [BENIGN, "DoS", BENIGN, "Bot", BENIGN]  # wrong family still counts as detected
    out = binary_view(y_true, y_pred)
    assert out["detection_rate"] == pytest.approx(0.5)
    assert out["false_alarm_rate"] == pytest.approx(1 / 3)
    assert out["alert_precision"] == pytest.approx(0.5)


def test_unknown_anomaly_is_an_alert_but_not_a_correct_family():
    y_true = np.array([BENIGN] * 8 + ["Bot"] * 2)
    y_pred = np.array([BENIGN] * 8 + [UNKNOWN, "Bot"])
    m = evaluate(y_true, y_pred)
    assert m["detection_rate"] == 1.0 and m["false_alarm_rate"] == 0.0
    assert m["per_class"]["Bot"]["recall"] == 0.5


def test_perfect_predictions_and_rare_class_metrics():
    y = np.array([BENIGN] * 990 + ["Bot"] * 5 + ["DoS"] * 5)
    proba = np.zeros((len(y), 3))
    for j, c in enumerate([BENIGN, "Bot", "DoS"]):
        proba[y == c, j] = 1.0
    m = evaluate(y, y, proba, [BENIGN, "Bot", "DoS"])
    assert m["macro_f1"] == 1.0 and m["mcc"] == 1.0 and m["macro_pr_auc"] == 1.0
    assert sorted(m["rare_classes"]) == ["Bot", "DoS"] and m["rare_macro_f1"] == 1.0


def test_tiny_families_are_counted_and_excluded_from_supported_macro_f1():
    y = np.array([BENIGN] * 100 + ["DoS"] * 20 + ["Tiny"] * 2)
    pred = y.copy()
    pred[-2:] = BENIGN  # both Tiny flows missed
    m = evaluate(y, pred)
    assert m["tiny_classes"] == {"Tiny": {"support": 2, "found": 0}}
    assert m["supported_classes"] == [BENIGN, "DoS"] and m["supported_macro_f1"] > 0.99 > m["macro_f1"]
    boot = bootstrap_metrics(y, {"m": pred}, n_boot=50)["models"]["m"]
    assert boot["supported_macro_f1"]["point"] == pytest.approx(m["supported_macro_f1"])
