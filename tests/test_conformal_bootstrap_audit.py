import numpy as np
import pandas as pd
import pytest

from nids import BENIGN, UNKNOWN
from nids.audit import identifier_suspects, shortcut_scan
from nids.conformal import MondrianConformal, coverage_range, coverage_report
from nids.data import PreparedData
from nids.evaluate import bootstrap_metrics, evaluate


def _noisy_proba(y_idx: np.ndarray, k: int, strength: float, rng) -> np.ndarray:
    """Probabilities that favour the true class with some noise (exchangeable across calls)."""
    logits = rng.normal(0, 1, (len(y_idx), k))
    logits[np.arange(len(y_idx)), y_idx] += strength
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def test_mondrian_coverage_holds_per_class_including_rare():
    rng = np.random.default_rng(0)
    classes = [BENIGN, "DoS", "Rare"]
    priors = [0.90, 0.09, 0.01]
    y_cal = rng.choice(3, 20_000, p=priors)
    y_test = rng.choice(3, 20_000, p=priors)
    conf = MondrianConformal(alpha=0.1).fit(_noisy_proba(y_cal, 3, 1.5, rng), np.array(classes)[y_cal], classes)
    sets = conf.sets(_noisy_proba(y_test, 3, 1.5, rng))
    report = coverage_report(sets, np.array(classes)[y_test], classes, BENIGN)
    for c in classes:  # the guarantee is per class, rare class included (small tolerance for sampling)
        assert report["per_class"][c]["coverage"] >= 0.88, c
    assert sets.sum(axis=1).min() >= 1  # never empty


def test_family_too_small_for_a_guarantee_is_only_shown_as_top_prediction():
    rng = np.random.default_rng(1)
    y = np.array([BENIGN] * 200 + ["Tiny"] * 3)  # 3 < 9 calibration flows: no 90% guarantee is possible
    proba = _noisy_proba((y == "Tiny").astype(int), 2, 2.0, rng)
    conf = MondrianConformal(alpha=0.1).fit(proba, y, [BENIGN, "Tiny"])
    assert conf.guaranteed_classes == [BENIGN] and not conf.guaranteed_[1]
    sets = conf.sets(proba)
    top_is_tiny = proba.argmax(axis=1) == 1
    assert (sets[:, 1] == top_is_tiny).all()  # never added on its own, which would put it in every set
    assert conf.describe(np.array([[True, True], [True, False]])) == [f"{BENIGN} | Tiny", BENIGN]


def test_coverage_range_narrows_with_more_calibration_flows():
    lo_small, hi_small = coverage_range(20, 0.1)
    lo_big, hi_big = coverage_range(2000, 0.1)
    assert lo_small < lo_big <= 0.9 <= hi_big < hi_small <= 1.0
    assert coverage_range(5, 0.1) == (1.0, 1.0)  # too few flows: the class is always in the set


def test_bootstrap_ci_and_paired_difference():
    rng = np.random.default_rng(2)
    y = np.array([BENIGN] * 900 + ["DoS"] * 80 + ["Bot"] * 20)
    good = y.copy()
    good[rng.choice(len(y), 30, replace=False)] = BENIGN  # a few misses
    bad = y.copy()
    bad[900:] = np.where(rng.random(100) < 0.5, BENIGN, bad[900:])  # misses half of the attacks
    boot = bootstrap_metrics(y, {"bad": bad, "good": good, "same": good}, pairs=[("bad", "good"), ("good", "same")],
                             n_boot=300, seed=0)
    m = boot["models"]["good"]["macro_f1"]
    assert m["low"] <= m["point"] <= m["high"]
    assert m["point"] == pytest.approx(evaluate(y, good)["macro_f1"])  # same metric definition as evaluate()
    up = boot["pairs"]["bad->good"]["macro_f1"]
    assert up["low"] > 0 and up["p_not_better"] < 0.01
    same = boot["pairs"]["good->same"]["macro_f1"]
    assert same["delta"] == 0 and same["low"] == 0 == same["high"]


def test_bootstrap_counts_unknown_as_alert():
    y = np.array([BENIGN] * 50 + ["Bot"] * 10)
    pred = y.copy()
    pred[50:55] = UNKNOWN
    pt = bootstrap_metrics(y, {"m": pred}, n_boot=50)["models"]["m"]
    assert pt["detection_rate"]["point"] == 1.0 and pt["false_alarm_rate"]["point"] == 0.0


def _audit_data(n=3000, seed=3) -> PreparedData:
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.choice([BENIGN, "DoS", "Bot"], n, p=[0.8, 0.15, 0.05]))
    X = pd.DataFrame({
        "noise": rng.normal(size=n),
        "duration": rng.integers(1, 10**7, n).astype(float),             # unique-ish integers, NOT an ID
        "row_id": np.arange(n, dtype=float),                             # an ID that tracks row order
        "leak": (y == "Bot").astype(float) * 100 + rng.normal(0, 1, n),  # isolates Bot perfectly
    })
    return PreparedData(X, y, pd.DataFrame(index=X.index), None, list(X.columns), [], "label", "test")


def test_shortcut_scan_flags_a_feature_that_isolates_a_family():
    scan = shortcut_scan(_audit_data(), full_macro_f1=0.9)
    assert scan.loc["leak", "best_isolated_class"] == "Bot" and scan.loc["leak", "best_class_f1"] > 0.98
    assert scan.loc["leak", "flag"] != "" and scan.loc["noise", "flag"] == ""


def test_identifier_suspects_finds_ids_not_measurements():
    suspects = identifier_suspects(_audit_data())
    assert "row_id" in suspects and "duration" not in suspects and "noise" not in suspects
