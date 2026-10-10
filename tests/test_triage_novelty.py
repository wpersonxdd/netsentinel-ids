import numpy as np
import pandas as pd
import pytest

from nids import BENIGN, UNKNOWN
from nids.novelty import NoveltyDetector, psi
from nids.triage import DEFAULT_SEVERITY, BenignBaseline, build_alerts, group_incidents, priority_label


def test_priority_cut_points():
    assert list(priority_label(np.array([0.95, 0.7, 0.5, 0.3, 0.1]))) == ["P1", "P1", "P2", "P3", "P4"]


def _benign(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "destination_port": rng.choice([80, 443], n).astype(float),
        "conn_count": rng.poisson(3, n).astype(float),
        "bytes": rng.lognormal(6, 0.5, n),
    })


def test_alerts_are_prioritised_and_explained():
    baseline = BenignBaseline().fit(_benign())
    assert "destination_port" not in baseline.columns_  # ports are identifiers, not magnitudes
    X = pd.DataFrame({"destination_port": [443.0, 80.0, 22.0, 443.0],
                      "conn_count": [3.0, 150.0, 40.0, 3.0],
                      "bytes": [400.0, 300.0, 5e6, 400.0]})
    prediction = {
        "label": np.array([BENIGN, "DDoS", UNKNOWN, "PortScan"]),
        "confidence": np.array([0.99, 0.97, 0.9, 0.55]),
        "anomaly_score": np.array([1.0, 3.0, 4.0, 1.1]),
        "novel": np.array([False, True, True, False]),
    }
    ctx = pd.DataFrame({"src_ip": ["a", "b", "c", "d"], "dst_ip": "z", "dst_port": [443, 80, 22, 443]})
    alerts = build_alerts(prediction, X, ctx, baseline, DEFAULT_SEVERITY, novelty_threshold=1.5)

    assert list(alerts["predicted_class"]) == ["DDoS", UNKNOWN, "PortScan"]  # benign dropped, sorted by priority
    assert list(alerts["alert_id"]) == ["A-000001", "A-000002", "A-000003"]
    # each alert carries its flow's position and feature values (what retraining needs)
    assert list(alerts["flow_index"]) == [1, 2, 3]
    assert alerts.loc[0, "feat_conn_count"] == 150.0 and alerts.loc[1, "feat_bytes"] == 5e6
    assert alerts.loc[0, "priority"] == "P1" and alerts.loc[2, "priority"] in ("P3", "P4")
    assert "conn_count" in alerts.loc[0, "explanation"]
    unknown = alerts[alerts["predicted_class"] == UNKNOWN].iloc[0]
    assert unknown["needs_review"] and unknown["severity"] == "low" and "bytes" in unknown["explanation"]
    assert bool(alerts.loc[2, "needs_review"])  # confidence 0.55 < 0.6


def test_incidents_group_by_shared_endpoint_and_time():
    t0 = pd.Timestamp("2026-09-01 10:00")
    alerts = pd.DataFrame({
        "alert_id": [f"A-{i}" for i in range(8)],
        "priority": "P1",
        "priority_score": [0.9, 0.95, 0.9, 0.9, 0.5, 0.5, 0.5, 0.4],
        "predicted_class": ["DDoS"] * 4 + ["PortScan"] * 3 + ["DDoS"],
        "severity": ["critical"] * 4 + ["low"] * 3 + ["critical"],
        "confidence": 0.9,
        "anomaly_score": 1.0,
        "needs_review": False,
        # DDoS: many sources, one victim; the last DDoS flow comes hours later -> separate incident
        "src_ip": ["1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4", "9.9.9.9", "9.9.9.9", "9.9.9.9", "5.5.5.5"],
        "dst_ip": ["10.0.0.5"] * 4 + ["10.0.0.7", "10.0.0.8", "10.0.0.9", "10.0.0.5"],
        "timestamp": [t0 + pd.Timedelta(minutes=m) for m in (0, 1, 2, 3, 5, 6, 7, 300)],
        "explanation": "x",
        "recommended_action": "y",
    })
    incidents, mapped = group_incidents(alerts)
    assert len(incidents) == 3
    first = incidents.iloc[0]
    assert first["predicted_class"] == "DDoS" and first["flows"] == 4 and first["grouped_by"] == "dst_ip"
    assert first["top_alert"] == "A-1"  # highest priority member
    scan = incidents[incidents["predicted_class"] == "PortScan"].iloc[0]
    assert scan["grouped_by"] == "src_ip" and scan["shared_endpoint"] == "9.9.9.9" and scan["flows"] == 3
    assert mapped["incident_id"].nunique() == 3 and mapped["incident_id"].notna().all()


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi(ref, pd.Series(rng.normal(1.5, 1, 5000))) > 0.25


@pytest.mark.parametrize("method", ["lof", "iforest"])
def test_novelty_detector_flags_outliers_within_budget(method):
    benign = _benign(2000, seed=1)
    detector = NoveltyDetector(["destination_port", "conn_count", "bytes"], [], benign_fpr=0.02,
                               method=method).fit(benign.iloc[:1500], benign.iloc[1500:])
    fresh = _benign(1000, seed=2)
    assert detector.flag(fresh).mean() < 0.08
    outliers = pd.DataFrame({"destination_port": [443.0] * 20, "conn_count": [400.0] * 20,
                             "bytes": np.linspace(5e7, 9e7, 20)})
    assert detector.flag(outliers).mean() > 0.9
