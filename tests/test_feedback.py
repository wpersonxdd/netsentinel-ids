import numpy as np
import pandas as pd
import pytest

from nids import BENIGN, UNKNOWN
from nids.feedback import (
    NEW_FAMILY,
    SIMULATED_NOTE,
    append_feedback,
    feedback_rows,
    load_feedback_examples,
    resolve_label,
    simulate_feedback,
)
from nids.triage import FEATURE_PREFIX

NUMERIC, CATEGORICAL = ["bytes", "conn"], ["proto"]


def _alerts(n=6):
    t0 = pd.Timestamp("2026-09-01 10:00")
    return pd.DataFrame({
        "alert_id": [f"A-{i:06d}" for i in range(1, n + 1)],
        "flow_index": np.arange(n) * 3,
        "priority": "P1",
        "priority_score": np.linspace(0.9, 0.5, n),
        "predicted_class": ["DDoS"] * 3 + [UNKNOWN] * 2 + ["WebAttack"],
        "severity": "critical",
        "confidence": [0.99, 0.98, 0.97, 0.7, 0.7, 0.4],
        "anomaly_score": 1.0,
        "needs_review": [False, False, False, True, True, True],
        "src_ip": ["1.1.1.1", "2.2.2.2", "3.3.3.3", "10.0.0.9", "10.0.0.9", "8.8.8.8"],
        "dst_ip": ["10.0.0.5"] * 3 + ["52.1.1.1", "52.1.1.1", "10.0.0.6"],
        "timestamp": [t0 + pd.Timedelta(minutes=m) for m in range(n)],
        "explanation": "x",
        "recommended_action": "y",
        "true_label": ["DDoS", "DDoS", BENIGN, "Exfiltration", "Exfiltration", BENIGN],
        f"{FEATURE_PREFIX}bytes": [100.0, 120.0, 90.0, 5e6, 6e6, 300.0],
        f"{FEATURE_PREFIX}conn": [150.0, 140.0, 3.0, 1.0, 1.0, 4.0],
        f"{FEATURE_PREFIX}proto": ["TCP"] * 6,
    })


@pytest.mark.parametrize("decision, corrected, note, expected", [
    ("Confirmed threat", "DDoS", "", "DDoS"),
    ("Escalated to incident response", "Botnet", "", "Botnet"),
    ("Needs more data", "DDoS", "", None),
    ("Confirmed threat", NEW_FAMILY, "CryptoMining", "CryptoMining"),
    ("Confirmed threat", NEW_FAMILY, "", None),
    ("Confirmed threat", UNKNOWN, "", None),
    ("Confirmed threat", NEW_FAMILY, SIMULATED_NOTE, None),
])
def test_resolve_label(decision, corrected, note, expected):
    assert resolve_label(decision, corrected, note) == expected


def test_false_positive_is_labelled_benign_and_features_travel(tmp_path):
    rows = feedback_rows(_alerts().iloc[:2], "False positive", "DDoS", source="file.csv")
    assert list(rows["corrected_label"]) == [BENIGN, BENIGN]
    assert f"{FEATURE_PREFIX}bytes" in rows.columns and list(rows["flow_index"]) == [0, 3]

    path = tmp_path / "fb.csv"
    append_feedback(rows, path)
    append_feedback(feedback_rows(_alerts().iloc[:1], "Confirmed threat", "DDoS", source="file.csv"), path)
    X, y, fb, summary = load_feedback_examples(path, NUMERIC, CATEGORICAL)
    # the later decision on A-000001 wins; every row keeps its features
    assert summary["rows"] == 3 and summary["usable"] == 2
    assert dict(zip(fb["alert_id"], y)) == {"A-000001": "DDoS", "A-000002": BENIGN}
    assert list(X.columns) == NUMERIC + CATEGORICAL and X["bytes"].tolist() == [120.0, 100.0]


def test_feedback_without_features_is_rejected(tmp_path):
    path = tmp_path / "old.csv"
    pd.DataFrame({"alert_id": ["A-1"], "source": ["x"], "decision": ["Confirmed threat"], "corrected_label": ["DoS"],
                  "note": [""], "incident_id": ["INC-1"], "flow_index": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="feature values"):
        load_feedback_examples(path, NUMERIC, CATEGORICAL)


def test_simulated_analyst_reviews_needs_review_incidents_first():
    rows = simulate_feedback(_alerts(), budget=1, order="review-first")
    # the two Unknown-Anomaly flows share a source and form the top needs-review incident
    assert set(rows["alert_id"]) == {"A-000004", "A-000005"}
    assert set(rows["corrected_label"]) == {"Exfiltration"} and (rows["note"] == SIMULATED_NOTE).all()

    rows = simulate_feedback(_alerts(), budget=1, order="priority")
    assert set(rows["predicted_class"]) == {"DDoS"} and len(rows) == 3
    assert dict(zip(rows["alert_id"], rows["decision"]))["A-000003"] == "False positive"

    with pytest.raises(ValueError, match="ground truth"):
        simulate_feedback(_alerts().drop(columns="true_label"))
