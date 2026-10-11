"""Runs every command on a small synthetic dataset: the same path the hackathon dataset will take."""

import json
from pathlib import Path

import pandas as pd

from nids.bundle import ModelBundle
from nids.cli import main
from nids.experiments import (run_audit, run_benchmark, run_drift, run_novelty, run_protocols, run_retrain,
                              run_train)
from nids.feedback import simulate_feedback
from nids.report import write_report


def test_full_pipeline(small_cfg, synthetic_csv, tmp_path):
    bench = run_benchmark(small_cfg)
    assert bench["best"]["model"] == "hist_gb" and len(bench["rows"]) == 2

    final = run_train(small_cfg)
    m = final["metrics"]
    assert 0.5 < m["hybrid"]["macro_f1"] <= 1.0
    assert m["hybrid"]["detection_rate"] >= m["classifier"]["detection_rate"]  # novelty only adds alerts
    model_file = Path(final["artifacts"]["model"])
    assert model_file.exists() and Path(final["artifacts"]["alerts"]).exists()
    # uncertainty and per-family guarantees travel with the headline numbers
    ci = final["bootstrap"]["models"]["classifier"]["macro_f1"]
    assert ci["low"] <= m["classifier"]["macro_f1"] <= ci["high"]
    assert "baseline->classifier" in final["bootstrap"]["pairs"]
    assert final["conformal"]["target_coverage"] == 0.9 and final["conformal"]["mean_set_size"] >= 1

    audit = run_audit(small_cfg)
    assert audit["shortcut_scan"] and "identifier_suspects" in audit
    protocols = run_protocols(small_cfg)["protocols"]
    assert {"random", "session", "temporal"} <= set(protocols)
    assert "(0 shared)" in protocols["session"]["note"]  # no session spans train and test
    assert "Exfiltration" in protocols["temporal"]["unseen_in_training"]

    nov = run_novelty(small_cfg)
    assert nov["rows"] and all(r["hybrid_detection"] >= r["classifier_detection"] for r in nov["rows"])

    drift = run_drift(small_cfg)
    assert "future_static" in drift["metrics"] and "Exfiltration" in [f["family"] for f in drift["new_families"]]

    # close the loop: a simulated analyst labels incidents, the model is retrained on them
    artifacts = Path(small_cfg["paths"]["artifacts"])
    feedback = simulate_feedback(pd.read_csv(artifacts / "alerts_test.csv"), budget=15)
    feedback.to_csv(tmp_path / "feedback.csv", index=False)
    retrained_path = tmp_path / "retrained.joblib"
    rt = run_retrain(small_cfg, tmp_path / "feedback.csv", out_path=retrained_path)
    assert rt["feedback"]["usable"] == len(feedback) > 0 and rt["feedback"]["simulated"]
    ev = rt["evaluation"]
    assert ev["excluded_reviewed_flows"] == feedback["flow_index"].nunique()
    assert ev["evaluated_flows"] == ev["test_flows"] - ev["excluded_reviewed_flows"]
    assert retrained_path.exists() and model_file.exists()  # --out elsewhere leaves the deployed model in place
    assert ModelBundle.load(retrained_path).metadata["feedback_flows"] == len(feedback)

    report = write_report(small_cfg).read_text(encoding="utf-8")
    for heading in ("### Data audit", "## 2. Benchmark", "## 3. Final model", "### 3.1 Is the gain real",
                    "### 3.2 How optimistic", "### 3.3 New: per-family guarantees", "## 6. Novel attacks",
                    "## 7. Drift", "## 8. Closing"):
        assert heading in report

    bundle = ModelBundle.load(model_file)
    raw = pd.read_csv(synthetic_csv, nrows=1500)
    alerts, flows = bundle.score_frame(raw)
    assert len(flows) == 1500 and len(alerts) > 0
    assert {"alert_id", "priority", "predicted_class", "explanation", "recommended_action"} <= set(alerts.columns)
    assert alerts["priority_score"].is_monotonic_decreasing


def test_retrain_replaces_model_and_archives_previous(small_cfg, tmp_path):
    run_train(small_cfg)
    artifacts = Path(small_cfg["paths"]["artifacts"])
    simulate_feedback(pd.read_csv(artifacts / "alerts_test.csv"), budget=5).to_csv(tmp_path / "fb.csv", index=False)
    rt = run_retrain(small_cfg, tmp_path / "fb.csv")
    archived = Path(rt["artifacts"]["archived_previous"])
    assert archived.exists() and archived.parent.name == "model_history"
    assert ModelBundle.load(artifacts / "netsentinel_model.joblib").metadata["retrained_from"]


def test_cli_score_command(small_cfg, synthetic_csv, tmp_path):
    run_train(small_cfg)
    model = Path(small_cfg["paths"]["artifacts"]) / "netsentinel_model.joblib"
    out = tmp_path / "alerts.csv"
    assert main(["score", "--model", str(model), "--input", str(synthetic_csv), "--out", str(out)]) == 0
    alerts = pd.read_csv(out)
    incidents = pd.read_csv(tmp_path / "alerts_incidents.csv")
    assert len(alerts) > 0 and 0 < len(incidents) < len(alerts)
    assert set(alerts["incident_id"]) == set(incidents["incident_id"])
    assert json.loads((Path(small_cfg["paths"]["reports"]) / "results" / "final.json").read_text())["model"]
