"""Builds reports/REPORT.md (the evaluation deliverable) from the JSON results of each experiment."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from nids import BENIGN, UNKNOWN
from nids.config import output_dirs
from nids.experiments import (ALERTS_FILE, load_json, plot_benchmark, plot_drift, plot_final, plot_novelty,
                              plot_retrain)
from nids.models import MODEL_LABELS, STRATEGY_LABELS

log = logging.getLogger(__name__)

METRIC_LABELS = {
    "macro_f1": "Macro-F1",
    "rare_macro_f1": "Rare-class F1 (<1% families)",
    "macro_recall": "Macro recall",
    "mcc": "MCC",
    "macro_pr_auc": "Macro PR-AUC",
    "detection_rate": "Detection rate (attacks alerted)",
    "false_alarm_rate": "False-alarm rate (benign alerted)",
}


def _num(v, digits: int = 3) -> str:
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.{digits}f}"


def _pct(v, digits: int = 1) -> str:
    return "–" if v is None else f"{100 * v:.{digits}f}%"


def _table(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    frame = df.reset_index() if index else df
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _img(name: str, alt: str, figures: Path) -> str:
    return f"![{alt}](figures/{name})" if (figures / name).exists() else f"_({name} not generated)_"


def _dataset_section(ds: dict, figures: Path) -> list[str]:
    counts = pd.Series(ds["class_counts"])
    table = pd.DataFrame({"Flows": counts.map("{:,}".format),
                          "Share": (counts / counts.sum()).map(lambda v: f"{v:.2%}")})
    table.index.name = "Class"
    lines = [
        "## 1. Dataset",
        "",
        f"- Source: `{ds['source']}`, label column `{ds['label_column']}`",
        f"- {ds['rows']:,} flows after cleaning; {ds['features']} features "
        f"({ds['numeric_features']} numeric, {ds['categorical_features']} categorical)",
        f"- Imbalance ratio (largest / smallest class): **{ds['imbalance_ratio']:,.0f} : 1**",
        f"- Exact duplicate flows removed: {ds['dropped_duplicates']:,}"
        + (f"; classes dropped as too rare to learn: {ds['dropped_rare_classes']}" if ds["dropped_rare_classes"] else ""),
    ]
    if ds.get("split"):
        s = ds["split"]
        lines.append(f"- Stratified split: {s['train']:,} train / {s['validation']:,} validation / {s['test']:,} test")
    lines += ["", _table(table), "", _img("class_distribution.png", "Class distribution", figures), ""]
    return lines


def _benchmark_section(bench: dict, figures: Path) -> list[str]:
    rows = [r for r in bench["rows"] if "error" not in r]
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    best = bench["best"]
    none_rows = df[df["strategy"] == "none"]
    best_none = none_rows.iloc[0] if len(none_rows) else None
    out = pd.DataFrame({
        "Model": df["model"].map(MODEL_LABELS),
        "Imbalance strategy": df["strategy"].map(STRATEGY_LABELS),
        "Macro-F1": df["macro_f1"].map(_num),
        "Rare-class F1": df["rare_macro_f1"].map(_num),
        "Macro recall": df["macro_recall"].map(_num),
        "MCC": df["mcc"].map(_num),
        "PR-AUC": df["macro_pr_auc"].map(_num),
        "Detection": df["detection_rate"].map(_pct),
        "False alarms": df["false_alarm_rate"].map(lambda v: _pct(v, 2)),
        "Fit (s)": df["fit_seconds"].map(lambda v: f"{v:.0f}"),
    })
    lines = [
        "## 2. Benchmark: models × imbalance strategies",
        "",
        f"Every combination was trained on the training split and scored on the **validation** split "
        f"(selection metric: {METRIC_LABELS.get(bench['selection_metric'], bench['selection_metric'])}). "
        "The test split is not touched until section 3.",
        "",
    ]
    if best_none is not None:
        top = df.iloc[0]
        lines += [
            f"**Takeaway:** the best configuration is **{MODEL_LABELS[best['model']]} + "
            f"{STRATEGY_LABELS[best['strategy']]}** (macro-F1 {_num(top['macro_f1'])}, rare-class F1 "
            f"{_num(top['rare_macro_f1'])}). The best model trained on the raw class distribution "
            f"({MODEL_LABELS[best_none['model']]}) reaches macro-F1 {_num(best_none['macro_f1'])} and rare-class F1 "
            f"{_num(best_none['rare_macro_f1'])}, so imbalance handling is worth "
            f"**{top['macro_f1'] - best_none['macro_f1']:+.3f} macro-F1** and "
            f"**{(top['rare_macro_f1'] or 0) - (best_none['rare_macro_f1'] or 0):+.3f} rare-class F1**.",
            "",
        ]
    lines += [
        _img("benchmark_macro_f1.png", "Benchmark macro-F1 heatmap", figures), "",
        _img("benchmark_rare_f1.png", "Benchmark rare-class F1 heatmap", figures), "",
        _table(out, index=False), "",
    ]
    failed = [r for r in bench["rows"] if "error" in r]
    if failed:
        lines += ["Failed configurations: " + ", ".join(f"{r['model']} + {r['strategy']}" for r in failed), ""]
    return lines


def _final_section(final: dict, figures: Path, alerts_path: Path) -> list[str]:
    m = final["metrics"]
    label = MODEL_LABELS[final["model"]]
    head = pd.DataFrame({
        f"{label}, no imbalance handling": {k: m["baseline"].get(k) for k in METRIC_LABELS},
        f"{label} + {STRATEGY_LABELS[final['strategy']]}": {k: m["classifier"].get(k) for k in METRIC_LABELS},
        "+ novelty detector (deployed)": {k: m["hybrid"].get(k) for k in METRIC_LABELS},
    })
    head.index = [METRIC_LABELS[k] for k in head.index]
    pct_rows = {METRIC_LABELS["detection_rate"], METRIC_LABELS["false_alarm_rate"]}
    head = head.apply(lambda col: [(_pct(v, 2) if idx in pct_rows else _num(v)) for idx, v in zip(head.index, col)])
    head.index.name = "Metric (test set)"

    pc_final = pd.DataFrame(m["classifier"]["per_class"]).T
    pc_base = pd.DataFrame(m["baseline"]["per_class"]).T
    per_class = pd.DataFrame({
        "Test flows": pc_final["support"].astype(int).map("{:,}".format),
        "Precision": pc_final["precision"].map(_num),
        "Recall": pc_final["recall"].map(_num),
        "F1": pc_final["f1"].map(_num),
        "PR-AUC": pc_final["pr_auc"].map(_num),
        "F1 without imbalance handling": pc_base["f1"].reindex(pc_final.index).map(_num),
        "Δ F1": (pc_final["f1"] - pc_base["f1"].reindex(pc_final.index)).map(lambda v: f"{v:+.3f}"),
    })
    per_class.index.name = "Class"
    throughput = 1e6 / final["predict_ms_per_1k_flows"] if final["predict_ms_per_1k_flows"] else None

    lines = [
        "## 3. Final model on the held-out test set",
        "",
        f"Deployed configuration: **{label} + {STRATEGY_LABELS[final['strategy']]}** "
        f"({final['selected_because']}), combined with a "
        f"{'Local Outlier Factor' if final.get('novelty_method', 'lof') == 'lof' else 'Isolation Forest'} "
        "novelty detector trained on benign traffic "
        f"(false-alarm budget {_pct(final['benign_fpr_budget'])} of benign traffic).",
        "",
        _table(head), "",
        "Per-class results (classifier, test set):", "",
        _table(per_class), "",
        _img("per_class_f1.png", "Per-class F1 before/after", figures), "",
        _img("confusion_matrix.png", "Confusion matrix", figures), "",
        _img("pr_curves.png", "Precision-recall curves", figures), "",
        f"Training time {final['fit_seconds']:.0f} s. Inference (classifier + novelty detector): "
        f"{final['predict_ms_per_1k_flows']:.1f} ms per 1,000 flows"
        + (f" ≈ {throughput:,.0f} flows/s on one machine." if throughput else "."),
        "",
        "## 4. Explainability",
        "",
        "Global: permutation importance, i.e. how much validation macro-F1 drops when one feature is shuffled.",
        "",
        _img("feature_importance.png", "Feature importance", figures), "",
        "Local: every alert carries the features that deviate most from the benign baseline. "
        "Top of the test-set alert queue:",
        "",
    ]
    if alerts_path.exists():
        alerts = pd.read_csv(alerts_path, nrows=2000)
        sample = (alerts.groupby("predicted_class", sort=False).head(1).head(6)
                  [["alert_id", "priority", "predicted_class", "confidence", "explanation"]])
        sample["explanation"] = sample["explanation"].str.replace("|", "/", regex=False)
        lines += [_table(sample, index=False), ""]
    a = final["alerts"]
    lines += [
        "## 5. Alert triage output",
        "",
        f"The test split produced **{a['total']:,} flow-level alerts** ("
        + ", ".join(f"{k}: {v:,}" for k, v in a["by_priority"].items())
        + f"). {a['needs_review']:,} are flagged for manual review (low confidence or unknown family), "
        f"including {a['unknown_anomaly']:,} `{UNKNOWN}` alerts raised by the novelty detector.",
        "",
    ]
    if a.get("incidents"):
        lines += [
            f"Grouped by attack family, shared endpoint and time, they collapse into **{a['incidents']:,} incidents** "
            f"({', '.join(f'{k}: {v:,}' for k, v in a['incidents_by_priority'].items())}), i.e. "
            f"**{a['total'] / a['incidents']:.0f}× fewer items** for an analyst to open. Largest incidents:",
            "",
            _table(pd.DataFrame(a["largest_incidents"]).rename(columns={
                "incident_id": "Incident", "priority": "Priority", "predicted_class": "Family",
                "flows": "Flows", "shared_endpoint": "Shared endpoint"}), index=False),
            "",
        ]
    lines += ["Open them in the analyst console: `streamlit run app/streamlit_app.py`.", ""]
    return lines


def _novelty_section(nov: dict, figures: Path) -> list[str]:
    df = pd.DataFrame(nov["rows"])
    if df.empty:
        return []
    table = pd.DataFrame({
        "Held-out family": df["held_out_family"],
        "Test flows": df["test_flows"].map("{:,}".format),
        "Classifier only": df["classifier_detection"].map(_pct),
        "Classifier + novelty detector": df["hybrid_detection"].map(_pct),
        "Caught only by novelty detector": df["caught_only_by_anomaly_detector"].map(_pct),
        "Label the classifier gave": df["most_common_label_given"],
        "False alarms (hybrid)": df["hybrid_false_alarm_rate"].map(lambda v: _pct(v, 2)),
    })
    return [
        "## 6. Novel attacks: leave-one-attack-out",
        "",
        f"To simulate a zero-day, each attack family in turn is removed from training "
        f"({MODEL_LABELS[nov['model']]} + {STRATEGY_LABELS[nov['strategy']]}) and the model is asked to "
        "raise an alert on that family's test flows. Any non-benign label counts as a detection, "
        "because the analyst is alerted even when the family name is wrong.",
        "",
        f"Mean detection of unseen families: **{_pct(nov['mean_classifier_detection'])} classifier only → "
        f"{_pct(nov['mean_hybrid_detection'])} with the novelty detector**, at the cost of "
        f"{_pct(nov['anomaly_detector_false_alarm_rate'], 2)} extra false alarms on benign traffic.",
        "",
        _img("novel_attack_detection.png", "Novel attack detection", figures), "",
        _table(table, index=False), "",
    ]


def _drift_section(drift: dict, figures: Path) -> list[str]:
    if drift.get("skipped"):
        return ["## 7. Drift over time", "", f"Skipped: {drift['skipped']}.", ""]
    m = drift["metrics"]
    keys = ["macro_f1", "rare_macro_f1", "detection_rate", "false_alarm_rate"]
    table = pd.DataFrame({
        "History (validation)": [m["past_validation"].get(k) for k in keys],
        "Recent traffic, static model": [m["future_static"].get(k) for k in keys],
        "Recent traffic, retrained": [m["future_retrained"].get(k) for k in keys],
    }, index=[METRIC_LABELS[k] for k in keys])
    table = table.apply(lambda col: [(_pct(v, 2) if "rate" in idx else _num(v)) for idx, v in zip(table.index, col)])
    table.index.name = "Metric (hybrid detector)"
    tr = drift["time_ranges"]
    sz = drift["sizes"]
    sig = drift["signals"]
    lines = [
        "## 7. Drift over time",
        "",
        f"Train on history ({str(tr['past'][0])[:10]} → {str(tr['past'][1])[:10]}), evaluate on recent traffic "
        f"({str(tr['future'][0])[:10]} → {str(tr['future'][1])[:10]}). Then retrain with "
        f"{sz['future_labelled']:,} labelled recent flows, i.e. what analysts confirm through the triage UI, "
        f"and re-evaluate on the remaining {sz['future_evaluation']:,}.",
        "",
        _table(table), "",
        "**Label-free drift monitor**: signals a production deployment can compute without ground truth.",
        "",
        _table(pd.DataFrame({
            "Signal": [
                "Max feature PSI, all traffic",
                "Max feature PSI, alerted traffic",
                "PSI of the predicted-class mix",
                "Unknown-Anomaly rate",
                "Mean confidence of attack predictions",
            ],
            "Value": [
                _num(sig["max_feature_psi_all_traffic"]),
                _num(sig["max_feature_psi_alerted_traffic"]),
                _num(sig["prediction_mix_psi"]),
                f"{_pct(sig['unknown_anomaly_rate']['history'], 2)} → {_pct(sig['unknown_anomaly_rate']['recent'], 2)}",
                f"{_num(sig['mean_alert_confidence']['history'])} → {_num(sig['mean_alert_confidence']['recent'])}",
            ],
            "Alert when": [
                f"> {drift['psi_alert']}",
                f"> {drift['psi_alert']}",
                "> 0.1",
                "doubles vs. history",
                "(context)",
            ],
        }), index=False),
        "",
        f"Triggers fired: {', '.join(k.replace('_', ' ') for k, v in drift['triggers'].items() if v) or 'none'} "
        f"→ retraining **{'triggered' if drift['retraining_triggered'] else 'not needed'}**. "
        f"Drifted features: {', '.join(f'`{f}`' for f in drift['drifted_features'][:8]) or 'none'}. "
        "Whole-traffic PSI alone can miss this kind of drift: attacks are a small share of all flows, so changes in "
        "them barely move whole-traffic distributions. That is why the monitor watches the alerted population "
        "and the prediction mix.",
        "",
        _img("drift_psi.png", "Feature drift (PSI)", figures), "",
        _img("drift_recall.png", "Per-class recall before/after retraining", figures), "",
    ]
    if drift["new_families"]:
        nf = pd.DataFrame(drift["new_families"])
        lines += [
            "Attack families that only appear in recent traffic:", "",
            _table(pd.DataFrame({
                "Family": nf["family"],
                "Flows": nf["flows"],
                "Static model: alerted": nf["static_hybrid_detection"].map(_pct),
                "…as Unknown-Anomaly": nf["flagged_as_unknown"].map(_pct),
                "Recall after retraining": nf["retrained_recall"].map(_num),
            }), index=False), "",
        ]
    return lines


def _retrain_section(rt: dict, figures: Path) -> list[str]:
    fb, ev = rt["feedback"], rt["evaluation"]
    before, after = rt["metrics"]["before"], rt["metrics"]["after"]
    vb, va = rt["alert_volume"]["before"], rt["alert_volume"]["after"]
    keys = ["macro_f1", "rare_macro_f1", "macro_recall", "detection_rate", "false_alarm_rate"]
    table = pd.DataFrame({
        "Deployed model": [before.get(k) for k in keys] + [vb["incidents"], vb["unknown_alerts"]],
        "Retrained with analyst feedback": [after.get(k) for k in keys] + [va["incidents"], va["unknown_alerts"]],
    }, index=[METRIC_LABELS[k] for k in keys] + ["Incidents in the queue", "Unknown-Anomaly alerts"])
    table = table.apply(lambda col: [
        (_pct(v, 2) if "rate" in idx else (f"{int(v):,}" if idx.startswith(("Incidents", "Unknown")) else _num(v)))
        for idx, v in zip(table.index, col)])
    table.index.name = f"Unreviewed test flows (n = {ev['evaluated_flows']:,})"
    who = ("a **simulated analyst** who reviews incidents flagged *needs review* first and labels every flow "
           "correctly (ground truth stands in for human judgement)" if fb["simulated"] else "analysts in the console")
    labels = ", ".join(f"{k}: {v:,}" for k, v in sorted(fb["by_label"].items(), key=lambda kv: -kv[1]))
    lines = [
        "## 8. Closing the loop: analyst feedback → retraining",
        "",
        f"Decisions made in the triage console are stored with each flow's feature values, so every decision is a "
        f"labelled training example. Here {who} worked through **{fb['incidents']:,} incidents**, producing "
        f"**{fb['usable']:,} labelled flows** ({labels}). `nids retrain` added them to the original training data and "
        f"refit the classifier and the novelty detector ({rt['fit_seconds']:.0f} s).",
        "",
        f"The {ev['excluded_reviewed_flows']:,} reviewed flows are removed from the test set before comparing, because "
        "they are training data now:",
        "",
        _table(table), "",
        f"Net effect: rare-class F1 {(after.get('rare_macro_f1') or 0) - (before.get('rare_macro_f1') or 0):+.3f}, "
        f"macro-F1 {after['macro_f1'] - before['macro_f1']:+.3f}, false-alarm rate "
        f"{100 * (after['false_alarm_rate'] - before['false_alarm_rate']):+.2f} pp.",
        "",
        "How to read this: removing the reviewed flows also removes the deployed model's reviewed hits and false "
        "alarms. What remains is the traffic analysts did not look at, including the attacks the deployed model "
        "missed entirely. The comparison measures what retraining changes there, not on flows it was taught.",
        "",
        _img("retrain_per_class_f1.png", "Per-class F1 before/after feedback retraining", figures), "",
    ]
    if rt["new_families"]:
        lines += [f"New families learned from analyst labels: {', '.join(rt['new_families'])}.", ""]
    return lines


LIMITATIONS = [
    "## 9. Limitations and next steps",
    "",
    "- **Flow features only.** Payload-level attacks that leave flow statistics untouched are invisible; "
    "pairing NetSentinel with signature IDS (Suricata/Zeek) covers that gap.",
    "- **Random splits overestimate.** Stratified random splits put flows from the same attack session in train "
    "and test. The temporal experiment (section 7) is the more realistic estimate; use it when timestamps exist.",
    "- **SMOTE creates plausible but synthetic flows** in feature space; it never touches validation/test data, "
    "but very rare families (tens of flows) remain hard to learn and should be reported with that caveat.",
    "- **Anomaly threshold is a budget.** A 1% benign false-alarm budget is cheap at the flow level but can be "
    "noisy at SOC scale; tune `novelty.benign_fpr` to the team's alert capacity.",
    "- Next: adversarial robustness checks (feature-space evasion), streaming deployment (Kafka/Zeek logs → "
    "`nids score`), active learning that routes the least-confident alerts to analysts first.",
    "",
]


def write_report(cfg: dict) -> Path:
    dirs = output_dirs(cfg)
    results, figures = dirs["results"], dirs["figures"]
    bench = load_json(results / "benchmark.json")
    final = load_json(results / "final.json")
    nov = load_json(results / "novelty.json")
    drift = load_json(results / "drift.json")
    if not (bench or final):
        raise FileNotFoundError("No results yet: run `python -m nids all` (or benchmark/train) first.")

    dataset = (final or bench)["dataset"]
    generated = (final or bench)["generated_at"][:16].replace("T", " ")
    lines = ["# NetSentinel: evaluation report", ""]
    lines += [f"_Generated {generated} UTC from `{dataset['source']}`._", ""]
    if "synthetic" in dataset["source"]:
        lines += [
            "> **Note:** these numbers come from the bundled *synthetic* dataset, which exists so the pipeline can run "
            "end-to-end before the official data is available. They demonstrate the method, not real-world "
            "performance. Re-run `python -m nids all --data <official dataset>` to regenerate this report.",
            "",
        ]
    # Figures that depend only on the results JSON are redrawn here, so they always match the text.
    lines += _dataset_section(dataset, figures)
    if bench:
        plot_benchmark(bench, figures)
        lines += _benchmark_section(bench, figures)
    if final:
        plot_final(final, figures)
        lines += _final_section(final, figures, Path(cfg["paths"]["artifacts"]) / ALERTS_FILE)
    if nov:
        plot_novelty(nov, figures)
        lines += _novelty_section(nov, figures)
    if drift:
        plot_drift(drift, figures)
        lines += _drift_section(drift, figures)
    retrain = load_json(results / "retrain.json")
    if retrain:
        plot_retrain(retrain, figures)
        lines += _retrain_section(retrain, figures)
    lines += LIMITATIONS
    path = dirs["reports"] / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", path)
    return path
