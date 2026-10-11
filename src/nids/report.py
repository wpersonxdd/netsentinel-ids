"""Builds reports/REPORT.md (the evaluation deliverable) from the JSON results of each experiment."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from nids import BENIGN, UNKNOWN
from nids.config import output_dirs
from nids.experiments import (ALERTS_FILE, load_json, plot_audit, plot_benchmark, plot_drift, plot_final,
                              plot_novelty, plot_retrain)
from nids.evaluate import BOOTSTRAP_METRICS
from nids.models import MODEL_LABELS, STRATEGY_LABELS

log = logging.getLogger(__name__)

METRIC_LABELS = {
    "macro_f1": "Macro-F1",
    "supported_macro_f1": "Macro-F1, families with ≥ 10 test flows",
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
        f"- {ds['rows']:,} flows {'used' if ds.get('sampling') else 'after cleaning'}; {ds['features']} features "
        f"({ds['numeric_features']} numeric, {ds['categorical_features']} categorical)",
        f"- Imbalance ratio (largest / smallest class): **{ds['imbalance_ratio']:,.0f} : 1**",
        f"- Exact duplicate flows removed: {ds['dropped_duplicates']:,}"
        + (f"; classes dropped as too rare to learn: {ds['dropped_rare_classes']}" if ds["dropped_rare_classes"] else ""),
    ]
    if ds.get("sampling"):
        sm = ds["sampling"]
        lines.append(
            f"- **Stratified sample** of {ds['rows']:,} of the {sm['rows_before']:,} cleaned flows. Families with fewer "
            f"than {sm['min_per_class']:,} flows are kept whole, so they are over-represented relative to benign traffic "
            "compared with the full capture: their test *precision* is somewhat optimistic, while recall is unaffected.")
    if ds.get("split"):
        s = ds["split"]
        lines.append(f"- Stratified split: {s['train']:,} train / {s['validation']:,} validation / {s['test']:,} test")
    lines += ["", _table(table), "", _img("class_distribution.png", "Class distribution", figures), ""]
    return lines


def _benchmark_section(bench: dict, figures: Path) -> list[str]:
    rows = [r for r in bench["rows"] if "error" not in r]
    df = pd.DataFrame(rows)
    df["score"] = df["selection_score"] if "selection_score" in df else df["macro_f1"]
    df = df.sort_values(["score", "macro_f1"], ascending=False)
    best = bench["best"]
    out = pd.DataFrame({
        "Model": df["model"].map(MODEL_LABELS),
        "Imbalance strategy": df["strategy"].map(STRATEGY_LABELS),
        "Selection score": df["score"].map(_num),
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
        "Every combination was trained on the training split and scored on the **validation** split. "
        "The test split is not touched until section 3.",
        "",
    ]
    if bench.get("selection_classes"):
        support = bench.get("val_support", {})
        excluded = bench.get("excluded_from_selection", [])
        note = (f"**Selection score** = macro-F1 over the {len(bench['selection_classes'])} families with at least "
                f"{bench['min_val_support']} validation flows.")
        if excluded:
            listed = ", ".join(f"{c} ({support.get(c, 0)} validation flow{'' if support.get(c, 0) == 1 else 's'})"
                               for c in excluded)
            note += (f" Excluded from *selection* (still evaluated everywhere): {listed}. With so few flows a "
                     "family's F1 is 0 or 1 by the luck of a single flow, which would move a "
                     f"{len(support)}-family macro-F1 by up to {1 / max(len(support), 1):.2f}, more than the real gap "
                     "between good configurations.")
        lines += [note, ""]
    handled = df[df["strategy"] != "none"]
    natural = df[df["strategy"] == "none"]
    if len(handled) and len(natural):
        h, n = handled.iloc[0], natural.iloc[0]
        winner = "an imbalance strategy" if best["strategy"] != "none" else "the natural class distribution"
        lines += [
            f"**Takeaway:** the winner is **{MODEL_LABELS[best['model']]} + {STRATEGY_LABELS[best['strategy']]}**, "
            f"i.e. {winner}. Best with imbalance handling: {MODEL_LABELS[h['model']]} + "
            f"{STRATEGY_LABELS[h['strategy']]} (score {_num(h['score'])}, rare-class F1 {_num(h['rare_macro_f1'])}); "
            f"best on the natural distribution: {MODEL_LABELS[n['model']]} (score {_num(n['score'])}, rare-class F1 "
            f"{_num(n['rare_macro_f1'])}). Difference: **{h['score'] - n['score']:+.3f}** selection score.",
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


PCT_METRICS = {"detection_rate", "false_alarm_rate"}


def _ci(entry: dict | None, metric: str) -> str:
    """'0.835 [0.822, 0.848]' (or percentages for rate metrics)."""
    if not entry:
        return "–"
    if metric in PCT_METRICS:
        return f"{100 * entry['point']:.2f}% [{100 * entry['low']:.2f}, {100 * entry['high']:.2f}]"
    return f"{entry['point']:.3f} [{entry['low']:.3f}, {entry['high']:.3f}]"


def _delta(entry: dict, metric: str) -> str:
    if metric in PCT_METRICS:
        return f"{100 * entry['delta']:+.2f} pp [{100 * entry['low']:+.2f}, {100 * entry['high']:+.2f}]"
    return f"{entry['delta']:+.3f} [{entry['low']:+.3f}, {entry['high']:+.3f}]"


def _audit_section(audit: dict, figures: Path) -> list[str]:
    scan = pd.DataFrame(audit["shortcut_scan"])
    top = scan.head(8)
    table = pd.DataFrame({
        "Feature": top["feature"].map(lambda f: f"`{f}`"),
        "Macro-F1 from this feature alone": top["single_feature_macro_f1"].map(_num),
        "Share of full model": top["share_of_full_model"].map(lambda v: f"{v:.0%}")
        if "share_of_full_model" in top else "–",
        "Family it isolates best (F1)": [f"{c} ({f:.2f})" for c, f in zip(top["best_isolated_class"], top["best_class_f1"])],
        "Flag": top["flag"].replace("", "–"),
    })
    missing = audit["missing_rates"]
    suspects = audit["identifier_suspects"]
    flagged = ", ".join(f"`{f}`" for f in audit["flagged"]) or "none"
    if "share_of_full_model" in scan and (scan["share_of_full_model"] >= 0.8).any():
        verdict = "At least one feature alone nearly matches the full model: review it before trusting the results."
    else:
        verdict = ("No single feature comes close to the full model, so the classifier is not riding on one "
                   "shortcut. Flagged features isolate one family each; check that this reflects attack behaviour.")
    return [
        "### Data audit",
        "",
        f"- Missing values: {', '.join(f'`{k}` {v:.2%}' for k, v in missing.items()) or 'none'} "
        "(imputed with the training median inside the pipeline).",
        f"- Identifier- or clock-like numeric columns: "
        f"{'; '.join(f'`{k}`: {v}' for k, v in suspects.items()) or 'none found'}.",
        "- **Shortcut scan.** A depth-4 tree is trained on each feature *alone*. A feature that nearly matches the full "
        "model, or perfectly isolates one family, is flagged for a human to judge: real attack behaviour (a scan "
        "*is* many destination ports) or a capture artefact (a lab IP, TTL or window size).",
        "",
        _img("shortcut_scan.png", "Shortcut scan", figures), "",
        _table(table, index=False), "",
        f"Flagged (blue in the chart): {flagged}. {verdict}",
        "",
    ]


def _strategy_text(strategy: str) -> str:
    return "no imbalance handling" if strategy == "none" else STRATEGY_LABELS[strategy]


def _verdict(d: dict | None) -> str:
    if d is None:
        return "n/a"
    return "significant" if d["low"] > 0 or d["high"] < 0 else "not significant: CI includes 0"


def _tiny_section(final: dict) -> list[str]:
    m = final["metrics"]
    tiny = m["classifier"].get("tiny_classes") or {}
    if not tiny:
        return []
    base_tiny = m["baseline"].get("tiny_classes") or {}
    compare = final.get("comparison_strategy", "none")
    parts = [f"**{c}**: {v['found']} of {v['support']} found (vs. {base_tiny.get(c, {}).get('found', '?')} of "
             f"{v['support']} with {_strategy_text(compare)})" for c, v in tiny.items()]
    return [
        f"**Families with fewer than 10 test flows**: {'; '.join(parts)}. Their F1 is decided by one or two flows, "
        "and a bootstrap cannot express that uncertainty (resampling two correctly found flows always gives F1 = 1), so "
        "they are reported as counts. They dominate the all-family macro-F1, which is why the row "
        "*Macro-F1, families with ≥ 10 test flows* is the measurable comparison.",
        "",
    ]


def _gain_section(final: dict) -> list[str]:
    pairs = final.get("bootstrap", {}).get("pairs", {})
    if "baseline->classifier" not in pairs:
        return []
    boot = final["bootstrap"]
    compare = final.get("comparison_strategy", "none")
    first = ("Imbalance handling (same model)" if compare == "none"
             else f"Natural distribution vs. {STRATEGY_LABELS[compare]} (same model)")
    rows = []
    for key, label in [("baseline->classifier", first), ("classifier->hybrid", "Adding the novelty detector")]:
        shown = BOOTSTRAP_METRICS if key == "baseline->classifier" else ("detection_rate", "false_alarm_rate")
        for m, d in pairs.get(key, {}).items():
            if m not in shown:
                continue
            rows.append({"Change": label, "Metric": METRIC_LABELS[m], "Δ [95% CI]": _delta(d, m),
                         "Resamples where it was not better": f"{d['p_not_better']:.1%}"})
    main = pairs["baseline->classifier"]
    macro, rare = main["macro_f1"], main.get("rare_macro_f1")
    sup = main.get("supported_macro_f1") if final["metrics"]["classifier"].get("tiny_classes") else None
    if compare == "none" and sup:
        summary = (f"Imbalance handling: all-family macro-F1 {_delta(macro, 'macro_f1')}, rare-class F1 "
                   f"{_delta(rare, 'rare_macro_f1') if rare else 'n/a'}. These intervals look decisive but cannot "
                   "reflect the uncertainty of the tiny families that drive them (see above). The measurable claim is "
                   f"the change on families with at least 10 test flows: {_delta(sup, 'supported_macro_f1')} "
                   f"({_verdict(sup)}).")
    elif compare == "none":
        summary = (f"Imbalance handling: macro-F1 {_delta(macro, 'macro_f1')}, rare-class F1 "
                   f"{_delta(rare, 'rare_macro_f1') if rare else 'n/a'}. "
                   + ("Both intervals exclude zero: the gain is not a fluke of this test split."
                      if macro["low"] > 0 and (rare or {"low": 1})["low"] > 0
                      else "At least one interval includes zero: treat that part of the gain as uncertain."))
    else:
        summary = (f"Here the natural class distribution won the benchmark. Compared with the same model trained with "
                   f"{STRATEGY_LABELS[compare]}, it changes macro-F1 by {_delta(macro, 'macro_f1')} ({_verdict(macro)}) "
                   f"and rare-class F1 by {_delta(rare, 'rare_macro_f1') if rare else 'n/a'} ({_verdict(rare)}).")
    return [
        "### 3.1 Is the gain real? Paired bootstrap",
        "",
        f"Both models are scored on the same {boot['n_boot']:,} stratified bootstrap resamples of the test set (every "
        "resample keeps each family's size, so rare families never drop out). The CI of the *difference* says whether "
        "the improvement could be luck.",
        "",
        _table(pd.DataFrame(rows), index=False), "",
        summary,
        "",
    ]


def _protocols_section(protocols: dict, synthetic: bool = False) -> list[str]:
    p = protocols["protocols"]
    names = {"random": "Stratified random split", "session": "Session-grouped split", "temporal": "Temporal split"}
    table = pd.DataFrame([{
        "Protocol": names.get(k, k),
        "Train / test flows": f"{v['train_flows']:,} / {v['test_flows']:,}",
        "Macro-F1 [95% CI]": _ci(v["ci"].get("macro_f1"), "macro_f1"),
        "Rare-class F1 [95% CI]": _ci(v["ci"].get("rare_macro_f1"), "rare_macro_f1"),
        "False alarms": _pct(v["false_alarm_rate"], 2),
        "Families unseen in training": ", ".join(v["unseen_in_training"]) or "–",
        "How it splits": v["note"],
    } for k, v in p.items()])
    lines = [
        "### 3.2 How optimistic is the headline? Three evaluation protocols",
        "",
        f"The same configuration ({MODEL_LABELS[protocols['model']]} + {STRATEGY_LABELS[protocols['strategy']]}, "
        "classifier only) trained and tested under three protocols. Random splits are the field's default but let "
        "flows of one attack session sit on both sides; grouped and temporal splits remove that advantage.",
        "",
        _table(table, index=False), "",
    ]
    if "random" in p:
        gaps = [f"{names[k].lower()} {p[k]['macro_f1'] - p['random']['macro_f1']:+.3f}" for k in ("session", "temporal")
                if k in p]
        if gaps:
            lines += [f"Macro-F1 change vs. the random split: {'; '.join(gaps)}.", ""]
    if "session" in p and "random" in p:
        if p["session"]["macro_f1"] >= p["random"]["ci"]["macro_f1"]["low"]:
            lines += [
                "- **Session leakage: none detected.** Keeping whole sessions on one side does not lower the score, "
                "so the random-split number is not inflated by near-duplicate flows of the same session."
                + (" This is expected here: the synthetic generator draws flows independently. On real captures, "
                   "where flows of one session are correlated, this check is what exposes leakage."
                   if synthetic else ""), ""]
        else:
            lines += [
                "- **Session leakage detected.** Holding out whole sessions lowers the score beyond the random-split "
                "confidence interval: part of the headline comes from flows of the same session on both sides. Quote "
                "the session-grouped number as the realistic estimate.", ""]
    if "temporal" in p:
        lines += [
            "- **Temporal split** is the honest estimate for deployment: tomorrow's traffic is never in the training "
            "set" + ((f", and {', '.join(unseen)} never appear{'s' if len(unseen) == 1 else ''} before the cut, so "
                      f"no classifier can name {'it' if len(unseen) == 1 else 'them'}; that is the novelty detector's "
                      "job (§6).") if (unseen := p["temporal"]["unseen_in_training"]) else "."), ""]
    return lines


def _conformal_section(final: dict, figures: Path) -> list[str]:
    c = final.get("conformal")
    if not c:
        return []
    pc = pd.DataFrame(c["per_class"]).T
    calib = pd.Series(c["calibration_flows"])
    thresholds = pd.Series(c["thresholds"])
    ranges = c.get("expected_coverage_range", {})

    guaranteed = c.get("guaranteed", {})

    def expected(cls: str, cov: float) -> str:
        if guaranteed and not guaranteed.get(cls, True):
            return "no guarantee possible"
        if cls not in ranges:
            return "–"
        lo, hi = ranges[cls]
        mark = "✓" if lo - 0.03 <= cov else "✗"  # small allowance for test-set sampling noise
        return f"{lo:.0%}–{hi:.0%} {mark}"

    table = pd.DataFrame({
        "Calibration flows": calib.reindex(pc.index).fillna(0).astype(int).map("{:,}".format),
        "In the set when p ≥": [
            "top prediction only" if guaranteed and not guaranteed.get(cls, True)
            else (f"{max(1 - t, 0):.4f}" if 1 - t > 0.995 else f"{max(1 - t, 0):.3f}")
            for cls, t in zip(pc.index, thresholds.reindex(pc.index))],
        f"Coverage (target {c['target_coverage']:.0%})": pc["coverage"].map(lambda v: f"{v:.1%}"),
        "Expected range (95%)": [expected(cls, cov) for cls, cov in zip(pc.index, pc["coverage"])],
        "Mean set size": pc["mean_set_size"].map(lambda v: f"{v:.2f}"),
        "Single-family sets": pc["singleton_rate"].map(lambda v: f"{v:.0%}"),
        "Alerted under guaranteed-recall alerting": pc["alerted"].map(lambda v: f"{v:.1%}"),
    })
    trivial = [k for k, g in guaranteed.items() if not g] if guaranteed else \
        [k for k, t in thresholds.items() if t >= 1.0]
    min_n = next((n for n in range(1, 1000) if -(-(n + 1) * (1 - c["alpha"]) // 1) <= n), None)
    table.index.name = "Family"
    hyb = final["metrics"]["hybrid"]
    clf = final["metrics"]["classifier"]
    attack_calib = calib.drop(labels=[BENIGN], errors="ignore")
    loosest = (1 - thresholds.drop(labels=[BENIGN] + trivial, errors="ignore")).idxmin()
    return [
        "### 3.3 New: per-family guarantees with conformal prediction",
        "",
        "A confidence of 0.7 promises nothing. **Mondrian (class-conditional) conformal prediction** calibrates one "
        f"threshold per family on the validation split and returns a *set* of plausible families per flow, with a "
        f"finite-sample guarantee that the true family is in the set for ≥ {c['target_coverage']:.0%} of flows **of "
        "every family separately**. A marginal guarantee would be dominated by benign traffic and say almost nothing "
        "about a family that is 0.2% of flows.",
        "",
        _table(table), "",
        _img("conformal_coverage.png", "Conformal coverage per family", figures), "",
        f"- Mean set size {c['mean_set_size']:.2f}; {c['singleton_rate']:.0%} of flows get exactly one family. "
        "In the triage console an ambiguous set (more than one family) is a second *needs review* trigger next to low "
        "confidence. The two catch different cases, because thresholds differ per family: a 50/50 flow can still get a "
        "one-family set.",
        "- The guarantee holds on average over calibration draws. With n calibration flows the realised coverage of a "
        "family varies within the *expected range* column (Beta distribution of conformal coverage); ✓ means the "
        "test-set coverage is consistent with the guarantee. Coverage *above* the range is expected: the most probable "
        "family is always added to the set, which can only raise coverage.",
        f"- **The price of a guarantee.** Alerting whenever *any* attack family is in the set (guaranteed-recall "
        f"alerting) would detect {_pct(c['guaranteed_alerting_detection_rate'])} of attacks at "
        f"{_pct(c['guaranteed_alerting_false_alarm_rate'], 2)} false alarms, vs. "
        f"{_pct(hyb['detection_rate'])} at {_pct(hyb['false_alarm_rate'], 2)} for the deployed detector. "
        + ((f"The cost is driven by the families with the fewest calibration flows: {loosest} "
            f"({int(attack_calib.get(loosest, 0))} flows) enters a set whenever p ≥ "
            f"{max(1 - thresholds[loosest], 0):.3f}. **Every analyst-confirmed flow of a rare family tightens its "
            "threshold, so the feedback loop (§8) directly lowers the price of the guarantee.**")
           if c["guaranteed_alerting_false_alarm_rate"] - clf["false_alarm_rate"] > 0.01 else
           "Here the guarantee is nearly free: the classifier is confident enough that a second family rarely enters "
           "a set."),
    ] + ([f"- **No guarantee is possible** for "
          f"{', '.join(f'{k} ({int(calib.get(k, 0))} calibration flow' + ('' if int(calib.get(k, 0)) == 1 else 's') + ')' for k in trivial)}: "
          f"a {c['target_coverage']:.0%} guarantee needs at least {min_n} calibration flows of the family. The strict "
          "procedure would then put the family into *every* set, which tells an analyst nothing, so these families "
          "appear in a set only as the top prediction, and their coverage is simply their recall. Collecting labelled "
          "flows of them (§8) is what makes a guarantee possible."] if trivial else []) + [""]


def _final_section(final: dict, figures: Path, alerts_path: Path, protocols: dict | None = None) -> list[str]:
    m = final["metrics"]
    label = MODEL_LABELS[final["model"]]
    boot = final.get("bootstrap", {}).get("models", {})

    def column(name: str) -> dict:
        return {k: (_ci(boot[name].get(k), k) if name in boot and k in boot[name]
                    else (_pct(m[name].get(k), 2) if k in PCT_METRICS else _num(m[name].get(k))))
                for k in METRIC_LABELS}

    compare = final.get("comparison_strategy", "none")
    head = pd.DataFrame({
        f"{label}, {_strategy_text(compare)}": column("baseline"),
        f"{label}, {_strategy_text(final['strategy'])}": column("classifier"),
        "+ novelty detector (deployed)": column("hybrid"),
    })
    if not m["classifier"].get("tiny_classes"):  # every family measurable: the extra row would repeat macro-F1
        head = head.drop(index="supported_macro_f1", errors="ignore")
    head.index = [METRIC_LABELS[k] for k in head.index]
    head.index.name = "Metric (test set) [95% CI]" if boot else "Metric (test set)"

    pc_final = pd.DataFrame(m["classifier"]["per_class"]).T
    pc_base = pd.DataFrame(m["baseline"]["per_class"]).T
    per_class = pd.DataFrame({
        "Test flows": pc_final["support"].astype(int).map("{:,}".format),
        "Precision": pc_final["precision"].map(_num),
        "Recall": pc_final["recall"].map(_num),
        "F1": pc_final["f1"].map(_num),
        "PR-AUC": pc_final["pr_auc"].map(_num),
        f"F1, {_strategy_text(compare)}": pc_base["f1"].reindex(pc_final.index).map(_num),
        "Δ F1": (pc_final["f1"] - pc_base["f1"].reindex(pc_final.index)).map(lambda v: f"{v:+.3f}"),
    })
    per_class.index.name = "Class"
    throughput = 1e6 / final["predict_ms_per_1k_flows"] if final["predict_ms_per_1k_flows"] else None

    lines = [
        "## 3. Final model on the held-out test set",
        "",
        f"Deployed configuration: **{label}, {_strategy_text(final['strategy'])}** "
        f"({final['selected_because']}), combined with a "
        f"{'Local Outlier Factor' if final.get('novelty_method', 'lof') == 'lof' else 'Isolation Forest'} "
        "novelty detector trained on benign traffic "
        f"(false-alarm budget {_pct(final['benign_fpr_budget'])} of benign traffic).",
        "",
        _table(head), "",
        *_tiny_section(final),
        *_gain_section(final),
        "Per-class results (classifier, test set):", "",
        _table(per_class), "",
        _img("per_class_f1.png", "Per-class F1 before/after", figures), "",
        _img("confusion_matrix.png", "Confusion matrix", figures), "",
        _img("pr_curves.png", "Precision-recall curves", figures), "",
        f"Training time {final['fit_seconds']:.0f} s. Inference (classifier + novelty detector): "
        f"{final['predict_ms_per_1k_flows']:.1f} ms per 1,000 flows"
        + (f" ≈ {throughput:,.0f} flows/s on one machine." if throughput else "."),
        "",
        *(_protocols_section(protocols, "synthetic" in final["dataset"]["source"]) if protocols else []),
        *_conformal_section(final, figures),
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
        + f"). {a['needs_review']:,} are flagged for manual review (prediction set is not a single family, or the "
        "family is unknown), "
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


def _retrain_effect(rt: dict) -> str:
    pair = rt.get("bootstrap", {}).get("pairs", {}).get("before->after")
    before, after = rt["metrics"]["before"], rt["metrics"]["after"]
    if not pair:
        return (f"Net effect: rare-class F1 {(after.get('rare_macro_f1') or 0) - (before.get('rare_macro_f1') or 0):+.3f}, "
                f"macro-F1 {after['macro_f1'] - before['macro_f1']:+.3f}, false-alarm rate "
                f"{100 * (after['false_alarm_rate'] - before['false_alarm_rate']):+.2f} pp.")
    parts = [f"{METRIC_LABELS[m]} {_delta(pair[m], m)}" for m in ("rare_macro_f1", "macro_f1", "false_alarm_rate")
             if m in pair]
    solid = [METRIC_LABELS[m].split(" (")[0] for m in ("rare_macro_f1", "macro_f1")
             if m in pair and pair[m]["low"] > 0]
    return (f"Net effect with 95% paired-bootstrap CIs: {'; '.join(parts)}. "
            + (f"The improvement in {' and '.join(solid)} holds across resamples."
               if solid else "The intervals include zero: with this little feedback the gain is not yet conclusive."))


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
        _retrain_effect(rt),
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
    audit = load_json(results / "audit.json")
    if audit:
        plot_audit(audit, figures)
        lines += _audit_section(audit, figures)
    if bench:
        plot_benchmark(bench, figures)
        lines += _benchmark_section(bench, figures)
    if final:
        plot_final(final, figures)
        lines += _final_section(final, figures, Path(cfg["paths"]["artifacts"]) / ALERTS_FILE,
                                load_json(results / "protocols.json"))
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
