"""The four experiments behind the report.

1. benchmark - every model x every imbalance strategy, scored on the validation set.
2. train     - the selected configuration, evaluated once on the untouched test set against the same
               model without imbalance handling; saves the deployable bundle and the test alert queue.
3. novelty   - leave-one-attack-out: can the system raise an alert for a family it never saw?
4. drift     - train on the past, test on the future, then retrain with a small labelled sample.
5. retrain   - close the loop: retrain the deployed model with analyst decisions from the console.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from nids import BENIGN, UNKNOWN, __version__, plotting
from nids.bundle import ModelBundle
from nids.config import output_dirs
from nids.data import PreparedData, Split, class_order, load_dataset, split_indices, stratified_subsample
from nids.deep import TORCH_AVAILABLE
from nids.audit import identifier_suspects, quality_table, shortcut_scan
from nids.conformal import MondrianConformal, coverage_range, coverage_report
from nids.evaluate import bootstrap_metrics, evaluate, headline
from nids.feedback import load_feedback_examples
from nids.models import MODEL_LABELS, MODELS, STRATEGY_LABELS, available_models, build_pipeline
from nids.novelty import NoveltyDetector, psi_table
from nids.resampling import STRATEGIES, resampled_size
from nids.triage import BenignBaseline, group_incidents

log = logging.getLogger(__name__)

MODEL_FILE = "netsentinel_model.joblib"
ALERTS_FILE = "alerts_test.csv"
INCIDENTS_FILE = "incidents_test.csv"


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_json(obj: dict, path: Path) -> Path:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    log.info("Wrote %s", path)
    return path


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dataset_summary(data: PreparedData, split: Split | None = None) -> dict:
    counts = data.y.value_counts()
    out = {
        "source": data.source,
        "rows": int(len(data.y)),
        "features": len(data.feature_columns),
        "numeric_features": len(data.numeric),
        "categorical_features": len(data.categorical),
        "label_column": data.label_column,
        "class_counts": {c: int(counts[c]) for c in class_order(counts.index)},
        "imbalance_ratio": float(counts.max() / counts.min()),
        "dropped_duplicates": data.dropped_duplicates,
        "dropped_rare_classes": data.dropped_rare,
        "has_timestamps": data.time is not None,
        "sampling": data.sampling,
    }
    if split is not None:
        out["split"] = {"train": len(split.train), "validation": len(split.val), "test": len(split.test)}
    return out


def _prepare(cfg: dict) -> tuple[PreparedData, Split, dict]:
    dirs = output_dirs(cfg)
    data = load_dataset(cfg["data"], cfg["seed"])
    split = split_indices(data.y, cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["seed"])
    return data, split, dirs


def _parts(data: PreparedData, idx: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    return data.X.iloc[idx], data.y.iloc[idx]


def _trainable(X: pd.DataFrame, y: pd.Series, context: str) -> tuple[pd.DataFrame, pd.Series]:
    """Drop classes with a single training flow: nothing can be learned from them, and stratified internal
    splits (early stopping) fail on them. Happens with small labelled samples, e.g. in the drift experiment."""
    counts = y.value_counts()
    single = counts[counts < 2]
    if len(single):
        log.warning("%s: ignoring classes with a single training flow: %s", context, list(single.index))
        keep = (~y.isin(single.index)).to_numpy()
        return X[keep], y[keep]
    return X, y


def _predict(pipe, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proba = pipe.predict_proba(X)
    classes = np.asarray(pipe.classes_).astype(str)
    return classes[proba.argmax(axis=1)], proba, classes


def choose_final_config(cfg: dict, results_dir: Path) -> tuple[str, str, str]:
    """Model + strategy for the deployable model: from the config, else the benchmark winner."""
    model, strategy = cfg["final_model"]["model"], cfg["final_model"]["strategy"]
    reason = "set in config"
    if "auto" in (model, strategy):
        bench = load_json(results_dir / "benchmark.json")
        if bench and bench.get("best"):
            model = bench["best"]["model"] if model == "auto" else model
            strategy = bench["best"]["strategy"] if strategy == "auto" else strategy
            reason = f"best validation {bench['selection_metric']} in the benchmark"
        else:
            model = ("ensemble" if model == "auto" else model)
            strategy = ("hybrid" if strategy == "auto" else strategy)
            reason = "default choice (run `benchmark` first to select automatically)"
    if model == "deep_mlp" and not TORCH_AVAILABLE:
        log.warning("deep_mlp selected but PyTorch is missing; falling back to hist_gb")
        model = "hist_gb"
    return model, strategy, reason


def _fit_conformal(cfg: dict, pipe, X_cal: pd.DataFrame, y_cal: pd.Series) -> MondrianConformal:
    """Per-class conformal thresholds calibrated on held-out flows the classifier never trained on."""
    return MondrianConformal(cfg["conformal"]["alpha"]).fit(pipe.predict_proba(X_cal), y_cal, pipe.classes_)


def _fit_novelty(cfg: dict, data: PreparedData, X_fit, y_fit, X_cal, y_cal, X_always=None) -> NoveltyDetector:
    nc = cfg["novelty"]
    detector = NoveltyDetector(data.numeric, data.categorical, benign_fpr=nc["benign_fpr"], method=nc["method"],
                               n_neighbors=nc["n_neighbors"], max_fit_rows=nc["max_fit_rows"],
                               n_estimators=nc["n_estimators"], seed=cfg["seed"])
    return detector.fit(X_fit[(y_fit == BENIGN).to_numpy()], X_cal[(y_cal == BENIGN).to_numpy()], X_always)


# --------------------------------------------------------------------------------------------------
# 1. benchmark
# --------------------------------------------------------------------------------------------------


def run_benchmark(cfg: dict) -> dict:
    data, split, dirs = _prepare(cfg)
    X_tr, y_tr = _parts(data, split.train)
    if cfg["benchmark"].get("max_train_rows"):
        pos = stratified_subsample(y_tr, int(cfg["benchmark"]["max_train_rows"]), cfg["seed"])
        X_tr, y_tr = X_tr.iloc[pos], y_tr.iloc[pos]
        log.info("Benchmark training rows capped at %d", len(y_tr))
    X_va, y_va = _parts(data, split.val)

    models = available_models(cfg["benchmark"]["models"])
    strategies = cfg["benchmark"]["strategies"]
    rows = []
    for model in models:
        for strategy in strategies:
            log.info("Benchmark: %s + %s", model, strategy)
            try:
                pipe = build_pipeline(model, strategy, y_tr, data.numeric, data.categorical, cfg)
                t0 = time.perf_counter()
                pipe.fit(X_tr, y_tr)
                fit_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                pred, proba, classes = _predict(pipe, X_va)
                pred_s = time.perf_counter() - t0
            except Exception as exc:  # keep the benchmark going; record the failure
                log.exception("%s + %s failed", model, strategy)
                rows.append({"model": model, "strategy": strategy, "error": str(exc)})
                continue
            m = evaluate(y_va, pred, proba, classes)
            rows.append({
                "model": model,
                "strategy": strategy,
                "fit_seconds": round(fit_s, 1),
                "predict_ms_per_1k_flows": round(pred_s / len(X_va) * 1e6, 2),
                "train_rows_after_resampling": resampled_size(strategy, y_tr, cfg["resampling"]),
                **headline(m),
                "per_class_f1": {c: v["f1"] for c, v in m["per_class"].items()},
                "per_class_recall": {c: v["recall"] for c, v in m["per_class"].items()},
            })
            log.info("  macro-F1 %.3f | rare-class F1 %s | detection %.3f | false alarms %.4f | fit %.0fs",
                     m["macro_f1"], f"{m['rare_macro_f1']:.3f}" if m["rare_macro_f1"] is not None else "n/a",
                     m["detection_rate"], m["false_alarm_rate"], fit_s)

    if not [r for r in rows if "error" not in r]:
        raise RuntimeError("Every benchmark configuration failed; see the log above.")
    result = {
        "generated_at": _now(),
        "version": __version__,
        "dataset": dataset_summary(data, split),
        "evaluated_on": "validation",
        "selection_metric": cfg["benchmark"]["selection_metric"],
        "val_support": {str(k): int(v) for k, v in y_va.value_counts().items()},
        "rows": rows,
    }
    select_best(result, cfg["benchmark"].get("min_val_support", 10))
    best = result["best"]
    metric = result["selection_metric"]
    save_json(result, dirs["results"] / "benchmark.json")
    plot_benchmark(result, dirs["figures"])
    log.info("Best configuration: %s + %s (selection score %.4f on %s)", best["model"], best["strategy"],
             best["selection_score"], ", ".join(result["selection_classes"]))
    return result


def select_best(bench: dict, min_val_support: int = 10) -> dict:
    """Pick the benchmark winner on classes with enough validation flows to be measured.

    With 1-4 validation flows a class's F1 is 0 or 1 by the luck of one flow, and a 9-class macro-F1 then moves by
    ~0.1, more than the real gap between good configurations. Such classes are still evaluated and reported; they
    just do not decide which configuration wins. Adds ``selection_score`` to each row and ``best`` to ``bench``.
    """
    metric = bench["selection_metric"]
    support = bench.get("val_support", {})
    selection = [c for c, n in support.items() if n >= min_val_support] if support else []
    excluded = sorted(c for c in support if c not in selection)
    ok = [r for r in bench["rows"] if "error" not in r]
    for r in ok:
        if metric == "macro_f1" and selection and r.get("per_class_f1"):
            r["selection_score"] = float(np.mean([r["per_class_f1"].get(c, 0.0) for c in selection]))
        else:
            r["selection_score"] = r.get(metric) or 0.0
    best = max(ok, key=lambda r: (r["selection_score"], r.get("rare_macro_recall") or 0.0))
    bench.update(min_val_support=min_val_support, selection_classes=sorted(selection),
                 excluded_from_selection=excluded,
                 best={"model": best["model"], "strategy": best["strategy"], metric: best[metric],
                       "selection_score": best["selection_score"]})
    return bench


def plot_benchmark(bench: dict, figures: Path) -> None:
    """Heatmaps (model x strategy) and the class distribution, drawn from benchmark.json."""
    table = pd.DataFrame([r for r in bench["rows"] if "error" not in r])
    model_order = [m for m in MODELS if m in set(table["model"])]
    strategy_order = [s for s in STRATEGIES if s in set(table["strategy"])]
    for metric_name, fname, title in [
        ("macro_f1", "benchmark_macro_f1.png", "Validation macro-F1: model × imbalance strategy"),
        ("rare_macro_f1", "benchmark_rare_f1.png", "Validation F1 on rare attack families (<1% of traffic)"),
    ]:
        if table[metric_name].notna().any():
            pivot = table.pivot(index="model", columns="strategy", values=metric_name).reindex(
                index=model_order, columns=strategy_order)
            pivot.index = [MODEL_LABELS[m] for m in pivot.index]
            pivot.columns = [STRATEGY_LABELS[s] for s in pivot.columns]
            plotting.heatmap(pivot, figures / fname, title, xlabel="Imbalance strategy")
    counts = pd.Series(bench["dataset"]["class_counts"])
    plotting.class_distribution(counts, figures / "class_distribution.png", "Class distribution after cleaning")


# --------------------------------------------------------------------------------------------------
# 2. final model
# --------------------------------------------------------------------------------------------------


def run_train(cfg: dict) -> dict:
    data, split, dirs = _prepare(cfg)
    model, strategy, reason = choose_final_config(cfg, dirs["results"])
    log.info("Final model: %s + %s (%s)", model, strategy, reason)
    X_tr, y_tr = _parts(data, split.train)
    X_va, y_va = _parts(data, split.val)
    X_te, y_te = _parts(data, split.test)
    ctx_te = data.context.iloc[split.test].reset_index(drop=True)

    pipe = build_pipeline(model, strategy, y_tr, data.numeric, data.categorical, cfg)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    log.info("Fitted in %.0fs; fitting the no-imbalance-handling baseline for comparison", fit_s)
    compare = comparison_strategy(model, strategy, load_json(dirs["results"] / "benchmark.json"))
    log.info("Comparison model: %s + %s", model, compare)
    baseline_pipe = build_pipeline(model, compare, y_tr, data.numeric, data.categorical, cfg).fit(X_tr, y_tr)

    novelty = _fit_novelty(cfg, data, X_tr, y_tr, X_va, y_va)
    bundle = ModelBundle(
        classifier=pipe,
        novelty=novelty,
        baseline=BenignBaseline().fit(X_tr[(y_tr == BENIGN).to_numpy()]),
        numeric=data.numeric,
        categorical=data.categorical,
        severity=cfg["triage"]["severity"],
        conf_threshold=cfg["triage"]["confidence_threshold"],
        label_mode=cfg["data"]["label_mode"],
        conformal=_fit_conformal(cfg, pipe, X_va, y_va),
    )

    t0 = time.perf_counter()
    pred = bundle.predict(X_te)
    pred_s = time.perf_counter() - t0
    m_clf = evaluate(y_te, pred["classifier_label"], pred["proba"], pred["classes"])
    m_hyb = evaluate(y_te, pred["label"], pred["proba"], pred["classes"])
    base_pred, base_proba, base_classes = _predict(baseline_pipe, X_te)
    m_base = evaluate(y_te, base_pred, base_proba, base_classes)
    log.info("Test macro-F1: baseline %.3f -> final %.3f (hybrid %.3f)",
             m_base["macro_f1"], m_clf["macro_f1"], m_hyb["macro_f1"])

    log.info("Bootstrap confidence intervals (%d stratified resamples)", cfg["evaluation"]["bootstrap"])
    boot = bootstrap_metrics(
        y_te, {"baseline": base_pred, "classifier": pred["classifier_label"], "hybrid": pred["label"]},
        pairs=[("baseline", "classifier"), ("classifier", "hybrid")], n_boot=cfg["evaluation"]["bootstrap"],
        seed=cfg["seed"])
    conformal = coverage_report(pred["prediction_set"], y_te, pred["classes"], BENIGN)
    cp = bundle.conformal
    conformal.update(alpha=cp.alpha, target_coverage=1 - cp.alpha,
                     thresholds=dict(zip(cp.classes_, cp.thresholds_.round(4))),
                     calibration_flows=dict(zip(cp.classes_, cp.n_calibration_.tolist())),
                     guaranteed=dict(zip(cp.classes_, cp.guaranteed_.tolist())),
                     expected_coverage_range={c: coverage_range(int(n), cp.alpha)
                                              for c, n in zip(cp.classes_, cp.n_calibration_)})
    log.info("Conformal (alpha %.2f): mean set size %.2f, singleton rate %.1f%%", bundle.conformal.alpha,
             conformal["mean_set_size"], 100 * conformal["singleton_rate"])

    log.info("Permutation feature importance on the validation set")
    pos = stratified_subsample(y_va, cfg["explain"]["permutation_rows"], cfg["seed"], min_per_class=50)
    imp = permutation_importance(pipe, X_va.iloc[pos], y_va.iloc[pos], scoring="f1_macro",
                                 n_repeats=cfg["explain"]["permutation_repeats"], random_state=cfg["seed"])
    importance = pd.DataFrame({"importance": imp.importances_mean, "std": imp.importances_std},
                              index=X_va.columns).sort_values("importance", ascending=False)

    alerts = bundle.alerts(X_te, ctx_te, y_te.reset_index(drop=True), pred)
    incidents, alerts = group_incidents(alerts)
    alerts_path = dirs["artifacts"] / ALERTS_FILE
    alerts.to_csv(alerts_path, index=False)
    incidents.to_csv(dirs["artifacts"] / INCIDENTS_FILE, index=False)

    bundle.metadata = {
        "version": __version__,
        "trained_at": _now(),
        "model": model,
        "strategy": strategy,
        "dataset": data.source,
        "train_rows": int(len(y_tr)),
        "classes": bundle.classes,
        "test_macro_f1": m_clf["macro_f1"],
        "test_rare_macro_f1": m_clf["rare_macro_f1"],
        "time_parsing": {"dayfirst": cfg["data"].get("time_dayfirst", False),
                         "add_12h_before_hour": cfg["data"].get("time_add_12h_before_hour")},
    }
    model_path = bundle.save(dirs["artifacts"] / MODEL_FILE)
    log.info("Saved model bundle to %s", model_path)

    fig = dirs["figures"]
    plotting.confusion(y_te, pred["label"], class_order(y_te.unique()), fig / "confusion_matrix.png",
                       "Test set: hybrid detector, row-normalised")
    plotting.pr_curves(y_te, pred["proba"], pred["classes"], fig / "pr_curves.png", BENIGN)
    plotting.importance(importance, fig / "feature_importance.png")

    result = {
        "generated_at": _now(),
        "version": __version__,
        "model": model,
        "strategy": strategy,
        "selected_because": reason,
        "comparison_strategy": compare,
        "dataset": dataset_summary(data, split),
        "fit_seconds": round(fit_s, 1),
        "predict_ms_per_1k_flows": round(pred_s / len(X_te) * 1e6, 2),
        "novelty_method": novelty.method,
        "novelty_threshold": novelty.threshold_,
        "benign_fpr_budget": cfg["novelty"]["benign_fpr"],
        "metrics": {"baseline": m_base, "classifier": m_clf, "hybrid": m_hyb},
        "bootstrap": boot,
        "conformal": conformal,
        "feature_importance": importance.head(20).round(5).to_dict("index"),
        "alerts": {
            "total": int(len(alerts)),
            "by_priority": alerts["priority"].value_counts().sort_index().to_dict(),
            "by_class": alerts["predicted_class"].value_counts().to_dict(),
            "needs_review": int(alerts["needs_review"].sum()),
            "unknown_anomaly": int((alerts["predicted_class"] == UNKNOWN).sum()),
            "incidents": int(len(incidents)),
            "incidents_by_priority": incidents["priority"].value_counts().sort_index().to_dict() if len(incidents) else {},
            "largest_incidents": incidents.head(5)[["incident_id", "priority", "predicted_class", "flows",
                                                    "shared_endpoint"]].to_dict("records") if len(incidents) else [],
        },
        "artifacts": {"model": str(model_path), "alerts": str(alerts_path)},
    }
    save_json(result, dirs["results"] / "final.json")
    plot_final(result, fig)
    return result


def strategy_text(strategy: str) -> str:
    return "no imbalance handling" if strategy == "none" else STRATEGY_LABELS[strategy]


def comparison_strategy(model: str, strategy: str, bench: dict | None) -> str:
    """What the deployed model is compared against on the test set.

    Normally the same model *without* imbalance handling. If the natural distribution itself won the benchmark, the
    comparison flips to the same model with its best imbalance strategy, so the report still answers "what does
    imbalance handling do here?" instead of comparing a model with itself.
    """
    if strategy != "none":
        return "none"
    rows = [r for r in (bench or {}).get("rows", [])
            if r.get("model") == model and r.get("strategy") != "none" and "error" not in r]
    if not rows:
        return "class_weight"
    return max(rows, key=lambda r: r.get("selection_score", r.get("macro_f1") or 0.0))["strategy"]


def plot_final(final: dict, figures: Path) -> None:
    label = MODEL_LABELS[final["model"]]
    m = final["metrics"]
    compare = final.get("comparison_strategy", "none")
    plotting.before_after(
        pd.Series({c: v["f1"] for c, v in m["baseline"]["per_class"].items()}),
        pd.Series({c: v["f1"] for c, v in m["classifier"]["per_class"].items()}),
        figures / "per_class_f1.png", "Per-class F1 on the test set",
        before_label=f"{label}, {strategy_text(compare)}",
        after_label=f"{label}, {strategy_text(final['strategy'])} (deployed)", xlabel="F1")
    if final.get("conformal"):
        per_class = pd.DataFrame(final["conformal"]["per_class"]).T
        labels = [f"{cov:.0%} · set {size:.2f}" for cov, size in zip(per_class["coverage"], per_class["mean_set_size"])]
        plotting.ranked_bars(
            per_class["coverage"], figures / "conformal_coverage.png",
            f"Per-family coverage of conformal prediction sets (target {final['conformal']['target_coverage']:.0%})",
            "Share of test flows whose true family is in the set", reference=final["conformal"]["target_coverage"],
            reference_label="guarantee", value_labels=labels, sort=False, xlim=(0, 1.15))


# --------------------------------------------------------------------------------------------------
# 2b. data audit and evaluation protocols
# --------------------------------------------------------------------------------------------------


def run_audit(cfg: dict) -> dict:
    """Data quality, identifier-like columns, and a single-feature shortcut scan."""
    dirs = output_dirs(cfg)
    data = load_dataset(cfg["data"], cfg["seed"])
    final = load_json(dirs["results"] / "final.json")
    full = final["metrics"]["classifier"]["macro_f1"] if final else None
    log.info("Audit: scanning %d features for shortcuts", len(data.feature_columns))
    quality = quality_table(data)
    scan = shortcut_scan(data, full, seed=cfg["seed"])
    result = {
        "generated_at": _now(),
        "rows": int(len(data.y)),
        "features": len(data.feature_columns),
        "dropped_duplicates": data.dropped_duplicates,
        "dropped_rare_classes": data.dropped_rare,
        "missing_rates": quality.loc[quality["missing_rate"] > 0, "missing_rate"].round(5).to_dict(),
        "identifier_suspects": identifier_suspects(data),
        "full_model_macro_f1": full,
        "shortcut_scan": scan.round(4).reset_index().to_dict("records"),
        "flagged": scan.index[scan["flag"] != ""].tolist(),
    }
    save_json(result, dirs["results"] / "audit.json")
    plot_audit(result, dirs["figures"])
    log.info("Audit: %d features flagged for review: %s", len(result["flagged"]), result["flagged"])
    return result


def plot_audit(audit: dict, figures: Path) -> None:
    scan = pd.DataFrame(audit["shortcut_scan"]).set_index("feature")
    plotting.ranked_bars(
        scan["single_feature_macro_f1"], figures / "shortcut_scan.png",
        "Shortcut scan: macro-F1 reachable from one feature alone",
        "Macro-F1 of a depth-4 tree using only this feature (3-fold CV)",
        reference=audit.get("full_model_macro_f1"), reference_label="full model",
        highlight=set(audit["flagged"]), top=15)


def session_groups(data: PreparedData, window: str = "1h") -> np.ndarray | None:
    """Session key per flow: source + destination + time window. None when addresses are unavailable."""
    ctx = data.context
    if not {"src_ip", "dst_ip"} <= set(ctx.columns):
        return None
    key = ctx["src_ip"].astype(str) + "|" + ctx["dst_ip"].astype(str)
    if "timestamp" in ctx.columns:
        key = key + "|" + pd.to_datetime(ctx["timestamp"]).dt.floor(window).astype(str)
    return pd.factorize(key)[0]


def run_protocols(cfg: dict) -> dict:
    """How optimistic is the headline number? Same model under three train/test protocols.

    * random     - stratified random split (the usual protocol; flows of one session can sit in train and test),
    * session    - session-grouped split: all flows sharing source, destination and hour stay on one side,
    * temporal   - oldest flows train, newest flows test (families that only appear later cannot be known).
    Classifier-only predictions, so the numbers isolate the protocol effect.
    """
    data, split, dirs = _prepare(cfg)
    model, strategy, _ = choose_final_config(cfg, dirs["results"])
    seed = cfg["seed"]
    X, y = data.X, data.y
    protocols: dict[str, dict] = {}

    def fit_eval(name: str, train_idx: np.ndarray, test_idx: np.ndarray, note: str) -> None:
        X_fit, y_fit = _trainable(X.iloc[train_idx], y.iloc[train_idx], f"Protocol {name}")
        log.info("Protocol %s: %d train / %d test flows", name, len(y_fit), len(test_idx))
        pipe = build_pipeline(model, strategy, y_fit, data.numeric, data.categorical, cfg).fit(X_fit, y_fit)
        pred, proba, classes = _predict(pipe, X.iloc[test_idx])
        y_test = y.iloc[test_idx]
        m = evaluate(y_test, pred, proba, classes)
        boot = bootstrap_metrics(y_test, {name: pred}, n_boot=cfg["evaluation"]["bootstrap"] // 2, seed=seed)
        protocols[name] = {
            **headline(m), "ci": boot["models"][name], "train_flows": int(len(y_fit)), "test_flows": int(len(test_idx)),
            "unseen_in_training": sorted(set(y_test) - set(y_fit)),
            "per_class_f1": {c: v["f1"] for c, v in m["per_class"].items()}, "note": note,
        }
        log.info("Protocol %s: macro-F1 %.3f, rare-class F1 %s", name, m["macro_f1"],
                 f"{m['rare_macro_f1']:.3f}" if m["rare_macro_f1"] is not None else "n/a")

    fit_eval("random", split.train, split.test, "stratified random 70/10/20 split, as in the headline results")
    groups = session_groups(data, cfg["protocols"]["session_window"])
    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
        shared = len(set(groups[train_idx]) & set(groups[test_idx]))
        fit_eval("session", train_idx, test_idx,
                 f"no session (source, destination, {cfg['protocols']['session_window']} window) spans train and test "
                 f"({shared} shared)")
    if data.time is not None:
        order = np.argsort(data.time.to_numpy(), kind="stable")
        cut = int(len(order) * cfg["drift"]["past_fraction"])
        fit_eval("temporal", np.sort(order[:cut]), np.sort(order[cut:]),
                 f"oldest {cfg['drift']['past_fraction']:.0%} of flows train, newest test")

    result = {"generated_at": _now(), "model": model, "strategy": strategy, "protocols": protocols}
    save_json(result, dirs["results"] / "protocols.json")
    return result


# --------------------------------------------------------------------------------------------------
# 3. novel attacks (leave-one-attack-out)
# --------------------------------------------------------------------------------------------------


def run_novelty(cfg: dict) -> dict:
    data, split, dirs = _prepare(cfg)
    nc = cfg["novelty"]
    X_tr, y_tr = _parts(data, split.train)
    X_va, y_va = _parts(data, split.val)
    X_te, y_te = _parts(data, split.test)
    detector = _fit_novelty(cfg, data, X_tr, y_tr, X_va, y_va)
    benign_te = X_te[(y_te == BENIGN).to_numpy()]
    benign_flag = detector.flag(benign_te)

    families = [c for c in class_order(y_te.unique())
                if c != BENIGN and int((y_te == c).sum()) >= nc["min_family_test_count"]]
    rows = []
    for fam in families:
        log.info("Leave-one-attack-out: training without %s", fam)
        keep = (y_tr != fam).to_numpy()
        X_fit, y_fit = _trainable(X_tr[keep], y_tr[keep], "Leave-one-attack-out")
        pipe = build_pipeline(nc["loao_model"], nc["loao_strategy"], y_fit, data.numeric, data.categorical, cfg)
        pipe.fit(X_fit, y_fit)
        X_f = X_te[(y_te == fam).to_numpy()]
        pred_f = np.asarray(pipe.predict(X_f)).astype(str)
        flag_f = detector.flag(X_f)
        pred_b = np.asarray(pipe.predict(benign_te)).astype(str)
        clf_alert = pred_f != BENIGN
        labels_given = pd.Series(pred_f[clf_alert]).value_counts()
        rows.append({
            "held_out_family": fam,
            "test_flows": int(len(X_f)),
            "classifier_detection": float(clf_alert.mean()),
            "hybrid_detection": float((clf_alert | flag_f).mean()),
            "caught_only_by_anomaly_detector": float((~clf_alert & flag_f).mean()),
            "most_common_label_given": str(labels_given.index[0]) if len(labels_given) else BENIGN,
            "classifier_false_alarm_rate": float((pred_b != BENIGN).mean()),
            "hybrid_false_alarm_rate": float(((pred_b != BENIGN) | benign_flag).mean()),
        })
        log.info("  detected: classifier %.1f%% -> hybrid %.1f%%",
                 100 * rows[-1]["classifier_detection"], 100 * rows[-1]["hybrid_detection"])

    df = pd.DataFrame(rows)
    result = {
        "generated_at": _now(),
        "model": nc["loao_model"],
        "strategy": nc["loao_strategy"],
        "benign_fpr_budget": nc["benign_fpr"],
        "anomaly_detector_false_alarm_rate": float(benign_flag.mean()),
        "mean_classifier_detection": float(df["classifier_detection"].mean()) if len(df) else None,
        "mean_hybrid_detection": float(df["hybrid_detection"].mean()) if len(df) else None,
        "rows": rows,
    }
    save_json(result, dirs["results"] / "novelty.json")
    plot_novelty(result, dirs["figures"])
    return result


def plot_novelty(novelty: dict, figures: Path) -> None:
    df = pd.DataFrame(novelty["rows"])
    if df.empty:
        return
    df = df.set_index("held_out_family")
    plotting.before_after(df["classifier_detection"], df["hybrid_detection"], figures / "novel_attack_detection.png",
                          "Attack families never seen in training: share raised as an alert",
                          "Classifier only", "Classifier + novelty detector", "Detection rate")


# --------------------------------------------------------------------------------------------------
# 4. drift over time
# --------------------------------------------------------------------------------------------------


def _safe_split(idx: np.ndarray, labels: pd.Series, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        a, b = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    except ValueError:  # some class has a single member in this period
        a, b = train_test_split(idx, test_size=test_size, random_state=seed)
    return np.sort(a), np.sort(b)


def run_drift(cfg: dict) -> dict:
    dirs = output_dirs(cfg)
    data = load_dataset(cfg["data"], cfg["seed"])
    if data.time is None:
        log.warning("No usable timestamp column: skipping the drift experiment")
        result = {"generated_at": _now(), "skipped": "dataset has no usable timestamp column"}
        save_json(result, dirs["results"] / "drift.json")
        return result

    dc, seed = cfg["drift"], cfg["seed"]
    X, y = data.X, data.y
    order = np.argsort(data.time.to_numpy(), kind="stable")
    cut = int(len(order) * dc["past_fraction"])
    past, future = np.sort(order[:cut]), np.sort(order[cut:])
    train, val = _safe_split(past, y.iloc[past], cfg["data"]["val_size"], seed)
    evaluation, labelled = _safe_split(future, y.iloc[future], dc["label_budget"], seed)
    model, strategy, _ = choose_final_config(cfg, dirs["results"])
    log.info("Drift: %d past flows (train %d), %d future flows (%d labelled for retraining); model %s + %s",
             len(past), len(train), len(future), len(labelled), model, strategy)

    def fit(train_idx):
        Xt, yt = _trainable(X.iloc[train_idx], y.iloc[train_idx], "Drift")
        pipe = build_pipeline(model, strategy, yt, data.numeric, data.categorical, cfg).fit(Xt, yt)
        detector = _fit_novelty(cfg, data, Xt, yt, X.iloc[val], y.iloc[val])
        return pipe, detector

    def hybrid(pipe, detector, idx):
        pred, proba, classes = _predict(pipe, X.iloc[idx])
        flags = detector.flag(X.iloc[idx])
        return pred, np.where((pred == BENIGN) & flags, UNKNOWN, pred), proba, classes

    static_pipe, static_det = fit(train)
    clf_val, hyb_val, p_val, c_val = hybrid(static_pipe, static_det, val)
    m_past = evaluate(y.iloc[val], hyb_val, p_val, c_val)
    clf_s, hyb_s, p_s, c_s = hybrid(static_pipe, static_det, evaluation)
    m_static = evaluate(y.iloc[evaluation], hyb_s, p_s, c_s)

    log.info("Drift: retraining with the labelled sample of recent traffic")
    re_pipe, re_det = fit(np.concatenate([train, labelled]))
    _, hyb_r, p_r, c_r = hybrid(re_pipe, re_det, evaluation)
    m_re = evaluate(y.iloc[evaluation], hyb_r, p_r, c_r)

    # Label-free drift signals, i.e. what a production monitor can compute without ground truth.
    # Attacks are a small share of traffic, so drift in them barely moves whole-traffic feature
    # distributions; the alerted population and the prediction mix are far more sensitive.
    psi = psi_table(X.iloc[train], X.iloc[future], data.numeric)
    psi_alerted = psi_table(X.iloc[val][hyb_val != BENIGN], X.iloc[evaluation][hyb_s != BENIGN], data.numeric)
    alert = dc["psi_alert"]
    drifted = sorted(set(psi[psi > alert].index) | set(psi_alerted[psi_alerted > alert].index),
                     key=lambda f: -max(psi.get(f, 0), psi_alerted.get(f, 0)))
    signals = {
        "max_feature_psi_all_traffic": float(psi.max()),
        "max_feature_psi_alerted_traffic": float(psi_alerted.max()),
        "prediction_mix_psi": _category_psi(hyb_val, hyb_s),
        "unknown_anomaly_rate": {"history": float((hyb_val == UNKNOWN).mean()),
                                 "recent": float((hyb_s == UNKNOWN).mean())},
        "mean_alert_confidence": {"history": float(p_val.max(axis=1)[clf_val != BENIGN].mean()),
                                  "recent": float(p_s.max(axis=1)[clf_s != BENIGN].mean())},
    }
    triggers = {
        "feature_drift": bool(drifted),
        "prediction_mix_shift": signals["prediction_mix_psi"] > dc.get("prediction_shift_alert", 0.1),
        "unknown_rate_rise": signals["unknown_anomaly_rate"]["recent"]
        > dc.get("unknown_rate_ratio_alert", 2.0) * max(signals["unknown_anomaly_rate"]["history"], 1e-3),
    }
    y_eval = y.iloc[evaluation].to_numpy()
    new_families = []
    for fam in sorted(set(y.iloc[future]) - set(y.iloc[past])):
        mask = y_eval == fam
        new_families.append({
            "family": fam,
            "flows": int(mask.sum()),
            "static_classifier_detection": float((clf_s[mask] != BENIGN).mean()),
            "static_hybrid_detection": float((hyb_s[mask] != BENIGN).mean()),
            "flagged_as_unknown": float((hyb_s[mask] == UNKNOWN).mean()),
            "retrained_recall": m_re["per_class"].get(fam, {}).get("recall"),
        })

    result = {
        "generated_at": _now(),
        "model": model,
        "strategy": strategy,
        "time_ranges": {
            "past": [data.time.iloc[past].min(), data.time.iloc[past].max()],
            "future": [data.time.iloc[future].min(), data.time.iloc[future].max()],
        },
        "sizes": {"past_train": len(train), "past_validation": len(val), "future_labelled": len(labelled),
                  "future_evaluation": len(evaluation)},
        "metrics": {"past_validation": m_past, "future_static": m_static, "future_retrained": m_re},
        "psi_all_traffic": psi.round(4).to_dict(),
        "psi_alerted_traffic": psi_alerted.round(4).to_dict(),
        "psi_alert": alert,
        "drifted_features": drifted,
        "signals": signals,
        "triggers": triggers,
        "retraining_triggered": any(triggers.values()),
        "new_families": new_families,
    }
    result["label_budget"] = dc["label_budget"]
    save_json(result, dirs["results"] / "drift.json")
    plot_drift(result, dirs["figures"])
    log.info("Drift: macro-F1 past %.3f -> future static %.3f -> retrained %.3f; triggers %s; drifted features %s",
             m_past["macro_f1"], m_static["macro_f1"], m_re["macro_f1"], triggers, drifted)
    return result


def plot_drift(drift: dict, figures: Path) -> None:
    if drift.get("skipped"):
        return
    m = drift["metrics"]
    plotting.before_after(
        pd.Series({c: v["recall"] for c, v in m["future_static"]["per_class"].items()}),
        pd.Series({c: v["recall"] for c, v in m["future_retrained"]["per_class"].items()}),
        figures / "drift_recall.png", "Recent traffic: per-class recall before and after retraining",
        "Static model (trained on history only)",
        f"Retrained with {drift.get('label_budget', 0.1):.0%} labelled recent flows", "Recall")
    plotting.psi_bars(pd.Series(drift["psi_alerted_traffic"]), drift["psi_alert"], figures / "drift_psi.png",
                      title="Feature drift in alerted traffic (history vs. recent)")


def _category_psi(a: np.ndarray, b: np.ndarray) -> float:
    """PSI between two label distributions (e.g. the mix of predicted classes)."""
    cats = sorted(set(a) | set(b))
    pa = np.clip(np.array([(a == c).mean() for c in cats]), 1e-4, None)
    pb = np.clip(np.array([(b == c).mean() for c in cats]), 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def model_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["artifacts"]) / MODEL_FILE


# --------------------------------------------------------------------------------------------------
# 5. closing the loop: analyst feedback -> retraining
# --------------------------------------------------------------------------------------------------


def run_retrain(cfg: dict, feedback_path: str | Path, model_file: str | Path | None = None,
                out_path: str | Path | None = None) -> dict:
    """Retrain the deployed model on the original training data plus analyst-labelled flows.

    Flows analysts already labelled are removed from the test set before comparing old and new models: they are
    training data now, and evaluating on them would flatter the new model. By default the retrained model
    replaces the deployed one and the previous version is archived under ``artifacts/model_history/``.
    """
    data, split, dirs = _prepare(cfg)
    model_file = Path(model_file or model_path(cfg))
    old = ModelBundle.load(model_file)
    cols = old.numeric + old.categorical
    if set(cols) != set(data.feature_columns):
        log.warning("Model features differ from the current dataset's; retraining with the model's feature set")
    X_fb, y_fb, fb, summary = load_feedback_examples(feedback_path, old.numeric, old.categorical)
    if not len(y_fb):
        raise ValueError(f"No usable analyst decisions in {feedback_path}")
    log.info("Feedback: %d labelled flows from %d incidents %s", len(y_fb), summary["incidents"], summary["by_label"])

    # Evaluation set: the test split minus flows that analysts labelled (now training data).
    test_pos = split.test
    flow_index = pd.to_numeric(fb["flow_index"], errors="coerce")
    on_test = ((fb["source"] == "test-set") & flow_index.between(0, len(test_pos) - 1)).to_numpy()
    reviewed = np.unique(flow_index[on_test].astype(int).to_numpy())
    if on_test.any():  # sanity check: the feedback must describe the same test flows
        mine = X_fb.loc[on_test, old.numeric].to_numpy(dtype=float)
        ref = data.X.iloc[test_pos[flow_index[on_test].astype(int).to_numpy()]][old.numeric].to_numpy(dtype=float)
        same = np.isclose(mine, ref, rtol=1e-6, equal_nan=True).all(axis=1).mean()
        if same < 0.99:
            log.warning("Test-set feedback does not match the current test split (%.0f%% match): the dataset or "
                        "split changed since review, so the old-vs-new comparison may be optimistic", 100 * same)
    eval_pos = test_pos[np.setdiff1d(np.arange(len(test_pos)), reviewed)]
    X_ev, y_ev = data.X.iloc[eval_pos][cols], data.y.iloc[eval_pos]

    model = old.metadata.get("model") or choose_final_config(cfg, dirs["results"])[0]
    strategy = old.metadata.get("strategy") or choose_final_config(cfg, dirs["results"])[1]
    X_tr, y_tr = _parts(data, split.train)
    X_va, y_va = _parts(data, split.val)
    weight = max(int(cfg.get("feedback", {}).get("weight", 1)), 1)  # copies of each analyst-labelled flow
    X_new = pd.concat([X_tr[cols]] + [X_fb[cols]] * weight, ignore_index=True)
    y_new = pd.concat([y_tr] + [y_fb] * weight, ignore_index=True)
    X_new, y_new = _trainable(X_new, y_new, "Retrain")
    log.info("Retraining %s + %s on %d original + %d analyst-labelled flows (weight %d)",
             model, strategy, len(y_tr), len(y_fb), weight)
    t0 = time.perf_counter()
    pipe = build_pipeline(model, strategy, y_new, old.numeric, old.categorical, cfg).fit(X_new, y_new)
    fit_s = time.perf_counter() - t0
    # Confirmed false positives always join the novelty detector's benign reference set, so that exact
    # pattern stops being reported as Unknown-Anomaly.
    confirmed_benign = X_fb.loc[(y_fb == BENIGN).to_numpy(), cols]
    novelty = _fit_novelty(cfg, data, X_tr[cols], y_tr, X_va[cols], y_va, X_always=confirmed_benign)
    new_families = sorted(set(y_fb) - set(old.classes))
    new = ModelBundle(
        classifier=pipe,
        novelty=novelty,
        baseline=BenignBaseline().fit(X_new[(y_new == BENIGN).to_numpy()]),
        numeric=old.numeric,
        categorical=old.categorical,
        severity=old.severity,
        conf_threshold=old.conf_threshold,
        label_mode=old.label_mode,
        conformal=_fit_conformal(cfg, pipe, X_va[cols], y_va),
        metadata={**old.metadata, "trained_at": _now(), "retrained_from": old.metadata.get("trained_at"),
                  "feedback_flows": int(len(y_fb)), "feedback_file": str(feedback_path)},
    )
    new.metadata["classes"] = new.classes

    def score(bundle: ModelBundle) -> tuple[dict, dict, np.ndarray]:
        pred = bundle.predict(X_ev)
        metrics = evaluate(y_ev, pred["label"], pred["proba"], pred["classes"])
        alerts = bundle.alerts(X_ev, data.context.iloc[eval_pos].reset_index(drop=True), prediction=pred)
        incidents, _ = group_incidents(alerts)
        volume = {"alerts": int(len(alerts)), "incidents": int(len(incidents)),
                  "unknown_alerts": int((alerts["predicted_class"] == UNKNOWN).sum()) if len(alerts) else 0}
        return metrics, volume, pred["label"]

    m_old, v_old, labels_old = score(old)
    m_new, v_new, labels_new = score(new)
    boot = bootstrap_metrics(y_ev, {"before": labels_old, "after": labels_new}, pairs=[("before", "after")],
                             n_boot=cfg["evaluation"]["bootstrap"], seed=cfg["seed"])
    new.metadata["test_macro_f1"] = m_new["macro_f1"]

    out_path = Path(out_path) if out_path else model_file
    archived = None
    if out_path.resolve() == model_file.resolve():
        history = model_file.parent / "model_history"
        history.mkdir(parents=True, exist_ok=True)
        archived = history / f"{model_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{model_file.suffix}"
        model_file.replace(archived)
        log.info("Archived the previous model to %s", archived)
    new.save(out_path)
    log.info("Saved retrained model to %s", out_path)
    log.info("Remaining test flows: macro-F1 %.3f -> %.3f | false alarms %.2f%% -> %.2f%% | incidents %d -> %d",
             m_old["macro_f1"], m_new["macro_f1"], 100 * m_old["false_alarm_rate"], 100 * m_new["false_alarm_rate"],
             v_old["incidents"], v_new["incidents"])

    result = {
        "generated_at": _now(),
        "model": model,
        "strategy": strategy,
        "feedback": summary,
        "new_families": new_families,
        "fit_seconds": round(fit_s, 1),
        "evaluation": {"test_flows": int(len(test_pos)), "excluded_reviewed_flows": int(len(reviewed)),
                       "evaluated_flows": int(len(eval_pos))},
        "metrics": {"before": m_old, "after": m_new},
        "alert_volume": {"before": v_old, "after": v_new},
        "bootstrap": boot,
        "artifacts": {"model": str(out_path), "archived_previous": str(archived) if archived else None},
    }
    save_json(result, dirs["results"] / "retrain.json")
    plot_retrain(result, dirs["figures"])
    return result


def plot_retrain(retrain: dict, figures: Path) -> None:
    m = retrain["metrics"]
    plotting.before_after(
        pd.Series({c: v["f1"] for c, v in m["before"]["per_class"].items()}),
        pd.Series({c: v["f1"] for c, v in m["after"]["per_class"].items()}),
        figures / "retrain_per_class_f1.png", "After analyst feedback: per-class F1 on unreviewed test flows",
        "Deployed model", f"Retrained with {retrain['feedback']['usable']:,} analyst-labelled flows", "F1")
