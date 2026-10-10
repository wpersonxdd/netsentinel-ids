"""Command-line interface: ``python -m nids <command>`` (or ``nids <command>`` after ``pip install -e .``)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from nids.config import load_config


def _cfg(args) -> dict:
    overrides = {}
    if getattr(args, "data", None):
        overrides["data"] = {"path": args.data}
    if getattr(args, "max_rows", None):
        overrides.setdefault("data", {})["max_rows"] = args.max_rows
    return load_config(args.config, overrides)


def cmd_generate(args) -> None:
    from nids.synthetic import generate_flows

    df = generate_flows(args.n_samples, seed=args.seed, drift=not args.no_drift)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logging.info("Wrote %d synthetic flows to %s\n%s", len(df), out, df["Label"].value_counts().to_string())


def _ensure_data(cfg: dict) -> None:
    """Generate the synthetic demo dataset on first run if the config points at it and it is missing."""
    path = Path(cfg["data"]["path"])
    if not path.exists() and path.name == "synthetic_flows.csv":
        from nids.synthetic import generate_flows

        logging.info("%s not found: generating the synthetic demo dataset", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_flows(100_000, seed=cfg["seed"]).to_csv(path, index=False)


def cmd_benchmark(args) -> None:
    from nids.experiments import run_benchmark

    cfg = _cfg(args)
    _ensure_data(cfg)
    run_benchmark(cfg)


def cmd_train(args) -> None:
    from nids.experiments import run_train

    cfg = _cfg(args)
    _ensure_data(cfg)
    if args.model:
        cfg["final_model"]["model"] = args.model
    if args.strategy:
        cfg["final_model"]["strategy"] = args.strategy
    run_train(cfg)


def cmd_novelty(args) -> None:
    from nids.experiments import run_novelty

    cfg = _cfg(args)
    _ensure_data(cfg)
    run_novelty(cfg)


def cmd_drift(args) -> None:
    from nids.experiments import run_drift

    cfg = _cfg(args)
    _ensure_data(cfg)
    run_drift(cfg)


def cmd_report(args) -> None:
    from nids.report import write_report

    write_report(_cfg(args))


def cmd_score(args) -> None:
    from nids.bundle import ModelBundle
    from nids.data import read_table

    from nids.triage import group_incidents

    bundle = ModelBundle.load(args.model)
    raw = read_table(args.input)
    alerts, per_flow = bundle.score_frame(raw)
    incidents, alerts = group_incidents(alerts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(out, index=False)
    incidents_out = out.with_name(out.stem + "_incidents" + out.suffix)
    incidents.to_csv(incidents_out, index=False)
    if args.flows_out:
        per_flow.to_csv(args.flows_out, index=False)
    by_priority = incidents["priority"].value_counts().sort_index().to_dict() if len(incidents) else {}
    logging.info("Scored %d flows -> %d alerts in %d incidents %s; wrote %s and %s",
                 len(raw), len(alerts), len(incidents), by_priority, out, incidents_out)


def cmd_simulate_feedback(args) -> None:
    from nids.feedback import simulate_feedback

    cfg = _cfg(args)
    alerts = pd.read_csv(args.alerts or Path(cfg["paths"]["artifacts"]) / "alerts_test.csv", low_memory=False)
    budget = args.budget or cfg["feedback"]["simulated_incidents"]
    rows = simulate_feedback(alerts, budget, args.order or cfg["feedback"]["review_order"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out, index=False)
    logging.info("Wrote %d simulated analyst decisions to %s", len(rows), out)


def cmd_retrain(args) -> None:
    from nids.experiments import run_retrain

    cfg = _cfg(args)
    run_retrain(cfg, args.feedback, args.model, args.out)


def _feedback_demo(cfg: dict) -> None:
    """Close the loop on the test split with a simulated analyst; the deployed model is left untouched."""
    from nids.experiments import ALERTS_FILE, run_retrain
    from nids.feedback import simulate_feedback

    artifacts = Path(cfg["paths"]["artifacts"])
    alerts = pd.read_csv(artifacts / ALERTS_FILE, low_memory=False)
    if "true_label" not in alerts.columns:
        logging.info("No ground truth in the alert queue: skipping the simulated feedback loop")
        return
    sim = artifacts / "simulated_feedback.csv"
    simulate_feedback(alerts, cfg["feedback"]["simulated_incidents"], cfg["feedback"]["review_order"]).to_csv(
        sim, index=False)
    run_retrain(cfg, sim, out_path=artifacts / "netsentinel_model_retrained.joblib")


def cmd_all(args) -> None:
    from nids.experiments import run_benchmark, run_drift, run_novelty, run_train
    from nids.report import write_report
    from nids.synthetic import generate_flows

    cfg = _cfg(args)
    _ensure_data(cfg)
    started = time.perf_counter()
    if args.skip_benchmark:
        logging.info("Skipping the benchmark (final model from config or defaults)")
    else:
        run_benchmark(cfg)
    run_train(cfg)
    run_novelty(cfg)
    run_drift(cfg)
    _feedback_demo(cfg)
    demo = Path("data/raw/demo_new_traffic.csv")
    if not demo.exists():
        generate_flows(5_000, seed=cfg["seed"] + 1).to_csv(demo, index=False)
    write_report(cfg)
    logging.info("All done in %.1f min. Open reports/REPORT.md or run: streamlit run app/streamlit_app.py",
                 (time.perf_counter() - started) / 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nids", description="NetSentinel: imbalance-aware network intrusion detection")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p):
        p.add_argument("--config", default="configs/default.yaml", help="YAML config (default: %(default)s)")
        p.add_argument("--data", help="override data.path (file or directory of CSV/Parquet)")
        p.add_argument("--max-rows", type=int, help="override data.max_rows (stratified subsample)")
        return p

    p = sub.add_parser("generate-data", help="write the synthetic demo dataset")
    p.add_argument("--n-samples", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-drift", action="store_true", help="single period, no concept drift / new family")
    p.add_argument("--out", default="data/raw/synthetic_flows.csv")
    p.set_defaults(func=cmd_generate)

    with_config(sub.add_parser("benchmark", help="compare models x imbalance strategies on validation")).set_defaults(
        func=cmd_benchmark)
    p = with_config(sub.add_parser("train", help="train the final model, evaluate on test, save the bundle"))
    p.add_argument("--model", help="override final_model.model")
    p.add_argument("--strategy", help="override final_model.strategy")
    p.set_defaults(func=cmd_train)
    with_config(sub.add_parser("novelty", help="leave-one-attack-out novel-attack experiment")).set_defaults(
        func=cmd_novelty)
    with_config(sub.add_parser("drift", help="temporal drift + retraining experiment")).set_defaults(func=cmd_drift)
    with_config(sub.add_parser("report", help="write reports/REPORT.md from the saved results")).set_defaults(
        func=cmd_report)

    p = sub.add_parser("score", help="score new traffic with a saved model and write the alert queue")
    p.add_argument("--model", default="artifacts/netsentinel_model.joblib")
    p.add_argument("--input", required=True, help="CSV/Parquet file or directory of flow records")
    p.add_argument("--out", default="artifacts/alerts.csv")
    p.add_argument("--flows-out", help="also write per-flow predictions here")
    p.set_defaults(func=cmd_score)

    p = with_config(sub.add_parser("simulate-feedback",
                                   help="play an analyst who labels the top incidents of a labelled alert queue"))
    p.add_argument("--alerts", help="alert CSV with a true_label column (default: artifacts/alerts_test.csv)")
    p.add_argument("--budget", type=int, help="incidents to review (default: feedback.simulated_incidents)")
    p.add_argument("--order", choices=["review-first", "priority"], help="which incidents the analyst opens first")
    p.add_argument("--out", default="artifacts/simulated_feedback.csv")
    p.set_defaults(func=cmd_simulate_feedback)

    p = with_config(sub.add_parser("retrain", help="retrain the deployed model with analyst decisions"))
    p.add_argument("--feedback", default="artifacts/analyst_feedback.csv", help="decisions saved by the console")
    p.add_argument("--model", help="model to start from (default: artifacts/netsentinel_model.joblib)")
    p.add_argument("--out", help="where to save; default replaces the model and archives the old one")
    p.set_defaults(func=cmd_retrain)

    p = with_config(sub.add_parser("all", help="benchmark -> train -> novelty -> drift -> feedback loop -> report"))
    p.add_argument("--skip-benchmark", action="store_true")
    p.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    pd.set_option("display.width", 140)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
