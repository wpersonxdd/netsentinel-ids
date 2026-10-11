"""Experiment configuration: YAML file merged over built-in defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from nids.triage import DEFAULT_SEVERITY

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "paths": {"artifacts": "artifacts", "reports": "reports"},
    "data": {
        "path": "data/raw/synthetic_flows.csv",
        "label_column": None,
        "time_column": None,
        "label_mode": "family",
        "extra_drop_columns": [],
        "max_rows": None,
        "min_per_class": 1000,
        "time_dayfirst": False,
        "time_add_12h_before_hour": None,
        "min_class_count": 10,
        "test_size": 0.2,
        "val_size": 0.1,
    },
    "resampling": {"smote_target_ratio": 0.15, "majority_cap_ratio": 4.0},
    "models": {
        "logreg": {"C": 1.0, "max_iter": 1000},
        "random_forest": {"n_estimators": 150, "min_samples_leaf": 1},
        "hist_gb": {"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 31, "l2_regularization": 0.0},
        "deep_mlp": {
            "hidden_dims": [256, 128, 64],
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 512,
            "max_epochs": 30,
            "patience": 5,
            "gamma": 2.0,
        },
    },
    "benchmark": {
        "models": ["logreg", "random_forest", "hist_gb", "deep_mlp", "ensemble"],
        "strategies": ["none", "class_weight", "smote", "hybrid"],
        "selection_metric": "macro_f1",
        "min_val_support": 10,
        "max_train_rows": None,
    },
    "final_model": {"model": "auto", "strategy": "auto"},
    "explain": {"permutation_rows": 4000, "permutation_repeats": 3},
    "novelty": {
        "benign_fpr": 0.01,
        "method": "lof",
        "n_neighbors": 10,
        "max_fit_rows": 20000,
        "n_estimators": 200,
        "loao_model": "hist_gb",
        "loao_strategy": "smote",
        "min_family_test_count": 15,
    },
    "drift": {"past_fraction": 0.65, "label_budget": 0.10, "psi_alert": 0.25, "prediction_shift_alert": 0.1,
              "unknown_rate_ratio_alert": 2.0},
    "triage": {"confidence_threshold": 0.6, "severity": dict(DEFAULT_SEVERITY)},
    "feedback": {"simulated_incidents": 100, "review_order": "review-first", "weight": 1},
    "conformal": {"alpha": 0.1},
    "evaluation": {"bootstrap": 1000},
    "protocols": {"session_window": "1h"},
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load a YAML config (falling back to defaults when it is absent) and apply overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def output_dirs(cfg: dict) -> dict[str, Path]:
    """Create and return the output directories used by every command."""
    reports = Path(cfg["paths"]["reports"])
    dirs = {
        "artifacts": Path(cfg["paths"]["artifacts"]),
        "reports": reports,
        "figures": reports / "figures",
        "results": reports / "results",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
