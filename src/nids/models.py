"""Model zoo and the preprocessing -> resampling -> classifier pipeline."""

from __future__ import annotations

import logging

import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from nids.deep import TORCH_AVAILABLE, TorchMLPClassifier
from nids.features import build_preprocessor
from nids.resampling import make_samplers

log = logging.getLogger(__name__)

MODELS = ("logreg", "random_forest", "hist_gb", "deep_mlp", "ensemble")
MODEL_LABELS = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gb": "Gradient boosting",
    "deep_mlp": "Deep MLP (focal loss)",
    "ensemble": "Soft-voting ensemble",
}
STRATEGY_LABELS = {"none": "none", "class_weight": "class weights", "smote": "SMOTE", "hybrid": "undersample + SMOTE"}


def available_models(names: list[str]) -> list[str]:
    """Drop models whose optional dependencies are missing."""
    out = []
    for name in names:
        if name == "deep_mlp" and not TORCH_AVAILABLE:
            log.warning("PyTorch not installed: skipping deep_mlp (pip install torch)")
            continue
        out.append(name)
    return out


def make_classifier(name: str, strategy: str, model_cfg: dict, seed: int):
    """Instantiate one classifier. ``strategy='class_weight'`` switches on its built-in reweighting."""
    balanced = strategy == "class_weight"
    if name == "logreg":
        p = model_cfg.get("logreg", {})
        return LogisticRegression(C=p.get("C", 1.0), max_iter=p.get("max_iter", 1000),
                                  class_weight="balanced" if balanced else None)
    if name == "random_forest":
        p = model_cfg.get("random_forest", {})
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 150),
                                      min_samples_leaf=p.get("min_samples_leaf", 1),
                                      class_weight="balanced_subsample" if balanced else None,
                                      n_jobs=-1, random_state=seed)
    if name == "hist_gb":
        p = model_cfg.get("hist_gb", {})
        return HistGradientBoostingClassifier(max_iter=p.get("max_iter", 300),
                                              learning_rate=p.get("learning_rate", 0.1),
                                              max_leaf_nodes=p.get("max_leaf_nodes", 31),
                                              l2_regularization=p.get("l2_regularization", 0.0),
                                              class_weight="balanced" if balanced else None,
                                              early_stopping=True, random_state=seed)
    if name == "deep_mlp":
        p = model_cfg.get("deep_mlp", {})
        return TorchMLPClassifier(hidden_dims=tuple(p.get("hidden_dims", (256, 128, 64))),
                                  dropout=p.get("dropout", 0.2), lr=p.get("lr", 1e-3),
                                  batch_size=p.get("batch_size", 512), max_epochs=p.get("max_epochs", 30),
                                  patience=p.get("patience", 5),
                                  # The untreated baseline uses plain cross-entropy; every imbalance
                                  # strategy trains with focal loss.
                                  gamma=0.0 if strategy == "none" else p.get("gamma", 2.0),
                                  class_weight="balanced" if balanced else None, random_state=seed)
    if name == "ensemble":
        members = [("rf", make_classifier("random_forest", strategy, model_cfg, seed)),
                   ("hgb", make_classifier("hist_gb", strategy, model_cfg, seed))]
        if TORCH_AVAILABLE:
            members.append(("mlp", make_classifier("deep_mlp", strategy, model_cfg, seed)))
        return VotingClassifier(members, voting="soft")
    raise ValueError(f"Unknown model '{name}'. Choose from {MODELS}")


def build_pipeline(model: str, strategy: str, y_train: pd.Series, numeric: list[str], categorical: list[str],
                   cfg: dict) -> Pipeline:
    """preprocess -> (undersample) -> (SMOTE) -> classifier. Samplers run only during ``fit``."""
    steps = [("pre", build_preprocessor(numeric, categorical))]
    steps += make_samplers(strategy, y_train, cfg.get("resampling", {}), cfg.get("seed", 42))
    steps.append(("clf", make_classifier(model, strategy, cfg.get("models", {}), cfg.get("seed", 42))))
    return Pipeline(steps)
