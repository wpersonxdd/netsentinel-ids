"""Everything needed to score new traffic, saved as one file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nids import BENIGN, UNKNOWN
from nids.data import prepare_for_inference
from nids.novelty import NoveltyDetector
from nids.triage import BenignBaseline, build_alerts


@dataclass
class ModelBundle:
    classifier: Any                 # fitted imblearn Pipeline (preprocess -> resample -> classifier)
    novelty: NoveltyDetector
    baseline: BenignBaseline
    numeric: list[str]
    categorical: list[str]
    severity: dict
    conf_threshold: float = 0.6
    label_mode: str = "family"
    metadata: dict = field(default_factory=dict)
    conformal: Any = None           # MondrianConformal: per-family prediction sets with a coverage guarantee

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.classifier.classes_]

    def predict(self, X: pd.DataFrame) -> dict:
        """Hybrid decision: supervised label, overridden to Unknown-Anomaly for anomalous 'benign' flows."""
        proba = self.classifier.predict_proba(X)
        classes = np.asarray(self.classes)
        classifier_label = classes[proba.argmax(axis=1)]
        anomaly = self.novelty.score(X)
        novel = anomaly > self.novelty.threshold_
        label = classifier_label.astype(object)
        label[(classifier_label == BENIGN) & novel] = UNKNOWN
        out = {
            "label": label.astype(str),
            "classifier_label": classifier_label,
            "confidence": proba.max(axis=1),
            "proba": proba,
            "classes": classes,
            "anomaly_score": anomaly,
            "novel": novel,
        }
        if self.conformal is not None:
            sets = self.conformal.sets(proba)
            out.update(prediction_set=sets, set_size=sets.sum(axis=1),
                       possible_families=np.asarray(self.conformal.describe(sets)))
        return out

    def alerts(self, X: pd.DataFrame, context: pd.DataFrame, y_true=None, prediction: dict | None = None) -> pd.DataFrame:
        prediction = prediction or self.predict(X)
        return build_alerts(prediction, X, context, self.baseline, self.severity, self.novelty.threshold_,
                            self.conf_threshold, y_true)

    def score_frame(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score raw flow records (any supported column naming). Returns (alerts, per-flow predictions)."""
        time = self.metadata.get("time_parsing", {})
        X, context, y_true = prepare_for_inference(raw, self.numeric, self.categorical, self.label_mode,
                                                   time.get("dayfirst", False), time.get("add_12h_before_hour"))
        prediction = self.predict(X)
        per_flow = context.copy()
        per_flow["predicted_class"] = prediction["label"]
        per_flow["confidence"] = prediction["confidence"].round(3)
        per_flow["anomaly_score"] = prediction["anomaly_score"].round(3)
        if y_true is not None:
            per_flow["true_label"] = y_true.values
        return self.alerts(X, context, y_true, prediction), per_flow

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        return joblib.load(path)
