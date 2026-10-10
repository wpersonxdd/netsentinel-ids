"""Analyst feedback: decisions in the console become labelled training data.

Every decision is stored as one row per flow *including the flow's feature values* (the ``feat_*`` columns
alerts carry), so a feedback file is self-contained training data. It does not matter whether the flow came
from the test split or from a traffic file scored months later.

    console decision ──► artifacts/analyst_feedback.csv ──► `nids retrain` ──► new model (old one archived)

Labels: "False positive" → Benign; "Confirmed threat" / "Escalated" → the corrected label; "Needs more data" →
not used; "Other / new family" (or an unresolved Unknown-Anomaly) → the family name typed in the note, if any.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nids import BENIGN, UNKNOWN
from nids.data import as_category_text
from nids.triage import FEATURE_PREFIX, group_incidents

log = logging.getLogger(__name__)

DECISIONS = ["Confirmed threat", "False positive", "Escalated to incident response", "Needs more data"]
NEW_FAMILY = "Other / new family"
SIMULATED_NOTE = "simulated analyst (ground truth)"
META_COLUMNS = ["alert_id", "flow_index", "source", "incident_id", "reviewed_at", "decision", "corrected_label",
                "note", "predicted_class"]


def feedback_rows(alerts: pd.DataFrame, decision, corrected_label, note: str = "", source: str = "test-set",
                  reviewed_at: str | None = None) -> pd.DataFrame:
    """Feedback rows for the given alert rows. ``decision`` / ``corrected_label`` may be scalars or arrays."""
    n = len(alerts)
    decision = np.broadcast_to(np.asarray(decision, dtype=object), (n,))
    corrected = np.broadcast_to(np.asarray(corrected_label, dtype=object), (n,))
    meta = pd.DataFrame({
        "alert_id": alerts["alert_id"].to_numpy(),
        "flow_index": alerts["flow_index"].to_numpy() if "flow_index" in alerts.columns else -1,
        "source": source,
        "incident_id": alerts["incident_id"].to_numpy() if "incident_id" in alerts.columns else "",
        "reviewed_at": reviewed_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": decision,
        "corrected_label": np.where(decision == "False positive", BENIGN, corrected),
        "note": note,
        "predicted_class": alerts["predicted_class"].to_numpy(),
    })
    features = alerts[[c for c in alerts.columns if c.startswith(FEATURE_PREFIX)]].reset_index(drop=True)
    return pd.concat([meta, features], axis=1)


def append_feedback(rows: pd.DataFrame, path: str | Path) -> Path:
    """Append rows to a feedback CSV, keeping the existing file's column layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        rows = rows.reindex(columns=pd.read_csv(path, nrows=0).columns)
    rows.to_csv(path, mode="a", header=not exists, index=False)
    return path


def resolve_label(decision, corrected_label, note) -> str | None:
    """Training label for one decision, or None when the decision carries no usable label."""
    if decision == "Needs more data":
        return None
    label = "" if pd.isna(corrected_label) else str(corrected_label).strip()
    if label in ("", NEW_FAMILY, UNKNOWN):
        typed = "" if pd.isna(note) else str(note).strip()
        return typed if typed and typed != SIMULATED_NOTE else None
    return label


def load_feedback_examples(path: str | Path, numeric: list[str], categorical: list[str]):
    """Read a feedback CSV as training data. Returns (X, y, usable feedback rows, summary)."""
    raw = pd.read_csv(path, low_memory=False)
    fb = raw.drop_duplicates(["source", "alert_id"], keep="last").reset_index(drop=True)
    labels = pd.Series([resolve_label(d, c, n) for d, c, n in zip(fb["decision"], fb["corrected_label"], fb["note"])])
    usable = labels.notna().to_numpy()
    missing = [c for c in numeric + categorical if FEATURE_PREFIX + c not in fb.columns]
    if missing:
        raise ValueError(f"Feedback file lacks feature values for {missing[:5]}…; it was written by an older "
                         "version or for a different model. Re-review the alerts in the current console.")
    fb, y = fb[usable].reset_index(drop=True), labels[usable].reset_index(drop=True)
    X = pd.DataFrame({c: fb[FEATURE_PREFIX + c] for c in numeric + categorical})
    X[numeric] = X[numeric].apply(pd.to_numeric, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    for c in categorical:
        X[c] = as_category_text(X[c])
    summary = {
        "rows": int(len(raw)),
        "unique_flows": int(len(usable)),
        "usable": int(usable.sum()),
        "skipped_without_label": int((~usable).sum()),
        "by_decision": fb["decision"].value_counts().to_dict(),
        "by_label": y.value_counts().to_dict(),
        "by_source": fb["source"].value_counts().to_dict(),
        "incidents": int(fb["incident_id"].nunique()),
        "simulated": bool(len(fb)) and bool((fb["note"].astype(str) == SIMULATED_NOTE).all()),
    }
    return X, y, fb, summary


def simulate_feedback(alerts: pd.DataFrame, budget: int = 100, order: str = "review-first") -> pd.DataFrame:
    """Play an analyst who works through ``budget`` incidents and labels every member flow correctly.

    ``order="review-first"`` opens incidents flagged *needs review* first (low confidence / unknown family), then
    by priority, which is how a team uses analyst time where the model is least sure. ``order="priority"`` simply
    works the queue top-down. Needs a ``true_label`` column, i.e. alerts scored on labelled evaluation data.
    """
    if "true_label" not in alerts.columns:
        raise ValueError("Simulated feedback needs ground truth: score labelled data (e.g. the test split).")
    incidents, alerts = group_incidents(alerts.drop(columns=["incident_id"], errors="ignore"))
    keys = ["needs_review", "priority_score", "flows"] if order == "review-first" else ["priority_score", "flows"]
    chosen = incidents.sort_values(keys, ascending=False, kind="stable").head(budget)["incident_id"]
    members = alerts[alerts["incident_id"].isin(set(chosen))]
    truth = members["true_label"].astype(str).to_numpy()
    decision = np.where(truth == BENIGN, "False positive", "Confirmed threat")
    rows = feedback_rows(members, decision, truth, note=SIMULATED_NOTE, source="test-set")
    log.info("Simulated analyst reviewed %d incidents -> %d labelled flows (%d false positives)",
             len(chosen), len(rows), int((decision == "False positive").sum()))
    return rows
