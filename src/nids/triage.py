"""Turning predictions into a prioritised, explained alert queue for SOC analysts.

Every non-benign prediction becomes an alert with
* severity       - from the attack family (config ``triage.severity``),
* priority P1-P4 - severity weighted by model confidence, boosted when the flow is also anomalous,
* explanation    - the features that deviate most from the benign baseline, in analyst language
                   ("conn_count_same_dst_10s=148, 49 IQR above the benign median of 3"),
* playbook       - a recommended first response for that family,
* possible_families - the conformal prediction set: every family plausible at the guaranteed coverage level,
* needs_review   - low confidence, a conformal set with several families, or an unknown family: human eyes first,
* feat_*         - the flow's feature values, so an analyst decision doubles as a training example.

Alerts are then grouped into incidents (``group_incidents``), the unit an analyst actually works on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nids import BENIGN, UNKNOWN

SEVERITY_WEIGHT = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25, "none": 0.0}
DEFAULT_SEVERITY = {
    BENIGN: "none",
    "PortScan": "low", "Probe": "low", "Reconnaissance": "low", "Fuzzers": "low",
    "Analysis": "medium", "BruteForce": "medium", "R2L": "medium", "Generic": "medium",
    "DoS": "high", "WebAttack": "high", "Exploits": "high", UNKNOWN: "low",
    "DDoS": "critical", "Botnet": "critical", "Infiltration": "critical", "Exfiltration": "critical",
    "Exploit": "critical", "Backdoor": "critical", "Shellcode": "critical", "Worms": "critical", "U2R": "critical",
}
# Port numbers are identifiers, not magnitudes: "port 31337 is 85 IQR above the median" explains nothing.
NOT_EXPLAINABLE = {"destination_port", "dst_port", "dsport", "dport", "dstport"}
FEATURE_PREFIX = "feat_"  # alert columns holding the flow's feature values
PRIORITY_CUTS = [(0.70, "P1"), (0.45, "P2"), (0.25, "P3")]
VOLUME_ESCALATION = (20, 0.15)  # incidents with >= 20 flows gain 0.15 priority score

_RECON = "Low priority on its own: watch-list the source and correlate with follow-up exploitation attempts."
PLAYBOOK = {
    "DoS": "Rate-limit or block the source at the edge; check web-server connection limits "
           "(slow attacks: shorten header/body timeouts).",
    "DDoS": "Engage upstream DDoS mitigation/scrubbing; enable SYN cookies; rate-limit by ASN/geo.",
    "PortScan": _RECON, "Probe": _RECON, "Reconnaissance": _RECON, "Fuzzers": _RECON,
    "BruteForce": "Throttle/lock the targeted account or service; enforce MFA and fail2ban; "
                  "check for a successful login right after the burst.",
    "R2L": "Check the targeted service for unauthorised access; reset exposed credentials.",
    "WebAttack": "Inspect HTTP payloads and WAF logs for injection/XSS; verify input validation; "
                 "review database logs for unexpected queries.",
    "Botnet": "Isolate the internal host, block the C2 destination, run an EDR scan and hunt for "
              "other hosts with the same beaconing pattern.",
    "Infiltration": "Treat the source as a compromised internal host: isolate it, preserve memory/disk, "
                    "review lateral-movement paths.",
    "Exfiltration": "Block the destination, isolate the source host and estimate what data left the network.",
    "Exploit": "Patch or isolate the targeted service; check it for signs of code execution.",
    "Exploits": "Patch or isolate the targeted service; check it for signs of code execution.",
    "Backdoor": "Isolate the host; look for persistence mechanisms and unknown listening services.",
    "U2R": "Privilege escalation suspected: isolate the host and audit privileged accounts.",
    UNKNOWN: "Behaviour unlike any benign traffic and unlike known attack families: investigate manually, "
             "then label it so the next retraining learns it.",
}
DEFAULT_ACTION = "Investigate the flow and its source host; label the outcome for retraining."


def _fmt(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.1f}G"
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e4:
        return f"{v / 1e3:.0f}k"
    return f"{v:.3g}"


class BenignBaseline:
    """Robust per-feature statistics of benign traffic, used to explain why a flow looks suspicious."""

    def fit(self, X_benign: pd.DataFrame) -> "BenignBaseline":
        X = X_benign.select_dtypes(include="number")
        X = X[[c for c in X.columns if c not in NOT_EXPLAINABLE]]
        self.columns_ = list(X.columns)
        self.median_ = X.median()
        iqr = X.quantile(0.75) - X.quantile(0.25)
        fallback = X.std().where(X.std() > 0, self.median_.abs() * 0.1)
        self.scale_ = iqr.where(iqr > 0, fallback).replace(0, 1.0).fillna(1.0)
        return self

    def deviations(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X[self.columns_] - self.median_) / self.scale_

    def explain(self, X: pd.DataFrame, top_k: int = 3) -> list[str]:
        if len(X) == 0:
            return []
        Z = self.deviations(X).to_numpy(dtype=float)
        V = X[self.columns_].to_numpy(dtype=float)
        top = np.argsort(-np.nan_to_num(np.abs(Z)), axis=1)[:, :top_k]
        out = []
        for i, cols in enumerate(top):
            parts = []
            for j in cols:
                z = Z[i, j]
                if not np.isfinite(z) or abs(z) < 1:
                    continue
                name = self.columns_[j]
                direction = "above" if z > 0 else "below"
                parts.append(f"{name}={_fmt(V[i, j])} ({abs(z):.0f} IQR {direction} benign median "
                             f"{_fmt(self.median_.iloc[j])})")
            out.append("; ".join(parts) if parts else "No single feature stands out; pattern-level detection.")
        return out


def priority_label(score: np.ndarray) -> np.ndarray:
    out = np.full(len(score), "P4", dtype=object)
    for cut, name in reversed(PRIORITY_CUTS):
        out[score >= cut] = name
    return out


def build_alerts(prediction: dict, X: pd.DataFrame, context: pd.DataFrame, baseline: BenignBaseline,
                 severity_map: dict, novelty_threshold: float, conf_threshold: float = 0.6,
                 y_true: pd.Series | None = None) -> pd.DataFrame:
    """Alert queue (one row per non-benign prediction), highest priority first."""
    labels = np.asarray(prediction["label"]).astype(str)
    mask = labels != BENIGN
    if not mask.any():
        return pd.DataFrame(columns=["alert_id", "priority", "predicted_class"])

    lab = labels[mask]
    score = prediction["anomaly_score"][mask]
    novel = prediction["novel"][mask]
    conf = prediction["confidence"][mask].copy()
    # For unknown-family alerts the classifier's confidence is about "benign"; use anomaly strength instead.
    unknown = lab == UNKNOWN
    # Scale-free strength (works for LOF ratios and Isolation Forest scores): 0 at the threshold,
    # 0.5 at twice the threshold, approaching 1 for extreme outliers.
    strength = np.clip(1 - novelty_threshold / np.maximum(score, 1e-9), 0, 1)
    conf[unknown] = 0.5 + 0.5 * strength[unknown]

    severity = np.array([severity_map.get(c, "medium") for c in lab], dtype=object)
    weight = np.array([SEVERITY_WEIGHT.get(s, 0.5) for s in severity])
    pscore = np.clip(weight * (0.5 + 0.5 * conf) + 0.1 * novel, 0, 1)

    # Needs review: low confidence, OR (with conformal sets) more than one family plausible at the guaranteed
    # coverage, OR an unknown family. The two uncertainty signals are complementary: per-family conformal thresholds
    # differ widely, so a 50/50 flow can still get a one-family set; a confident flow can still be ambiguous between
    # two families whose thresholds are loose.
    uncertain = conf < conf_threshold
    if "set_size" in prediction:
        uncertain = uncertain | (prediction["set_size"][mask] != 1)
    ctx = context.iloc[np.flatnonzero(mask)].reset_index(drop=True)
    alerts = pd.DataFrame({
        "priority": priority_label(pscore),
        "priority_score": pscore.round(3),
        "predicted_class": lab,
        "severity": severity,
        "confidence": conf.round(3),
        "anomaly_score": score.round(3),
        "novel": novel,
        "needs_review": uncertain | unknown,
    })
    if "possible_families" in prediction:
        alerts["possible_families"] = np.asarray(prediction["possible_families"])[mask]
    for col in ("timestamp", "src_ip", "dst_ip", "dst_port"):
        if col in ctx.columns:
            alerts[col] = ctx[col].values
    positions = np.flatnonzero(mask)
    alerts["explanation"] = baseline.explain(X.iloc[positions])
    alerts["recommended_action"] = [PLAYBOOK.get(c, DEFAULT_ACTION) for c in lab]
    if y_true is not None:
        alerts["true_label"] = np.asarray(y_true)[mask]
    alerts["status"] = "new"
    # The flow's own feature values travel with the alert, so an analyst decision on it is a complete
    # labelled training example (see nids.feedback). flow_index = row position in the scored input.
    alerts["flow_index"] = positions
    alerts = pd.concat([alerts, X.iloc[positions].reset_index(drop=True).add_prefix(FEATURE_PREFIX)], axis=1)
    alerts = alerts.sort_values(["priority_score", "confidence"], ascending=False, kind="stable").reset_index(drop=True)
    alerts.insert(0, "alert_id", [f"A-{i + 1:06d}" for i in range(len(alerts))])
    return alerts


def _summarise(values: pd.Series, limit: int = 3) -> str:
    uniq = values.dropna().astype(str).value_counts()
    if len(uniq) <= limit:
        return ", ".join(uniq.index)
    return f"{', '.join(uniq.index[:2])} … ({len(uniq):,} distinct)"


def group_incidents(alerts: pd.DataFrame, gap: str = "1h") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse flow-level alerts into incidents an analyst handles as one ticket.

    Alerts join an incident when they share the predicted family and the endpoint that family's alerts have in
    common: the fewest distinct values among source and destination IP. A DDoS shares its victim, a scan or a
    brute-force run shares its source, a botnet shares its C2 server. A quiet period longer than ``gap`` starts a
    new incident. Returns (incidents, alerts with an ``incident_id`` column).
    """
    if alerts.empty:
        return pd.DataFrame(), alerts.assign(incident_id=pd.Series(dtype=str))
    df = alerts.copy()
    df["_t"] = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df.columns else pd.NaT
    endpoints = [c for c in ("src_ip", "dst_ip") if c in df.columns]
    df["_by"], df["_endpoint"] = "none", "all"
    for cls, part in df.groupby("predicted_class"):
        if endpoints:
            col = min(endpoints, key=lambda c: part[c].nunique())
            df.loc[part.index, "_by"] = col
            df.loc[part.index, "_endpoint"] = part[col].astype(str)

    df = df.sort_values(["predicted_class", "_endpoint", "_t"], kind="stable")
    same_group = (df["predicted_class"].eq(df["predicted_class"].shift()) & df["_endpoint"].eq(df["_endpoint"].shift()))
    quiet = df["_t"].diff() > pd.Timedelta(gap)
    df["_incident"] = (~same_group | quiet.fillna(False)).cumsum()

    top = df.loc[df.groupby("_incident")["priority_score"].idxmax()].set_index("_incident")
    grouped = df.groupby("_incident")
    incidents = pd.DataFrame({
        "predicted_class": top["predicted_class"],
        "priority_score": top["priority_score"],
        "severity": top["severity"],
        "flows": grouped.size(),
        "grouped_by": top["_by"],
        "shared_endpoint": top["_endpoint"],
        "first_seen": grouped["_t"].min(),
        "last_seen": grouped["_t"].max(),
        "sources": grouped["src_ip"].agg(_summarise) if "src_ip" in df.columns else "",
        "destinations": grouped["dst_ip"].agg(_summarise) if "dst_ip" in df.columns else "",
        "mean_confidence": grouped["confidence"].mean().round(3),
        "max_anomaly_score": grouped["anomaly_score"].max().round(3),
        "needs_review": grouped["needs_review"].any(),
        "top_alert": top["alert_id"],
        "explanation": top["explanation"],
        "recommended_action": top["recommended_action"],
    })
    if "possible_families" in top.columns:
        incidents["possible_families"] = top["possible_families"]
    # Sustained activity escalates: an incident with many flows outranks a single flow of the same family.
    min_flows, boost = VOLUME_ESCALATION
    incidents["priority_score"] = np.clip(incidents["priority_score"] + boost * (incidents["flows"] >= min_flows),
                                          0, 1).round(3)
    incidents.insert(0, "priority", priority_label(incidents["priority_score"].to_numpy()))
    incidents = incidents.sort_values(["priority_score", "flows"], ascending=False, kind="stable").reset_index()
    incidents.insert(0, "incident_id", [f"INC-{i + 1:05d}" for i in range(len(incidents))])
    mapping = incidents.set_index("_incident")["incident_id"]
    out = alerts.copy()
    out["incident_id"] = df["_incident"].map(mapping).reindex(out.index).values
    return incidents.drop(columns="_incident"), out
