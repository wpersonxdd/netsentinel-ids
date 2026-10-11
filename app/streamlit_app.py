"""NetSentinel analyst console: alert triage, model performance, novel attacks and drift.

Run from the project root:  streamlit run app/streamlit_app.py

Every trained run is offered in the sidebar: the synthetic demo (artifacts/, reports/) and each named run such as
CIC-IDS2017 (artifacts/cicids2017/, reports/cicids2017/, configs/cicids2017.yaml).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .`

from nids import BENIGN, UNKNOWN  # noqa: E402
from nids.bundle import ModelBundle  # noqa: E402
from nids.config import load_config  # noqa: E402
from nids.feedback import DECISIONS, META_COLUMNS, NEW_FAMILY, append_feedback, feedback_rows  # noqa: E402
from nids.models import MODEL_LABELS, STRATEGY_LABELS  # noqa: E402
from nids.triage import group_incidents  # noqa: E402

MODEL_FILE = "netsentinel_model.joblib"
RUN_LABELS = {"cicids2017": "CIC-IDS2017 (real traffic)"}


def discover_runs() -> dict[str, dict[str, Path]]:
    """Trained runs: named sub-folders of artifacts/ first (real data), then the synthetic demo."""
    runs: dict[str, dict[str, Path]] = {}
    artifacts = ROOT / "artifacts"
    if artifacts.exists():
        for d in sorted(p for p in artifacts.iterdir() if p.is_dir() and (p / MODEL_FILE).exists()):
            config = ROOT / "configs" / f"{d.name}.yaml"
            runs[RUN_LABELS.get(d.name, d.name)] = {
                "artifacts": d, "reports": ROOT / "reports" / d.name,
                "config": config if config.exists() else ROOT / "configs" / "default.yaml",
                "demo": ROOT / "data" / "raw" / f"demo_{d.name}.csv"}
    if (artifacts / MODEL_FILE).exists():
        runs["Synthetic demo data"] = {"artifacts": artifacts, "reports": ROOT / "reports",
                                       "config": ROOT / "configs" / "default.yaml",
                                       "demo": ROOT / "data" / "raw" / "demo_new_traffic.csv"}
    return runs

PRIORITY_BADGE = {"P1": "🔴 P1", "P2": "🟠 P2", "P3": "🟡 P3", "P4": "⚪ P4"}

st.set_page_config(page_title="NetSentinel alert triage", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model…")
def load_bundle(path: str) -> ModelBundle:
    return ModelBundle.load(path)


@st.cache_data
def load_json(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_alerts(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Scoring traffic…")
def score_traffic(raw_bytes: bytes, model_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = load_bundle(model_path)
    return bundle.score_frame(pd.read_csv(io.BytesIO(raw_bytes), low_memory=False))


def load_feedback() -> pd.DataFrame:
    """Analyst decisions so far (metadata columns only; the feature columns are for retraining)."""
    if FEEDBACK_PATH.exists() and FEEDBACK_PATH.stat().st_size:
        return pd.read_csv(FEEDBACK_PATH, usecols=lambda c: c in META_COLUMNS, low_memory=False)
    return pd.DataFrame(columns=META_COLUMNS)


def project_config() -> dict:
    """The project config with paths made absolute, so the console works from any working directory."""
    cfg = load_config(CONFIG_PATH)
    cfg["paths"] = {k: str(ROOT / v) if not Path(v).is_absolute() else v for k, v in cfg["paths"].items()}
    if not Path(cfg["data"]["path"]).is_absolute():
        cfg["data"]["path"] = str(ROOT / cfg["data"]["path"])
    cfg["data"]["cache_dir"] = str(ROOT / cfg["data"].get("cache_dir", "data/processed"))
    return cfg


def pct(v, digits: int = 1) -> str:
    return "–" if v is None else f"{100 * v:.{digits}f}%"


def num(v, digits: int = 3) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def figure(name: str, caption: str | None = None) -> None:
    path = FIGURES / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


# --------------------------------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------------------------------

RUNS = discover_runs()
if not RUNS:
    st.title("🛡️ NetSentinel")
    st.warning("No trained model yet. From the project root run:")
    st.code("python -m nids all", language="bash")
    st.stop()

with st.sidebar:
    st.title("🛡️ NetSentinel")
    st.caption("Imbalance-aware network intrusion detection")
    run_name = st.selectbox("Dataset / trained run", list(RUNS))

RUN = RUNS[run_name]
ARTIFACTS, REPORTS, CONFIG_PATH, DEMO_TRAFFIC = RUN["artifacts"], RUN["reports"], RUN["config"], RUN["demo"]
FIGURES = REPORTS / "figures"
MODEL_PATH = ARTIFACTS / MODEL_FILE
ALERTS_PATH = ARTIFACTS / "alerts_test.csv"
FEEDBACK_PATH = ARTIFACTS / "analyst_feedback.csv"
SCORED_KEY, RETRAIN_KEY = f"scored::{run_name}", f"retrain_result::{run_name}"

bundle = load_bundle(str(MODEL_PATH))
meta = bundle.metadata

with st.sidebar:
    source = st.radio("Alert source", ["Test-set alerts", "Score a traffic file"], horizontal=False)
    alerts = pd.DataFrame()
    source_name = "test-set"
    if source == "Test-set alerts":
        if ALERTS_PATH.exists():
            alerts = load_alerts(str(ALERTS_PATH))
        else:
            st.info("Run `python -m nids train` to create the test-set alert queue.")
    else:
        upload = st.file_uploader("Flow records (CSV, any supported schema)", type=["csv"])
        use_demo = DEMO_TRAFFIC.exists() and st.button("Use bundled demo traffic", width="stretch")
        raw = upload.getvalue() if upload else (DEMO_TRAFFIC.read_bytes() if use_demo else None)
        if raw is not None:
            st.session_state[SCORED_KEY] = (score_traffic(raw, str(MODEL_PATH)),
                                            upload.name if upload else DEMO_TRAFFIC.name)
        if SCORED_KEY in st.session_state:
            (alerts, flows), source_name = st.session_state[SCORED_KEY]
            st.success(f"{len(flows):,} flows scored → {len(alerts):,} alerts")

    st.divider()
    st.subheader("Filters")
    if len(alerts):
        priorities = st.multiselect("Priority", sorted(alerts["priority"].unique()),
                                    default=sorted(alerts["priority"].unique()))
        classes = st.multiselect("Predicted class", sorted(alerts["predicted_class"].unique()),
                                 default=sorted(alerts["predicted_class"].unique()))
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        review_only = st.toggle("Needs review only")
        hide_done = st.toggle("Hide alerts already reviewed")
        ip_query = st.text_input("Source / destination IP contains")
    st.divider()
    st.caption(
        f"Model: **{MODEL_LABELS.get(meta.get('model'), '?')} + "
        f"{STRATEGY_LABELS.get(meta.get('strategy'), '?')}**  \n"
        f"Trained: {str(meta.get('trained_at', '?'))[:16].replace('T', ' ')} UTC  \n"
        f"Test macro-F1: {num(meta.get('test_macro_f1'))}"
    )

tab_queue, tab_model, tab_novel = st.tabs(["🚨 Alert queue", "📈 Model performance", "🧪 Novel attacks & drift"])

# --------------------------------------------------------------------------------------------------
# alert queue
# --------------------------------------------------------------------------------------------------

with tab_queue:
    if alerts.empty:
        st.info("No alerts to show. Pick an alert source in the sidebar.")
    else:
        feedback = load_feedback()
        # Alert IDs restart for every scored file, so decisions are matched on (source, alert_id).
        reviewed = (feedback[feedback["source"].astype(str) == source_name]
                    .drop_duplicates("alert_id", keep="last").set_index("alert_id")["decision"])
        view = alerts.copy()
        view["status"] = view["alert_id"].map(reviewed).fillna("new")
        mask = (view["priority"].isin(priorities) & view["predicted_class"].isin(classes)
                & (view["confidence"] >= min_conf))
        if review_only:
            mask &= view["needs_review"].astype(bool)
        if hide_done:
            mask &= view["status"] == "new"
        if ip_query:
            ip_cols = [c for c in ("src_ip", "dst_ip") if c in view.columns]
            mask &= view[ip_cols].astype(str).apply(lambda col: col.str.contains(ip_query, regex=False)).any(axis=1)
        view = view[mask].reset_index(drop=True)
        incidents, view = group_incidents(view.drop(columns=["incident_id"], errors="ignore"))
        if len(incidents):
            member_status = view.groupby("incident_id")["status"].agg(
                lambda s: "new" if (s == "new").all() else (s[s != "new"].iloc[-1] if (s != "new").all() else "partly reviewed"))
            incidents["status"] = incidents["incident_id"].map(member_status)

        if "flash" in st.session_state:
            st.success(st.session_state.pop("flash"))

        mode = st.radio("View", ["Incidents", "Flow alerts"], horizontal=True, label_visibility="collapsed",
                        help="Incidents group flow alerts that share an attack family, an endpoint and a time window")
        unit = incidents if mode == "Incidents" else view
        k = st.columns(6)
        k[0].metric("Incidents", f"{len(incidents):,}", help="Flow alerts grouped by family, shared endpoint and time")
        k[1].metric("🔴 P1 incidents", f"{(incidents['priority'] == 'P1').sum() if len(incidents) else 0:,}")
        k[2].metric("Flow alerts", f"{len(view):,}", help=f"{len(alerts):,} before filters")
        k[3].metric("Needs review", f"{view['needs_review'].astype(bool).sum():,}", help="Flow alerts")
        k[4].metric("Unknown-Anomaly", f"{(view['predicted_class'] == UNKNOWN).sum():,}",
                    help="Benign-looking to the classifier, but unlike any benign traffic seen in training")
        k[5].metric("Reviewed", f"{(view['status'] != 'new').sum():,}", help="Flow alerts with an analyst decision")

        if len(unit):
            left, right = st.columns(2)
            left.caption(f"{mode} by predicted class")
            left.bar_chart(unit["predicted_class"].value_counts(), height=200)
            right.caption(f"{mode} by priority")
            right.bar_chart(unit["priority"].value_counts().sort_index(), height=200)

        def explanation_block(row: pd.Series, title: str) -> None:
            if "possible_families" in row.index and pd.notna(row["possible_families"]) and bundle.conformal is not None:
                st.markdown(f"**Possible families:** {row['possible_families']}  \n"
                            f"<small>Conformal prediction set: for every family, the true one is in such sets for "
                            f"≥ {1 - bundle.conformal.alpha:.0%} of flows.</small>", unsafe_allow_html=True)
            why, act = st.columns([3, 2])
            with why:
                st.markdown(f"**{title}**")
                for part in str(row["explanation"]).split("; "):
                    st.markdown(f"- {part}")
                if row["predicted_class"] == UNKNOWN:
                    st.markdown("- The classifier saw benign traffic, but the novelty detector found this traffic far "
                                "from every benign pattern in training: a possible new attack.")
            with act:
                st.markdown("**Recommended first response**")
                st.info(row["recommended_action"])

        def decision_form(key: str, rows: pd.DataFrame, predicted_class: str) -> None:
            """One decision labels every flow in ``rows``; each becomes a training example for `nids retrain`."""
            with st.form(f"decision_{key}", clear_on_submit=True):
                scope = f" Applies to all {len(rows):,} flows of this incident." if len(rows) > 1 else ""
                st.markdown(f"**Analyst decision**: saved decisions become labelled training data.{scope}")
                d1, d2 = st.columns(2)
                decision = d1.selectbox("Decision", DECISIONS)
                others = [c for c in bundle.classes if c not in (predicted_class, BENIGN)]
                if predicted_class == UNKNOWN:  # the analyst names the family (or confirms it is benign)
                    options = [NEW_FAMILY, BENIGN] + others
                else:
                    options = [predicted_class, BENIGN] + others + [NEW_FAMILY]
                corrected = d2.selectbox("Correct label", options)
                note = st.text_input("Note", placeholder=f"For “{NEW_FAMILY}”, type the family name here")
                if st.form_submit_button("Save decision", type="primary"):
                    append_feedback(feedback_rows(rows, decision, corrected, note, source=source_name), FEEDBACK_PATH)
                    st.session_state["flash"] = f"Saved “{decision}” for {key} ({len(rows):,} flow alerts)."
                    st.rerun()

        if mode == "Incidents" and len(incidents):
            table = incidents.copy()
            table["priority"] = table["priority"].map(PRIORITY_BADGE).fillna(table["priority"])
            st.caption("Select an incident to investigate it.")
            event = st.dataframe(
                table[["incident_id", "priority", "predicted_class", "flows", "shared_endpoint", "first_seen",
                       "last_seen", "mean_confidence", "needs_review", "status"]],
                hide_index=True, width="stretch", height=360, on_select="rerun", selection_mode="single-row",
                key="incident_queue",
                column_config={
                    "incident_id": "Incident", "priority": "Priority", "predicted_class": "Family",
                    "flows": st.column_config.NumberColumn("Flows", format="%d"),
                    "shared_endpoint": "Shared endpoint",
                    "first_seen": st.column_config.DatetimeColumn("First seen", format="YYYY-MM-DD HH:mm"),
                    "last_seen": st.column_config.DatetimeColumn("Last seen", format="YYYY-MM-DD HH:mm"),
                    "mean_confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0,
                                                                       format="%.2f"),
                    "needs_review": st.column_config.CheckboxColumn("Review"),
                },
            )
            rows = event.selection.rows if event and event.selection else []
            inc = incidents.iloc[rows[0] if rows else 0]
            members = view[view["incident_id"] == inc["incident_id"]]
            st.divider()
            st.subheader(f"{PRIORITY_BADGE.get(inc['priority'], inc['priority'])} · {inc['incident_id']} · "
                         f"{inc['predicted_class']}")
            c = st.columns(4)
            c[0].metric("Flows", f"{int(inc['flows']):,}")
            c[1].metric("Severity", str(inc["severity"]).title())
            c[2].metric("Mean confidence", num(inc["mean_confidence"], 2))
            span = (pd.to_datetime(inc["last_seen"]) - pd.to_datetime(inc["first_seen"]))
            c[3].metric("Duration", f"{span.total_seconds() / 60:,.0f} min" if pd.notna(span) else "–")
            endpoint_role = "target" if inc["grouped_by"] == "dst_ip" else "source"
            st.caption(f"Shared {endpoint_role}: **{inc['shared_endpoint']}**  ·  sources: {inc['sources']}  ·  "
                       f"destinations: {inc['destinations']}")
            if "true_label" in members.columns:
                truth = members["true_label"].value_counts()
                st.caption("Ground truth (evaluation data only): "
                           + ", ".join(f"{k} × {v}" for k, v in truth.items()))
            explanation_block(inc, f"Why: evidence from the strongest flow ({inc['top_alert']})")
            with st.expander(f"Member flows ({len(members):,})"):
                cols = [c for c in ["alert_id", "priority", "confidence", "timestamp", "src_ip", "dst_ip", "dst_port",
                                    "explanation", "status"] if c in members.columns]
                st.dataframe(members[cols].head(500), hide_index=True, width="stretch")
            decision_form(inc["incident_id"], members, inc["predicted_class"])

        elif mode == "Flow alerts" and len(view):
            table = view.copy()
            table["priority"] = table["priority"].map(PRIORITY_BADGE).fillna(table["priority"])
            columns = [c for c in ["alert_id", "priority", "predicted_class", "possible_families", "confidence",
                                   "anomaly_score", "severity", "timestamp", "src_ip", "dst_ip", "dst_port",
                                   "incident_id", "needs_review", "status"] if c in table.columns]
            st.caption("Select a row to investigate it.")
            event = st.dataframe(
                table[columns], hide_index=True, width="stretch", height=360, on_select="rerun",
                selection_mode="single-row", key="queue",
                column_config={
                    "alert_id": "Alert", "priority": "Priority", "predicted_class": "Predicted class",
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0,
                                                                  format="%.2f"),
                    "anomaly_score": st.column_config.NumberColumn("Anomaly score", format="%.2f"),
                    "needs_review": st.column_config.CheckboxColumn("Review"),
                    "dst_port": st.column_config.NumberColumn("Port", format="%d"),
                    "incident_id": "Incident", "possible_families": "Possible families",
                },
            )
            rows = event.selection.rows if event and event.selection else []
            alert = view.iloc[rows[0] if rows else 0]
            st.divider()
            st.subheader(f"{PRIORITY_BADGE.get(alert['priority'], alert['priority'])} · {alert['alert_id']} · "
                         f"{alert['predicted_class']}")
            c = st.columns(4)
            c[0].metric("Severity", str(alert["severity"]).title())
            c[1].metric("Model confidence", num(alert["confidence"], 2))
            c[2].metric("Anomaly score", num(alert["anomaly_score"], 2))
            c[3].metric("Status", str(alert["status"]))
            route = " → ".join(str(alert[col]) for col in ("src_ip", "dst_ip") if col in alert.index)
            if "dst_port" in alert.index and pd.notna(alert["dst_port"]):
                route += f":{int(alert['dst_port'])}"
            if "timestamp" in alert.index:
                route = f"{alert['timestamp']}  ·  {route}  ·  part of {alert['incident_id']}"
            st.caption(route)
            if "true_label" in alert.index and pd.notna(alert["true_label"]):
                st.caption(f"Ground truth (evaluation data only): **{alert['true_label']}**")
            explanation_block(alert, "Why this alert")
            decision_form(alert["alert_id"], view.iloc[[rows[0] if rows else 0]], alert["predicted_class"])

        st.divider()
        fb = load_feedback()
        st.subheader("🔁 Close the loop")
        usable = fb[~fb["decision"].isin(["Needs more data"])] if len(fb) else fb
        st.markdown(
            f"**{len(fb):,}** flow-level decisions saved ({fb['incident_id'].nunique() if len(fb) else 0:,} incidents, "
            f"{len(usable):,} with a usable label). Retraining adds them to the original training data, refits the "
            "classifier and the novelty detector, compares old and new models on test flows nobody reviewed, then "
            "deploys the new model and archives the old one.")
        r1, r2, r3 = st.columns(3)
        if r1.button("Retrain model with analyst feedback", type="primary", width="stretch", disabled=usable.empty):
            from nids.experiments import run_retrain

            with st.spinner("Retraining… (about a minute)"):
                result = run_retrain(project_config(), FEEDBACK_PATH, MODEL_PATH)
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state[RETRAIN_KEY] = result
            st.rerun()
        r2.download_button("⬇ Analyst feedback (CSV, with features)",
                           FEEDBACK_PATH.read_bytes() if FEEDBACK_PATH.exists() else b"",
                           file_name="analyst_feedback.csv", mime="text/csv", width="stretch", disabled=fb.empty)
        r3.download_button("⬇ Filtered alerts (CSV)", view.to_csv(index=False).encode("utf-8"),
                           file_name="netsentinel_alerts.csv", mime="text/csv", width="stretch")
        if RETRAIN_KEY in st.session_state:
            res = st.session_state[RETRAIN_KEY]
            before, after = res["metrics"]["before"], res["metrics"]["after"]
            st.success(f"Retrained and deployed. Previous model archived to `{res['artifacts']['archived_previous']}`. "
                       f"Compared on {res['evaluation']['evaluated_flows']:,} test flows no analyst reviewed:")
            c = st.columns(4)
            c[0].metric("Macro-F1", num(after["macro_f1"]), f"{after['macro_f1'] - before['macro_f1']:+.3f}")
            if after.get("rare_macro_f1") is not None and before.get("rare_macro_f1") is not None:
                c[1].metric("Rare-class F1", num(after["rare_macro_f1"]),
                            f"{after['rare_macro_f1'] - before['rare_macro_f1']:+.3f}")
            c[2].metric("False-alarm rate", pct(after["false_alarm_rate"], 2),
                        f"{100 * (after['false_alarm_rate'] - before['false_alarm_rate']):+.2f} pp",
                        delta_color="inverse")
            vb, va = res["alert_volume"]["before"], res["alert_volume"]["after"]
            c[3].metric("Incidents", f"{va['incidents']:,}", f"{va['incidents'] - vb['incidents']:+,}",
                        delta_color="inverse")
            if res["new_families"]:
                st.info("New attack families learned from analyst labels: " + ", ".join(res["new_families"]))
            st.caption("The alert queue above still shows the previous model's alerts; score traffic again to see "
                       "the new model's queue.")

# --------------------------------------------------------------------------------------------------
# model performance
# --------------------------------------------------------------------------------------------------

with tab_model:
    final = load_json(str(REPORTS / "results" / "final.json"))
    if not final:
        st.info("Run `python -m nids train` to produce evaluation results.")
    else:
        base, clf, hyb = (final["metrics"][k] for k in ("baseline", "classifier", "hybrid"))
        detector = "Local Outlier Factor" if final.get("novelty_method", "lof") == "lof" else "Isolation Forest"
        compare = final.get("comparison_strategy", "none")
        compare_text = "no imbalance handling" if compare == "none" else STRATEGY_LABELS[compare]
        compare_col = f"F1, {compare_text}"
        st.markdown(f"**Deployed:** {MODEL_LABELS[final['model']]} + {STRATEGY_LABELS[final['strategy']]} with a "
                    f"{detector} novelty detector. Metrics on the held-out test set "
                    f"({final['dataset']['split']['test']:,} flows); deltas vs. the same model with {compare_text}.")
        m = st.columns(5)
        m[0].metric("Macro-F1", num(hyb["macro_f1"]), f"{hyb['macro_f1'] - base['macro_f1']:+.3f}")
        if hyb.get("rare_macro_f1") is not None and base.get("rare_macro_f1") is not None:
            m[1].metric("Rare-class F1", num(hyb["rare_macro_f1"]),
                        f"{hyb['rare_macro_f1'] - base['rare_macro_f1']:+.3f}")
        m[2].metric("Detection rate", pct(hyb["detection_rate"]),
                    f"{100 * (hyb['detection_rate'] - base['detection_rate']):+.1f} pp")
        m[3].metric("False-alarm rate", pct(hyb["false_alarm_rate"], 2),
                    f"{100 * (hyb['false_alarm_rate'] - base['false_alarm_rate']):+.2f} pp", delta_color="inverse")
        m[4].metric("Throughput", f"{1e6 / final['predict_ms_per_1k_flows']:,.0f} flows/s")

        per_class = pd.DataFrame(clf["per_class"]).T
        per_class[compare_col] = pd.DataFrame(base["per_class"]).T["f1"]
        st.dataframe(
            per_class[["support", "precision", "recall", "f1", "pr_auc", compare_col]],
            width="stretch",
            column_config={
                "support": st.column_config.NumberColumn("Test flows", format="%d"),
                "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
                compare_col: st.column_config.NumberColumn(format="%.3f"),
            },
        )
        a, b = st.columns(2)
        with a:
            figure("per_class_f1.png")
            figure("benchmark_macro_f1.png")
            figure("feature_importance.png")
        with b:
            figure("confusion_matrix.png")
            figure("benchmark_rare_f1.png")
            figure("class_distribution.png")
        figure("pr_curves.png")

# --------------------------------------------------------------------------------------------------
# novel attacks and drift
# --------------------------------------------------------------------------------------------------

with tab_novel:
    novelty = load_json(str(REPORTS / "results" / "novelty.json"))
    drift = load_json(str(REPORTS / "results" / "drift.json"))
    st.subheader("Attack families never seen in training")
    if novelty and novelty["rows"]:
        st.markdown(
            f"Each family is removed from training in turn. Mean share of its flows still raised as an alert: "
            f"**{pct(novelty['mean_classifier_detection'])} classifier only → "
            f"{pct(novelty['mean_hybrid_detection'])} with the novelty detector**.")
        a, b = st.columns([3, 2])
        with a:
            figure("novel_attack_detection.png")
        with b:
            st.dataframe(pd.DataFrame(novelty["rows"])[
                ["held_out_family", "classifier_detection", "hybrid_detection", "most_common_label_given"]],
                hide_index=True, width="stretch",
                column_config={
                    "held_out_family": "Held-out family",
                    "classifier_detection": st.column_config.ProgressColumn("Classifier", format="%.2f",
                                                                            min_value=0.0, max_value=1.0),
                    "hybrid_detection": st.column_config.ProgressColumn("+ novelty", format="%.2f",
                                                                        min_value=0.0, max_value=1.0),
                    "most_common_label_given": "Labelled as",
                })
    else:
        st.info("Run `python -m nids novelty`.")

    st.subheader("Drift over time")
    if drift and not drift.get("skipped"):
        dm = drift["metrics"]
        c = st.columns(4)
        c[0].metric("Macro-F1, history", num(dm["past_validation"]["macro_f1"]))
        c[1].metric("Recent, static model", num(dm["future_static"]["macro_f1"]),
                    f"{dm['future_static']['macro_f1'] - dm['past_validation']['macro_f1']:+.3f}")
        c[2].metric("Recent, retrained", num(dm["future_retrained"]["macro_f1"]),
                    f"{dm['future_retrained']['macro_f1'] - dm['future_static']['macro_f1']:+.3f}")
        c[3].metric("Retraining trigger", "🔴 fired" if drift["retraining_triggered"] else "🟢 quiet")
        fired = [k.replace("_", " ") for k, v in drift["triggers"].items() if v]
        st.caption("Triggers: " + (", ".join(fired) if fired else "none"))
        a, b = st.columns(2)
        with a:
            figure("drift_psi.png")
        with b:
            figure("drift_recall.png")
    elif drift:
        st.info(f"Drift experiment skipped: {drift['skipped']}")
    else:
        st.info("Run `python -m nids drift`.")
