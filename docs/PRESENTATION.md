# Presentation outline, demo script and judge Q&A

Target: **10 slides, 6–7 minutes + 3-minute live demo**. Every number below comes from `reports/REPORT.md`.
Regenerate it on the official data and update the slides from it.

## Slides

| # | Title | Content | Visual |
|---|---|---|---|
| 1 | **NetSentinel** | One line: *imbalance-aware intrusion detection that notices what it was never taught.* Name, DN-CY-01 | Logo / shield |
| 2 | The problem is not "a classifier" | 4 sub-problems: imbalance (largest:smallest class ≈ 560:1), unseen attacks, drift, analysts drowning in alerts. Show that "always benign" scores ~80% accuracy and 0% detection | `class_distribution.png` |
| 3 | Data and leakage guards | Format-agnostic loader; label families; IPs/IDs/timestamps removed from features; duplicates removed before split; resampling inside training folds only | Pipeline diagram (README mermaid) |
| 4 | 20 configurations, one honest winner | 5 models × 4 imbalance strategies on validation; winner picked by macro-F1; test touched once | `benchmark_macro_f1.png` |
| 5 | Minority classes, before vs. after | Same model, same test set, with and without imbalance handling; rare-class F1 gain | `per_class_f1.png` + headline table |
| 6 | Attacks it has never seen | Leave-one-attack-out; LOF vs Isolation Forest vs autoencoder at the same 1% false-alarm budget | `novel_attack_detection.png` + detector table from METHODOLOGY §5 |
| 7 | Attackers change tactics | Train on past / test on future: static vs retrained; label-free drift signals; why whole-traffic PSI misses it | `drift_psi.png`, `drift_recall.png` |
| 8 | From prediction to action | Thousands of flow alerts → a much shorter list of incidents (REPORT §5); P1–P4, "why" explanations, playbooks; one analyst decision labels a whole incident | Console screenshot |
| 8b | The loop closes | Decisions are stored with flow features → `nids retrain` / one button → before/after on test flows nobody reviewed (REPORT §8); old model archived | `retrain_per_class_f1.png` |
| 9 | Production-ready | Throughput (flows/s), one-command pipeline, CLI scoring, tests + CI, Docker, config-driven | Small architecture diagram |
| 10 | Limits & next steps | Flow-only features (pair with Zeek/Suricata), benign-mimicking attacks, temporal splits; next: streaming, adversarial robustness, active learning | Short list |

**Storyline in one breath:** *Accuracy lies on imbalanced traffic → we measured what actually helps rare attacks →
we catch part of what the model never saw → we notice when it goes stale → and we hand analysts a short, explained,
prioritised queue that feeds the next retraining.*

## 3-minute demo script

1. **(20 s)** Terminal: `python -m nids all --skip-benchmark` already run; show `reports/REPORT.md` headline table.
2. **(60 s)** `streamlit run app/streamlit_app.py` → **Alert queue**, *Incidents* view:
   - KPI row: incidents vs. flow alerts (the reduction is the point), P1, Unknown-Anomaly.
   - Click the top **P1 DDoS** incident: many sources, one shared target, duration, explanation
     (`conn_count_same_dst_10s` far above the benign median), playbook, member flows.
   - Filter class = `Unknown-Anomaly`: explain "classifier said benign, novelty detector disagreed".
   - Save a decision ("Confirmed threat") on the incident → every member flow becomes retraining data.
   - Scroll to **Close the loop** → *Retrain model with analyst feedback* (≈1 min; cut the wait in the video) →
     before/after metrics on unreviewed flows, old model archived.
3. **(40 s)** Sidebar → **Score a traffic file** → *Use bundled demo traffic*: unseen traffic scored live, new queue.
4. **(30 s)** **Model performance** tab: per-class before/after, confusion matrix.
5. **(30 s)** **Novel attacks & drift** tab: leave-one-attack-out bars; drift trigger fired; retrained recovery.

Record it with OBS or Xbox Game Bar (Win+Alt+R), 1080p, with voice-over.

## Likely judge questions

| Question | Answer |
|---|---|
| Why not accuracy? | With ~80% benign traffic, predicting "benign" always is ~80% accurate and useless. Macro-F1 and rare-class F1 weight every family equally; PR-AUC is not inflated by the benign majority. |
| Doesn't SMOTE leak into the test set? | No. It is a step of an imblearn pipeline, so it runs only in `fit` on training folds; validation and test keep the real distribution. |
| Why did you pick this model? | Not by hand: the benchmark picks the best validation macro-F1 among 20 configurations; the test set is used once. |
| How do you know the gain comes from imbalance handling and not the model? | The final model is compared with *the same model* trained without imbalance handling on the same test set. |
| Why LOF and not Isolation Forest / an autoencoder? | Measured at the same 1% false-alarm budget: mean detection of unseen families LOF 59%, autoencoder 51%, Isolation Forest 31% (METHODOLOGY §5). Benign traffic is multi-modal (DNS, HTTPS, SMB…); LOF compares each flow with its *local* neighbourhood. The autoencoder wins on botnet beacons, so combining the two is our next step. |
| What can't it detect? | Attacks that look statistically like benign sessions at flow level (e.g. web attacks inside normal HTTP traffic). That needs payload inspection; we report it instead of hiding it. |
| How would this run in production? | `nids score` on flow logs (CICFlowMeter/Zeek) in batches or from a stream consumer; throughput is in the report. The drift monitor runs without labels; analyst decisions from the console are the labels for retraining. |
| How do analyst decisions improve the model? | Each decision is stored with the flow's features, so it is a labelled example. `nids retrain` adds them to the training data, refits the classifier and novelty detector, and compares old vs. new on test flows nobody reviewed (reviewed flows are excluded, since they are training data now). REPORT §8 shows the effect of one simulated analyst-day. |
| Isn't the IP address a useful feature? | It is a shortcut: it identifies the lab's attacker host, not attack behaviour, and does not transfer to a new network. IPs are kept only as context for analysts. |
| Results on synthetic data? | Synthetic data was only used before the official release to build and test the pipeline; the submitted numbers are on the official data (the report states its source). |
| Why random split if attacks are sessions? | Random stratified splits are standard but optimistic; the temporal experiment is the realistic estimate, and we show both. |
