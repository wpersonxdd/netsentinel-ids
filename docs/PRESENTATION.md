# Presentation outline, demo script and judge Q&A

Finished deck: [`NetSentinel_slides.pdf`](NetSentinel_slides.pdf) (14 slides; sources and speaker notes in
`presentation/project/slides/`), narrated video: [`media/NetSentinel_demo.mp4`](media/NetSentinel_demo.mp4).
The outline below is the talk plan the deck was built from.

Target: **10 slides, 6–7 minutes + 3-minute live demo**. Lead with the **real-traffic** numbers
(`reports/cicids2017/REPORT.md`, CIC-IDS2017); use the synthetic report (`reports/REPORT.md`) as the stress test.
Regenerate both on the official data when it is released and update the slides from them.

## Slides

| # | Title | Content | Visual |
|---|---|---|---|
| 1 | **NetSentinel** | One line: *imbalance-aware intrusion detection that notices what it was never taught.* Name, DN-CY-01 | Logo / shield |
| 2 | The problem is not "a classifier" | 4 sub-problems: imbalance (largest:smallest class ≈ 560:1), unseen attacks, drift, analysts drowning in alerts. Show that "always benign" scores ~80% accuracy and 0% detection | `class_distribution.png` |
| 3 | Data and leakage guards | Format-agnostic loader; label families; IPs/IDs/timestamps removed from features; duplicates removed before split; resampling inside training folds only | Pipeline diagram (README mermaid) |
| 4 | 20 configurations, one honest winner | 5 models × 4 imbalance strategies on validation; winner picked by macro-F1; test touched once | `benchmark_macro_f1.png` |
| 5 | Minority classes, before vs. after | Same model, same test set, with and without imbalance handling; rare-class F1 gain | `per_class_f1.png` + headline table |
| 5a | Real traffic: CIC-IDS2017 | 2.8 M real flows, 8 attack families + benign; quirks repaired (12-hour clock, empty rows, duplicates). Headline: macro-F1 0.766 → 0.973, *but* on measurable families only +0.005: the jump is 9 flows of Heartbleed/Infiltration (2/2 and 5/7 found vs 0). Say it before the judges do | `reports/cicids2017/figures/per_class_f1.png` |
| 5b | How much to trust the numbers | 95% CIs on every headline metric; paired bootstrap says the gain is not luck; the same model under random / session-grouped / temporal splits (quote the temporal one); data audit found no shortcut | REPORT §3.1–3.2, `shortcut_scan.png` |
| 5c | New: a guarantee for every family | Mondrian conformal prediction: true family in the set for ≥ 90% of flows of *each* family; alerts show "possible families"; the price of guaranteeing recall for rare families, and why analyst labels lower it | `conformal_coverage.png` |
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
| Macro-F1 0.973 on CIC-IDS2017, isn't that too good? | The random split is optimistic, and we measured by how much: session-grouped 0.888, temporal 0.180. The all-family jump from 0.766 is 9 flows of the two tiniest families; on measurable families the gain is +0.005. We report all of it. |
| Why is the temporal score only 0.18? | Every CIC-IDS2017 attack happens on one day. Trained on Monday to Thursday morning, the classifier has never seen Friday's DDoS, PortScan or Botnet. That is the real limit of supervised IDS, and the reason for the novelty detector (DDoS 61% alerted before any label) and fast retraining (0.925 after 10% labels). |
| Is the improvement statistically significant? | Yes: paired stratified bootstrap over 1,000 resamples, +0.055 macro-F1 [+0.032, +0.084], +0.125 rare-class F1 [+0.072, +0.188], better in 100% of resamples. The false-alarm cost is also significant and we report it. |
| Random splits overestimate IDS results, don't they? | They can, which is why we measure it: a session-grouped split (no source-destination-hour session on both sides) and a temporal split. Here the session split shows no leakage; the temporal split (0.738) is the number we would quote for deployment. |
| What is conformal prediction and why "Mondrian"? | It turns probabilities into a set of plausible families with a finite-sample coverage guarantee. Mondrian calibrates one threshold per family, so the guarantee holds for every family separately. A marginal guarantee would be carried by benign traffic and say nothing about a 0.2% family. |
| WebAttack coverage is 87%, below 90%. Is the guarantee broken? | No. The guarantee is on average over calibration sets; with 69 calibration flows the realised coverage varies between 82% and 96% (Beta distribution), and 87% is inside that range. The report shows this range for every family. |
| Isn't the IP address a useful feature? | It is a shortcut: it identifies the lab's attacker host, not attack behaviour, and does not transfer to a new network. IPs are kept only as context for analysts. |
| Results on synthetic data? | Synthetic data was only used before the official release to build and test the pipeline; the submitted numbers are on the official data (the report states its source). |
| Why random split if attacks are sessions? | Random stratified splits are standard but optimistic; the temporal experiment is the realistic estimate, and we show both. |
