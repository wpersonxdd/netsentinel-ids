# Data: synthetic demo, official dataset, public datasets

## The bundled synthetic dataset

`python -m nids generate-data` writes `data/raw/synthetic_flows.csv` (100,000 flows; `python -m nids all` creates it
automatically). It follows the CIC-IDS2017 column style and is built to be **hard in the same ways real data is**:

| Property | How it is modelled |
|---|---|
| Imbalance | Benign ~80%, DoS 7%, DDoS 5%, PortScan 4.5%, BruteForce 1.5%, WebAttack 0.7%, Botnet 0.6%, Infiltration 0.2%, Exfiltration 0.1% |
| Overlap with benign | Web attacks live inside ordinary HTTP sessions; infiltration looks like SMB file sharing; botnet C2 looks like small HTTPS exchanges; legitimate backups look like exfiltration |
| Label noise | 2% of each attack's rows are benign-looking flows inside the attack window, as in public captures |
| Concept drift | In the last 35% of the timeline: DoS shifts to slow variants, port scans become stealthy, botnet C2 moves to TLS/443, and **Exfiltration appears for the first time** |
| Data-quality quirks | `Infinity` rates for zero-duration flows, missing values, 0.5% exact duplicates |

It exists so every command runs end-to-end before the official data is released. **Numbers measured on it show
that the method works; they say nothing about real-world performance.**

## Plugging in the official hackathon dataset

1. Put the files in `data/raw/official/` (one file or many; CSV or Parquet; they are concatenated).
2. Look at it first:
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('data/raw/official/<file>.csv', nrows=5000); print(df.columns.tolist()); print(df.iloc[:, -1].value_counts())"
   ```
3. Run everything:
   ```bash
   python -m nids all --data data/raw/official
   ```
   or set `data.path` in `configs/default.yaml` (better: copy it to `configs/official.yaml` and pass `--config`).
4. Read `reports/REPORT.md`, open the console (`streamlit run app/streamlit_app.py`).

### Config knobs you may need

| Key | When to change it |
|---|---|
| `data.label_column` | Auto-detection picks the first multi-class column among `attack_cat, label, class, attack, attack_type, category, target`. Set it if the label is called something else |
| `data.time_column` | Auto-detected from `timestamp, stime, start_time, time, ts`. Without one, the drift experiment is skipped (everything else still runs) |
| `data.label_mode` | `family` (default) groups variants; `raw` keeps the original labels if the organisers want per-attack results |
| `data.extra_drop_columns` | Any column that leaks the answer (e.g. an `attack_id`, a per-capture identifier, a second label) |
| `data.max_rows` | Very large datasets (millions of flows): stratified subsample that never thins a class below 1,000 rows |
| `benchmark.max_train_rows` | Keep the full test set but cap benchmark training rows for speed |
| `data.min_class_count` | Classes below this are dropped (default 10) |
| `triage.severity` | Add severities for attack families the official data introduces (unknown families default to `medium`) |

### Leakage checklist for a new dataset

- [ ] No IP addresses, flow IDs, capture IDs or timestamps among the features (`Features:` and `dropped as identifiers`
      lines in the log show what was used/removed)
- [ ] No second label column (e.g. both `attack_cat` and binary `label`); the loader removes known names automatically
- [ ] Duplicates reported in the log are plausible
- [ ] If results look too good (macro-F1 ≈ 1.0), check permutation importance: a single dominant feature often means leakage
- [ ] Destination port is kept on purpose (it identifies the service under attack). If the capture used fixed lab
      ports so that the port alone gives away the attack, add it to `data.extra_drop_columns` and compare the two runs

## CIC-IDS2017: the real-traffic run

NetSentinel ships a ready config for CIC-IDS2017 (Canadian Institute for Cybersecurity, five days of labelled
traffic, 3–7 July 2017):

1. On https://www.unb.ca/cic/datasets/ids-2017.html download **`GeneratedLabelledFlows.zip`** (≈ 280 MB). Prefer it
   over `MachineLearningCSV.zip`: it keeps source/destination IPs and timestamps, which the session-grouped and temporal
   protocols, the drift experiment and incident grouping need. A download of a few hundred *bytes* is the server's
   "504 Gateway Timeout" page; retry later.
2. Extract the eight CSVs into `data/raw/cicids2017/`.
3. Run everything (≈ 1–1.5 h on a laptop; results in `reports/cicids2017/`, model in `artifacts/cicids2017/`):
   ```bash
   python -m nids all --config configs/cicids2017.yaml
   ```
4. In the console, pick **CIC-IDS2017 (real traffic)** in the sidebar.

Known dataset problems and how the pipeline deals with them:

| Problem in CIC-IDS2017 | Handling |
|---|---|
| Afternoon timestamps written 1:00–5:00 **without PM** (12-hour clock), so afternoon flows sort before morning ones | `time_add_12h_before_hour: 8` (capture ran 08:00–18:00) |
| Dates are day/month/year, and files mix `03/07/2017 08:55:58` with `7/7/2017 3:30` | `time_dayfirst: true` with explicit formats per file style |
| 288,602 completely empty rows in the Thursday WebAttacks file | rows without a label are dropped |
| Label text broken by encoding (`Web Attack � Brute Force`) | label normalisation maps it to `WebAttack` |
| Duplicate column (`Fwd Header Length` twice) | de-duplicated names (`fwd_header_length_1`) |
| `Infinity` / NaN in `Flow Bytes/s`, `Flow Packets/s` | → NaN → median imputation inside the pipeline |
| 308,218 exact duplicate flows | removed before splitting |
| 8 constant columns (bulk-rate features are always 0) | dropped |
| 2.5 M flows after cleaning, 80% benign, 11 Heartbleed / 36 Infiltration flows | stratified 600k sample, families < 5,000 flows kept whole (`max_rows`, `min_per_class`); the report states the effect on precision |
| Each attack type happens on one day only | the temporal protocol and the drift experiment therefore test *unseen* families, which is exactly what the novelty detector is for |

The prepared 600k-flow dataset is cached in `data/processed/` (`data.cache: true`), so the seven steps parse the 1.2 GB
of CSVs once.

## Public datasets to practise on

| Dataset | Notes |
|---|---|
| [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | ~2.8M flows, 15 labels, 8 daily CSVs; the closest to the synthetic schema. Use `data.max_rows` |
| [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) | Larger AWS-based successor, same feature style |
| [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | 9 attack categories in `attack_cat`; blank category for normal traffic is handled |
| [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) | Classic, small, headerless `.txt`; mapped to DoS / Probe / R2L / U2R |

All four load without code changes: `python -m nids all --data <folder or file>`.
