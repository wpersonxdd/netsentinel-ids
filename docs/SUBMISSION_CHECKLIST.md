# DataNova 2026 · DN-CY-01 submission checklist

**Problem:** Network Intrusion Detection with Imbalanced Threat Data. *Develop a network intrusion detection model
that handles highly imbalanced threat data and adapts to evolving attack patterns.*

## Required deliverables

| Requirement | Where | Status |
|---|---|---|
| **Code** | `src/nids/` (pipeline), `app/` (console), `scripts/`, `tests/` | ✅ |
| **Documentation** | `README.md`, `docs/METHODOLOGY.md`, `docs/DATASET.md` | ✅ |
| **Evaluation** | `reports/REPORT.md` + `reports/figures/` + `reports/results/*.json` (regenerate on the official data) | ⏳ re-run on official data |
| **Presentation** | Outline and demo script in `docs/PRESENTATION.md`; build the slides from it | ⏳ build slides |

## Expected outcome (from the problem statement)

| Expected | How NetSentinel delivers it | Evidence |
|---|---|---|
| Detection model with **robust performance on minority classes** | 4 imbalance strategies × 5 models benchmarked; focal-loss MLP; rare-class F1 as an explicit metric; baseline-vs-final comparison on the same test set | REPORT §2–3, `per_class_f1.png` |
| **Alert triage interface** | Streamlit console: P1–P4 queue, explanations, playbooks, analyst decisions, CSV export | `app/streamlit_app.py`, demo video |
| Handles **highly imbalanced** data | SMOTE / undersampling inside training folds only; PR-AUC and macro metrics instead of accuracy | METHODOLOGY §2, §4 |
| **Adapts to evolving attack patterns** | Novelty detector for unseen families; leave-one-attack-out test; temporal drift test; label-free drift monitor; `nids retrain` on analyst decisions from the console | REPORT §6–8 |
| Suggested tech: Python, deep learning, SMOTE, ensemble methods | PyTorch focal-loss MLP, imbalanced-learn SMOTE, RF/GBM + soft-voting ensemble | `deep.py`, `resampling.py`, `models.py` |

## Judging criteria → where to point the judges

| Criterion | Strongest evidence |
|---|---|
| Problem understanding | Four sub-problems framed in METHODOLOGY intro (imbalance, open world, drift, humans in the loop); why accuracy is misleading |
| Innovation & originality | Hybrid supervised + novelty decision; label-free drift signals on *alerted* traffic; analyst feedback → retraining loop |
| Technical depth | Leakage guards, imblearn pipelines, focal loss, LOF vs Isolation Forest vs autoencoder comparison, PSI monitoring |
| Data usage & methodology | Validation-only model selection, single test-set evaluation, LOAO and temporal protocols |
| AI/ML approach | 20-configuration benchmark, ensemble, deep model, novelty detection |
| Solution quality | Tests (unit + end-to-end), one config, reproducible results JSON, CI workflow |
| Feasibility & scalability | Throughput figure in REPORT §3; `nids score` CLI for batch scoring; Dockerfile; Parquet + directory input |
| Real-world impact | Flow alerts collapsed into incidents (REPORT §5), P1–P4 queue, explicit false-alarm budget, response playbooks, one decision labels a whole incident |
| Presentation & communication | `docs/PRESENTATION.md` storyline + live console demo |

## Before submitting (deadline: 25 Nov 2026)

- [ ] Official data in `data/raw/official/`, `python -m nids all --data data/raw/official` run, REPORT regenerated
- [ ] Synthetic-data note removed from REPORT automatically (it only appears when the source path contains "synthetic")
- [ ] `triage.severity` extended for any new attack families in the official data
- [ ] `pytest -q` green
- [x] Author name in `LICENSE` and the README author line
- [ ] Screenshot of the console (Incidents view with an incident open) saved as `docs/img/console.png` and added to
      the README under "Results"
- [x] Repository pushed to GitHub (https://github.com/wpersonxdd/netsentinel-ids)
- [ ] CI enabled: move `.github/ci.yml` to `.github/workflows/ci.yml` on github.com (see README.ru.md), badge green
- [ ] Repository made public (or judges invited) at submission time
- [ ] Demo video (3 min, see `docs/PRESENTATION.md`) recorded and linked in README
- [ ] Slides exported to PDF, linked in README and in the registration form ("Project presentation link")
