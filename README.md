# TRAC-Phish — Revision 8 (Full-Scale Executed Run)

Phishing URL detection study executed end-to-end at **full scale** (`TRAC_RUN_MODE=full`, no row caps) on:

| Corpus | Split | Rows |
|---|---|---|
| GramBeddings | train (raw → cleaned) | 640,000 → 639,335 |
| GramBeddings | test (raw → cleaned) | 160,000 → 159,943 |
| GramBeddings | train fold / val fold | 575,401 / 63,934 (stratified 90/10, seed 20260923) |
| PhreshPhish URL-only 2026 | zero-shot transfer test | 168,060 (provider temporal split) |

**Run ID:** `tracphish-rev8-full-20260923` · 1 coordinator + 40 specialist agents · 18/18 pipeline stages · 31/31 notebook cells, 0 errors.

## Headline results (best model: LogReg C=4.0, char 1–5-gram TF-IDF, 129,620 features)

| Metric | GB test (in-corpus) | PhreshPhish-2026 (zero-shot) |
|---|---|---|
| F1 | **0.9746** [0.9738–0.9754] | 0.5101 |
| ROC-AUC | **0.9964** | 0.7873 |
| MCC | **0.9497** [0.9482–0.9512] | 0.3249 |
| Accuracy | 0.9748 | 0.6602 |

Calibration (isotonic, val-fit): ECE **0.0021** · ERS **0.8034** · DTS **0.9733** · Robustness max flip rate 4.56%.

## Phase gates (a-priori thresholds, honest evaluation)

- **PASS:** G1 data integrity · G3 performance · G5 calibration · G6 ERS · G7 DTS · G8 sanity (null AUC 0.5064, bit-identical determinism)
- **FAIL (reported honestly, not suppressed):**
  - G2 no-URL-leakage: 317 train∩test duplicate URLs (0.198% > 0.100% threshold) — inherent to the released GramBeddings split
  - G4 transfer floor: zero-shot F1 0.5101 < 0.60 — genuine temporal+corpus-shift finding (KS drift: https flag 0.386, path_len 0.193, entropy 0.097)
- 9 negative results recorded in `trac_phish_revision8_FULL_RUN/tables/negative_results.csv`

## Repository structure

```
trac-phish-revision8.ipynb                  # source notebook (research protocol)
trac-phish-revision8_FULL_EXECUTED.ipynb    # executed notebook, 31/31 cells, outputs preserved
scripts/                                    # notebook builder + executor + infrastructure scripts
trac_phish_revision8_FULL_RUN/
  ├── tables/        # 23 CSV + 23 LaTeX (results, gates, leakage, sanity, negative results, ...)
  ├── figures/       # 9 PNG (eval curves, transfer, calibration, SHAP, ERS/DTS, robustness)
  ├── models/        # 11 trained models + TF-IDF vectorizer (joblib)
  ├── predictions/   # test-set prediction arrays (npz)
  ├── shap/          # XAI: global n-gram + lexical permutation importances
  ├── ers/           # Explanation Robustness Scores  |  dts/ DTS scores
  ├── perturbation/  # robustness suite (flip rates)
  ├── calibration/   # reliability results
  ├── statistical/   # bootstrap CIs, McNemar exact, paired AUC
  ├── case_studies/  # qualitative error analysis
  ├── reports/       # execution report, reproducibility (md/json), classification report
  ├── manifests/     # run manifest, artifact manifest, 40-agent roster
  └── logs/          # per-stage and per-agent logs
download/trac_phish_revision8_RESULTS_ONLY.zip  # all results bundled (0.8 MB)
worklog.md           # full execution worklog (16 runs, failure taxonomy, fixes)
```

## Reproducing

Python 3.12, scikit-learn 1.5.2, pandas 2.2.3, numpy 2.1.3, shap 0.52.0, scipy 1.14.1.
Datasets (not included in this repo; ~155 MB): `grambeddings_dataset_main.rar` and
`phreshphish_url_only_2026.zip` placed under `data_raw/` (SHA-256 in `manifests/run_manifest.json`).

```bash
export TRAC_RUN_MODE=full
unset TRAC_MAX_ROWS
jupyter nbconvert --to notebook --execute trac-phish-revision8.ipynb
```

All randomness is seeded (global seed 20260923); the executed run is bit-identical on re-run (verified in G8 sanity).

## Not included here (GitHub 100 MB file limit)

- `trac_phish_revision8_FULL_RUN_ALL_OUTPUTS.zip` (1.13 GB, SHA-256 `902fede208c3b8cc…`, 158 entries) — complete archive incl. models, predictions, and 1.8 GB resumable checkpoints; delivered via the session download panel
- `trac_phish_revision8_FULL_RUN/checkpoints/` — 10 joblib checkpoint matrices (byte-verified inside the archive)
- `data_raw/` — source datasets

## Provenance disclosure

The originally uploaded `trac-phish-revision8.ipynb` did not persist to the execution environment.
It was reconstructed from the execution directive's 12-section specification (stages, gates, ERS/DTS,
checkpointing, run-mode enforcement), executed at full scale, and cross-verified by 16 independent
post-execution agents. Two memory-forced solver substitutions (LinearSVC→SGD-hinge, liblinear→saga)
are documented in the negative-results ledger. Full details: `trac_phish_revision8_FULL_RUN/reports/execution_report.md`.

Contact: musakhan5572@gmail.com
