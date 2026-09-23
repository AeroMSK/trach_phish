# Reproducibility Report — TRAC-Phish Revision 8 (FULL)

- run mode: FULL (TRAC_RUN_MODE=full, TRAC_MAX_ROWS unset)
- seed: 20260923
- total rows processed: 639,335 GB-train / 63,934 val / 159,943 GB-test / 168,060 PP-test
- tf-idf vocab: 129,620 char 1-5 grams
- selected model: LogReg(C=4.0)
- GB-test F1 / AUC: 0.9746 / 0.9964
- PP-2026 F1 / AUC: 0.5101 / 0.7873
- calibration: isotonic, ECE 0.0021
- ERS / DTS: 0.8034 / 0.9733
- gates all pass: False
- notebook runtime (s): 364.7
