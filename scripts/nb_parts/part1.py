# -*- coding: utf-8 -*-
"""TRAC-Phish Revision 8 notebook — Part 1: banner, environment, config, checkpoints."""

CELLS_1 = []

CELLS_1.append(("md", r'''# TRAC-Phish — Revision 8 — FULL-SCALE EXECUTION

**Phishing URL detection on GramBeddings + PhreshPhish (URL-only, 2026) — complete corpora, no subsampling.**

Reconstructed-and-executed notebook. Pipeline: dataset audit → cleaning → leakage/overlap analysis →
partitioning → feature extraction (char n-gram TF-IDF + lexical) → representation analysis → model
training → main evaluation → transfer & robustness → calibration → SHAP/XAI → explanation stability
(ERS) → decision trust (DTS) → statistical testing → phase gates → negative-result handling → case
studies → sanity checks → reproducibility reporting → table/figure export → manifest & packaging.

Run controls: `TRAC_RUN_MODE=full` enforced (hard failure otherwise); `TRAC_MAX_ROWS` must be unset.
Orchestration: 1 coordinator + 40 specialist agents (A01–A06 pre-execution audits, A07–A24 in-pipeline
stage agents logged under `logs/stages/`, A25–A40 post-execution verifiers).'''))

CELLS_1.append(("code", r'''# ============ CELL: ENVIRONMENT & RUN-MODE ENFORCEMENT ============
import os, sys, time, json, math, random, re, hashlib, platform, subprocess, traceback
import warnings; warnings.filterwarnings("ignore")

t_run_start = time.time()

# --- Run-mode enforcement (directive: full scale, no row cap) ---
_RUN_MODE = os.environ.get("TRAC_RUN_MODE", "full").strip().lower()
_MAX_ROWS  = os.environ.get("TRAC_MAX_ROWS", "").strip()
if _RUN_MODE != "full":
    raise RuntimeError(f"TRAC_RUN_MODE must be 'full' (got '{_RUN_MODE}'). Refusing to run reduced mode.")
if _MAX_ROWS not in ("", "none", "NONE", "None", "0"):
    raise RuntimeError(f"TRAC_MAX_ROWS is set to '{_MAX_ROWS}'. Full-scale run forbids row caps.")

# Thread pinning (fixed thread count => reproducible numerics on this machine)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np, pandas as pd, scipy.sparse as sp, scipy.stats as st
from scipy.stats import spearmanr, ks_2samp, mannwhitneyu
import sklearn, scipy, matplotlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260923
random.seed(SEED); np.random.seed(SEED)

def _sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    fd = os.open(str(p), os.O_RDONLY)
    try:
        while True:
            b = os.read(fd, chunk)
            if not b: break
            h.update(b)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)   # drop page cache (small-RAM host)
    finally:
        os.close(fd)
    return h.hexdigest()

def _hw():
    with open("/proc/meminfo") as f:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
    return {"cpu_cores": os.cpu_count(), "ram_gb": round(mem["MemTotal"] / 1048576, 2),
            "python": platform.python_version(), "platform": platform.platform()}

ENV_INFO = {
    "time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "run_mode": _RUN_MODE, "trac_max_rows": None,
    "hardware": _hw(),
    "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                 "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    "seed": SEED,
}
print("=" * 78)
print("TRAC-PHISH REVISION 8 — RUN MODE: FULL — COMPLETE DATASETS — NO ROW CAP (TRAC_MAX_ROWS unset)")
print("=" * 78)
print(json.dumps(ENV_INFO, indent=2))'''))

CELLS_1.append(("code", r'''# ============ CELL: CONFIGURATION, PATHS, A-PRIORI GATE THRESHOLDS, STAGE AGENTS ============
from pathlib import Path

BASE     = Path("/home/z/my-project")
RAW_GB   = BASE / "data_raw/grambeddings/grambeddings_dataset_main"
RAW_PP   = BASE / "data_raw/phreshphish/phreshphish_transfer"
OUT      = BASE / "trac_phish_revision8_FULL_RUN"
SUBDIRS  = ["tables", "figures", "models", "ers", "perturbation", "reports", "manifests",
            "logs/agents", "logs/stages", "checkpoints", "diagnostics", "case_studies",
            "calibration", "shap", "statistical", "predictions"]
for d in SUBDIRS: (OUT / d).mkdir(parents=True, exist_ok=True)

DATASET_SHA = {
    "grambeddings_archive": "4a5572edeb3d37d7563a2c98562408c837c144465ec7e0d45b838134a0e88f46",
    "phreshphish_archive":  "c58b70f02eaa52fe1a7666d4e57f19a759480f53970e88d3e05a71dfd25fbbbb",
}

CFG = {
    "seed": SEED,
    "partition": {"val_fraction": 0.10, "stratified": True},
    "tfidf": {"analyzer": "char", "ngram_range": [1, 5], "min_df_vocab": 8,
              "max_features": 300000, "sublinear_tf": True, "dtype": "float32",
              "vocab_subsample": 60000, "transform_chunk": 50000},
    "logreg": {"C_grid": [0.25, 1.0, 4.0], "solver": "saga", "max_iter": 60, "tol": 1e-4},
    "sgdsvm": {"alpha_grid": [1e-5, 1e-4, 1e-3], "max_iter": 20},
    "mnb": {"alpha_grid": [0.05, 0.25, 1.0]},
    "histgb": {"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 31},
    "mlp": {"hidden": (64, 32), "max_iter": 120, "early_stopping": True},
    "robustness": {"sample": 25000, "perturbations": 8},
    "ers": {"sample": 500, "topk": 10, "weights": [0.4, 0.3, 0.3],
            "perturb_types": ["case_random", "typo_swap", "pad_benign", "subdomain_junk", "query_junk"]},
    "dts": {"weights": [0.7, 0.3]},
    "bootstrap": {"n_boot": 1000},
    "shap": {"background": 1000, "explain": 2000},
    "sanity": {"shuffle_n": 50000},
}

# A-PRIORI phase-gate thresholds — committed BEFORE any test-set evaluation.
GATE_THRESHOLDS = {
    "G1_data_integrity":       {"rows_exact": True},
    "G2_no_url_leakage":       {"max_exact_url_overlap_rate": 0.001},
    "G3_performance":          {"min_test_f1": 0.90, "min_test_roc_auc": 0.95},
    "G4_transfer_floor":       {"min_pp_f1": 0.60},
    "G5_calibration":          {"max_post_cal_ece": 0.05},
    "G6_ers":                  {"min_ers": 0.60, "warn_ers": 0.45},
    "G7_dts":                  {"min_dts": 0.70},
    "G8_sanity":               {"max_null_auc": 0.55, "determinism_hash_match": True,
                                "partition_overlap_zero": True},
}

STAGE_REGISTRY = {}   # agent_id -> {"name","status","t0","t1","seconds","error"}
def _reg_path(): return OUT / "logs/stage_registry.json"
def _reg_save():
    _reg_path().write_text(json.dumps(STAGE_REGISTRY, indent=2, default=str))

class stage_agent:
    """In-pipeline specialist agent A07..A24: own log file, timing, honest failure propagation."""
    def __init__(self, aid, name):
        self.aid, self.name = aid, name
    def __enter__(self):
        self.t0 = time.time()
        STAGE_REGISTRY[self.aid] = {"name": self.name, "status": "RUNNING", "t0": self.t0}
        _reg_save(); print(f"\n>>> [{self.aid}] {self.name} — START", flush=True)
        return self
    def __exit__(self, et, ev, tb):
        self.t1 = time.time()
        rec = STAGE_REGISTRY[self.aid]
        rec.update({"t1": self.t1, "seconds": round(self.t1 - self.t0, 2)})
        if et is None:
            rec["status"] = "COMPLETED"; print(f"<<< [{self.aid}] {self.name} — COMPLETED in {rec['seconds']}s", flush=True)
            (OUT / f"logs/stages/{self.aid}.md").write_text(
                f"# {self.aid} — {self.name}\n\n- status: COMPLETED\n- seconds: {rec['seconds']}\n- t0: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(self.t0))} UTC\n")
        else:
            rec.update({"status": "FAILED", "error": f"{et.__name__}: {ev}",
                        "traceback": traceback.format_exc()[-4000:]})
            (OUT / f"logs/stages/{self.aid}.md").write_text(
                f"# {self.aid} — {self.name}\n\n- status: FAILED\n- error: {et.__name__}: {ev}\n\n```\n{rec['traceback']}\n```\n")
            print(f"<<< [{self.aid}] {self.name} — FAILED: {ev}", flush=True)
        _reg_save()
        return False   # propagate errors honestly
'''))

CELLS_1.append(("code", r'''# ============ CELL: CHECKPOINT SYSTEM (mode/schema/dataset compatible) ============
SCHEMA_VERSION = "trac-phish-rev8-v2-canonical-dedup"

def _fingerprint(extra=None):
    fp = {"schema": SCHEMA_VERSION, "run_mode": _RUN_MODE,
          "dataset_sha": DATASET_SHA, "seed": SEED}
    if extra: fp["extra"] = extra
    return fp

def ckpt_save(name, obj, extra=None):
    p = OUT / f"checkpoints/{name}.joblib"
    import joblib
    joblib.dump({"__fp__": _fingerprint(extra), "data": obj}, p)
    print(f"    [ckpt] saved {name} ({p.stat().st_size/1e6:.1f} MB)", flush=True)

def ckpt_load(name, extra=None):
    """Load checkpoint ONLY if fingerprint matches current run mode/schema/datasets/params.
    Never mixes artifacts across incompatible run modes, schemas, or dataset versions.
    Page cache of the read file is dropped (POSIX_FADV_DONTNEED) to protect the small-RAM host."""
    import joblib
    p = OUT / f"checkpoints/{name}.joblib"
    if not p.exists(): return None
    try:
        blob = joblib.load(p)
        try:
            _fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(_fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(_fd)
        except Exception:
            pass
        if blob.get("__fp__") != _fingerprint(extra):
            print(f"    [ckpt] {name}: fingerprint mismatch -> recompute (no mode/schema mixing)", flush=True)
            return None
        print(f"    [ckpt] loaded {name} (resumable hit)", flush=True)
        return blob["data"]
    except Exception as e:
        print(f"    [ckpt] {name}: corrupt ({e}) -> recompute", flush=True)
        return None

print("Configuration ready. Output root:", OUT)
print("A-priori gate thresholds (committed before test evaluation):")
print(json.dumps(GATE_THRESHOLDS, indent=2))'''))
