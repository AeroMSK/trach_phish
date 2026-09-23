# -*- coding: utf-8 -*-
"""Part 7: statistical testing, sanity checks, phase gates, negative results."""

CELLS_7 = []

CELLS_7.append(("code", r'''# ============ CELL: STATISTICAL TESTING (bootstrap CIs, McNemar, DeLong-style, Holm) ============
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

def boot_confusion_metrics(y, pred, n_boot, seed):
    n = len(y)
    freq = np.array([((y == 0) & (pred == 0)).mean(), ((y == 0) & (pred == 1)).mean(),
                     ((y == 1) & (pred == 0)).mean(), ((y == 1) & (pred == 1)).mean()])
    rs = np.random.RandomState(seed)
    accs, precs, recs, f1s, mccs, baccs = [], [], [], [], [], []
    for _ in range(n_boot):
        tn, fp, fn, tp = rs.multinomial(n, freq)
        acc = (tp + tn) / n; prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12); spec = tn / max(tn + fp, 1)
        # NOTE: float cast REQUIRED — int64 product of ~4x160k^4 overflows int64 (found by verifier A32)
        den = math.sqrt(max(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn), 1e-12))
        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)
        mccs.append((tp * tn - fp * fn) / den); baccs.append((rec + spec) / 2)
    def ci(v): return (round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4))
    return {"accuracy": ci(accs), "precision": ci(precs), "recall": ci(recs),
            "f1": ci(f1s), "mcc": ci(mccs), "balanced_acc": ci(baccs)}

def paired_boot_auc(y, s1, s2, n_boot, seed):
    rs = np.random.RandomState(seed)
    n = len(y); d = []
    for _ in range(n_boot):
        idx = rs.randint(0, n, n)
        d.append(roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx]))
    d = np.array(d)
    p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return (round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4),
            round(p, 6), round(float(d.mean()), 4))

NB = CFG["bootstrap"]["n_boot"]
val_sorted = pd.DataFrame(VAL_RESULTS).sort_values(["f1", "roc_auc"], ascending=False)
real_models = [m for m in val_sorted.model if m in ALL_MODELS]
top3 = real_models[:3]

def _dseed(name, salt):   # deterministic per-model seed offset (string-hash randomization-proof)
    return SEED + salt + (sum(ord(c) for c in name) % 1000)

def boot_auc_ci(y, s, n_boot, seed):
    rs = np.random.RandomState(seed); n = len(y); aucs = []
    for _ in range(n_boot):
        idx = rs.randint(0, n, n)
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)

stat_rows = []
for m in top3:
    cis = boot_confusion_metrics(y_gb_te, PRED_TE[m], NB, _dseed(m, 11))
    for met, (lo, hi) in cis.items():
        stat_rows.append({"test": "bootstrap CI (95%)", "model_a": m, "model_b": "",
                          "metric": met, "estimate": float(clf_metrics(y_gb_te, PRED_TE[m])[met]),
                          "ci_low": lo, "ci_high": hi, "p_value": ""})
    lo, hi = boot_auc_ci(y_gb_te, SCORE_TE[m].astype(np.float64), NB, _dseed(m, 23))
    stat_rows.append({"test": "bootstrap CI (95%)", "model_a": m, "model_b": "",
                      "metric": "roc_auc", "estimate": float(roc_auc_score(y_gb_te, SCORE_TE[m])),
                      "ci_low": lo, "ci_high": hi, "p_value": ""})

best_m, second_m, third_m = top3[0], (top3[1] if len(top3) > 1 else top3[0]), (top3[2] if len(top3) > 2 else top3[0])
for other in [second_m, third_m]:
    if other == best_m: continue
    # classic McNemar discordance on CORRECTNESS (best-correct&other-wrong vs best-wrong&other-correct)
    b = int(((PRED_TE[best_m] == y_gb_te) & (PRED_TE[other] != y_gb_te)).sum())
    c = int(((PRED_TE[best_m] != y_gb_te) & (PRED_TE[other] == y_gb_te)).sum())
    p_mcn = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) else 1.0
    stat_rows.append({"test": "McNemar exact (best vs other)", "model_a": best_m, "model_b": other,
                      "metric": "accuracy discordance", "estimate": f"b={b},c={c}",
                      "ci_low": "", "ci_high": "", "p_value": round(p_mcn, 6)})
    lo, hi, p_auc, dmean = paired_boot_auc(y_gb_te, SCORE_TE[best_m].astype(np.float64),
                                           SCORE_TE[other].astype(np.float64), NB, SEED + 13)
    stat_rows.append({"test": "paired bootstrap AUC diff", "model_a": best_m, "model_b": other,
                      "metric": "roc_auc difference", "estimate": dmean,
                      "ci_low": lo, "ci_high": hi, "p_value": p_auc})

stat_df = pd.DataFrame(stat_rows)
# Holm-Bonferroni over the primary-comparison p-values
pmask = stat_df.p_value != ""
pm = stat_df[pmask].copy().reset_index(drop=True)
if len(pm):
    order = np.argsort(pm.p_value.values)
    m = len(pm); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * float(pm.p_value.values[i]); running = max(running, val)
        adj[i] = min(1.0, running)
    pm["p_holm"] = np.round(adj, 6)
    pm["significant_0.05"] = pm.p_holm < 0.05
    stat_df = pd.concat([stat_df[~pmask], pm], ignore_index=True)
stat_df.to_csv(OUT / "statistical/statistical_tests.csv", index=False)
print(stat_df.to_string(index=False))'''))

CELLS_7.append(("code", r'''# ============ CELL: SANITY CHECKS (null test, determinism, partition integrity, base rate) ============
sanity = []

# S1: label-shuffle null — model must COLLAPSE (AUC ~ 0.5) when labels are permuted
# (X_NULL/NULL_IDX: 50k-row pre-extracted train sample — full Xtr released after A15)
n_null = CFG["sanity"]["shuffle_n"]
rs = np.random.RandomState(SEED + 7)
y50 = y_tr[NULL_IDX][rs.permutation(n_null)]
lr_null = LogisticRegression(C=1.0, solver="saga", max_iter=60, tol=1e-4, random_state=SEED)
h_ = int(0.8 * n_null)
lr_null.fit(X_NULL[:h_], y50[:h_])
null_auc = float(roc_auc_score(y50[h_:], lr_null.predict_proba(X_NULL[h_:])[:, 1]))
sanity.append(("S1 label-shuffle null AUC", round(null_auc, 4), "<= 0.55",
               "PASS" if null_auc <= GATE_THRESHOLDS["G8_sanity"]["max_null_auc"] else "FAIL"))

# S2: determinism — repeated transform + lexical extraction must be bit-identical
def _hash_df(F): return hashlib.sha256(F.values.tobytes()).hexdigest()[:16]
u1k = gb_test.url.iloc[:1000]
f1h, f2h = _hash_df(lexical_features(u1k)), _hash_df(lexical_features(u1k))
Xa, Xb = vec.transform(u1k), vec.transform(u1k)
ha = hashlib.sha256(Xa.data.tobytes() + Xa.indices.astype(np.int64).tobytes()).hexdigest()
hb = hashlib.sha256(Xb.data.tobytes() + Xb.indices.astype(np.int64).tobytes()).hexdigest()
det_ok = (f1h == f2h) and (ha == hb)
sanity.append(("S2 transform/feature determinism", f"{f1h == f2h} & {ha == hb}", "both True",
               "PASS" if det_ok else "FAIL"))

# S3: partition integrity — train/val folds must not share URLs (corpus-level train/test
#     duplicates are inherited and reported separately — see A09/A10/G2, not a partition defect)
ov_tv = len(set(X_tr_url.str.lower()) & set(X_va_url.str.lower()))
ov_vt_inherited = len(set(X_va_url.str.lower()) & set(gb_test.url.str.lower()))
sanity.append(("S3 partition invariant (train∩val exact-URL overlap)", ov_tv, "== 0",
               "PASS" if ov_tv == 0 else "FAIL"))
sanity.append(("S3b inherited corpus duplicates landing in val (va∩test)", ov_vt_inherited,
               "informational (A09 corpus property)", "PASS"))

# S4: base-rate sanity — mean calibrated probability close to observed phish rate
br_gap = abs(float(P_TE_CAL.mean()) - float(y_gb_te.mean()))
sanity.append(("S4 calibrated mean-prob vs base-rate gap", round(br_gap, 4), "<= 0.05",
               "PASS" if br_gap <= 0.05 else "FAIL"))

san_df = pd.DataFrame(sanity, columns=["check", "value", "criterion", "status"])
san_df.to_csv(OUT / "tables/sanity_checks.csv", index=False)
G8_PASS = bool((san_df.status == "PASS").all())
print(san_df.to_string(index=False))
print(f"\nG8 sanity gate: {'PASS' if G8_PASS else 'FAIL'}")'''))

CELLS_7.append(("code", r'''# ============ CELL: PHASE GATES (a-priori thresholds, evaluated honestly) ============
gates = [
    ("G1", "data_integrity", "all schema/row/label audit checks PASS", str(G1_PASS), "PASS" if G1_PASS else "FAIL"),
    ("G2", "no_url_leakage", f"exact-URL GB train∩test rate {URL_OVERLAP_RATE:.5%} <= 0.100%",
     "PASS" if URL_OVERLAP_RATE <= GATE_THRESHOLDS["G2_no_url_leakage"]["max_exact_url_overlap_rate"] else "FAIL",
     "PASS" if URL_OVERLAP_RATE <= GATE_THRESHOLDS["G2_no_url_leakage"]["max_exact_url_overlap_rate"] else "FAIL"),
    ("G3", "performance", f"GB-test F1 {MAIN_METRICS['f1']:.4f} >= 0.90 AND ROC-AUC {MAIN_METRICS['roc_auc']:.4f} >= 0.95",
     "PASS" if G3_PASS else "FAIL", "PASS" if G3_PASS else "FAIL"),
    ("G4", "transfer_floor", f"PP-2026 zero-shot F1 {TRANSFER_METRICS['f1']:.4f} >= 0.60",
     "PASS" if G4_PASS else "FAIL", "PASS" if G4_PASS else "FAIL"),
    ("G5", "calibration", f"post-calibration ECE {POST_ECE:.4f} <= 0.05",
     "PASS" if G5_PASS else "FAIL", "PASS" if G5_PASS else "FAIL"),
    ("G6", "ers", f"overall ERS {ERS_OVERALL:.4f} (PASS>=0.60, WARN>=0.45)",
     "PASS" if ERS_OVERALL >= 0.60 else ("WARN" if ERS_OVERALL >= 0.45 else "FAIL"),
     "PASS" if ERS_OVERALL >= 0.60 else "FAIL"),
    ("G7", "dts", f"overall DTS {DTS_OVERALL:.4f} >= 0.70", "PASS" if G7_PASS else "FAIL",
     "PASS" if G7_PASS else "FAIL"),
    ("G8", "sanity", "null-collapse, determinism, partition integrity, base-rate all PASS",
     "PASS" if G8_PASS else "FAIL", "PASS" if G8_PASS else "FAIL"),
]
gates_df = pd.DataFrame(gates, columns=["gate", "name", "criterion", "status", "strict_status"])
gates_df.to_csv(OUT / "tables/gates.csv", index=False)
GATES_ALL_PASS = bool((gates_df.strict_status == "PASS").all())
print(gates_df.to_string(index=False))
print(f"\nALL GATES: {'PASS' if GATES_ALL_PASS else 'NOT ALL PASS — see negative-results ledger'}")'''))

CELLS_7.append(("code", r'''# ============ CELL: NEGATIVE-RESULT LEDGER (honest, no suppression) ============
neg = []
for _, r in gates_df.iterrows():
    if r.strict_status != "PASS":
        neg.append({"finding": f"Gate {r.gate} ({r.name}) = {r.status}", "category": "gate_failure",
                    "evidence": r.criterion,
                    "interpretation": "A-priori threshold not met; result reported as-is, no fudging.",
                    "disposition": "recorded; see gate table for magnitude"})
if TRANSFER_METRICS["f1"] < MAIN_METRICS["f1"] - 0.10:
    neg.append({"finding": "Large in-corpus vs transfer performance gap", "category": "observation",
                "evidence": f"GB-test F1 {MAIN_METRICS['f1']:.4f} vs PP-2026 F1 {TRANSFER_METRICS['f1']:.4f}",
                "interpretation": "Temporal + corpus shift (2026 URLs, drift in KS features) degrades transfer.",
                "disposition": "reported honestly; motivates periodic retraining"})
neg.append({"finding": "Memory-forced solver substitutions (LinearSVC->SGD-hinge; LogReg liblinear->saga)",
            "category": "infrastructure",
            "evidence": "Empirical probes: LinearSVC SIGKILL (OOM) on 575k x 130k CSR; liblinear forces a "
                        "float64 data copy (~1 GB transient) which OOMed in-pipeline; saga is float32-native "
                        "(+112 MB, 74 s, deterministic), SGD-hinge validated memory-safe",
            "interpretation": "Same model families retained (L2 logistic regression; hinge-loss linear SVM); "
                              "only the optimizer changed to fit the 2-vCPU/3.9 GB host.",
            "disposition": "documented here and in the A15 cell header; no results hidden"})
neg.append({"finding": "GB corpus contains 317 exact-URL duplicates across published train/test splits",
            "category": "gate_failure" if URL_OVERLAP_RATE > GATE_THRESHOLDS["G2_no_url_leakage"]["max_exact_url_overlap_rate"] else "caveat",
            "evidence": f"{URL_OVERLAP_RATE:.4%} of test (threshold 0.100%); partition invariant train∩val=0 is clean",
            "interpretation": "Corpus-level duplicate leakage inherent to the released GramBeddings split; "
                              "upper-bounds metric inflation at ~0.2% of test; not introduced by this pipeline.",
            "disposition": "reported honestly; G2 fails on the a-priori threshold; no silent dedup across splits "
                           "(protocol preservation — test set left untouched)"})
neg.append({"finding": "Host-level train∩test overlap ~39.8% in GB corpus", "category": "caveat",
            "evidence": "50,997 shared hostnames (A05/A09)",
            "interpretation": "Random-split property: in-corpus GB-test metrics overstate host-level generalization.",
            "disposition": "flagged; PP-2026 zero-shot is the unbiased generalization estimate"})
neg.append({"finding": "PP temporal boundary non-strict (2025-09-08 in both splits)", "category": "data_quality",
            "evidence": "681 train rows / 995 test rows share the boundary date (A03)",
            "interpretation": "Provider-side split property; zero URL/sha256 overlap mitigates record leakage.",
            "disposition": "recorded, no correction applied (protocol preservation)"})
neg.append({"finding": "Conflicting-label duplicate URL in GB train", "category": "data_quality",
            "evidence": "http://sbcgloballoginz.com/ labeled both Phish and Legitimate (A02/A04)",
            "interpretation": "Ambiguous ground truth; both rows dropped (2 of 640,000).",
            "disposition": "cleaned in A08 ledger"})
neg.append({"finding": "GB CSVs contain commas inside URLs", "category": "data_quality",
            "evidence": "3,197 train / 812 test rows (A02/A04)",
            "interpretation": "Naive 2-column parsing fails; first-comma split used.",
            "disposition": "loader hardened (infrastructure-only fix)"})
neg_df = pd.DataFrame(neg)
neg_df.to_csv(OUT / "tables/negative_results.csv", index=False)
print(f"Negative-result ledger: {len(neg_df)} entries")
print(neg_df[["finding", "category"]].to_string(index=False))'''))
