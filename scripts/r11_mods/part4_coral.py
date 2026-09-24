# Part 4: MOD 1.3 - Phase H CORAL + class-conditional MMD domain alignment (runs in parallel with
# Phase E self-training; both are additive, neither replaces the other).

MD_PHASEH = """## PHASE H (revision 11) — feature-alignment domain adaptation: CORAL and class-conditional MMD

Master-prompt modification 1.3. Phase E's lever is *self-training* (pseudo-labels change the training
set); this phase's lever is *feature alignment* (the feature space itself is transformed so the source
distribution matches the target). The two run **in parallel** - neither replaces the other, and the
Phase E / Phase G models stay untouched.

* **M5 CORAL** (Sun & Saenko, 2016) - whitens the source feature covariance and re-colours it to the
  target covariance (`X_al = (X - mu_s) C_s^{-1/2} C_t^{1/2} + mu_t`, computed in float64 with
  shrinkage for stability). The XGBoost is then **refitted on the aligned source TRAIN**. Uses target
  **TRAIN features only - no target labels, no target TEST rows**.
* **M5C class-conditional MMD (first-moment variant)** - the class-conditional mean term of the
  class-conditional MMD objective: target TRAIN rows receive *pseudo-labels from the frozen
  source-only model* (never ground truth), and each source row is shifted by
  `mu_t(pseudo-c) - mu_s(true-c)` for its class before CORAL is applied. Honest labelling: this
  implementation matches class-conditional **means** (the dominant kernel-MMD term), not the full
  kernel quantity.

Both variants are fitted on **F68-R (primary) and F54-R** as the master prompt requires, register into
`P2_SCORES` / `PHASE2_TABLE` with `claim = "unsupervised domain adaptation"`, and re-evaluate the
pinned Gate 2 with the exact same gate code Phase E uses (`assert_gate_spec` - the criterion string
and its SHA-256 are unchanged). The external threshold is selected on the **aligned source
validation** set, so no external number is consulted anywhere."""

CODE_PHASEH = r'''# ===================================================================================================
# PHASE H (REVISION 11, MOD 1.3) - CORAL + class-conditional-MMD(mean) feature alignment.
# Additive: Phase E/G models untouched; gate re-evaluation uses the SAME pinned criterion.
# ===================================================================================================
P11H_T0 = time.time()
REV11H = dict(CFG.revision11["coral"])
H_ROWS, H_INFO = [], {}
CORAL_RESULTS: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def _sym_sqrt_inv(C, eig_floor=1e-10):
    """Symmetric square root and inverse square root. Eigenvalues are clipped at a floor that is
    RELATIVE to the largest one (pseudo-inverse semantics): well-conditioned directions are exact,
    near-null directions stay bounded instead of exploding."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, max(w.max(), 1e-30) * eig_floor, None)
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


def _coral_params(Xs, Xt, eig_floor):
    """CORAL alignment operator: A = C_s^{-1/2} C_t^{1/2} with the mean pair (mu_s, mu_t).

    No additive shrinkage: the eigenvalue floor is RELATIVE to each covariance's own largest
    eigenvalue (pseudo-inverse semantics), so heterogeneous feature scales cannot distort the
    well-conditioned directions. cov(X_aligned) == C_t exactly for every direction above the floor."""
    d = Xs.shape[1]
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs.astype(np.float64), rowvar=False)
    Ct = np.cov(Xt.astype(np.float64), rowvar=False)
    _, Cs_inv_half = _sym_sqrt_inv(Cs, eig_floor)
    Ct_half, _ = _sym_sqrt_inv(Ct, eig_floor)
    return Cs_inv_half @ Ct_half, mu_s, mu_t


def _phase_h_variant(src, tgt, fset, variant):
    """Fit one alignment variant; returns eval row + scores. Cached by r7_cache."""
    def _compute():
        rk = f"{src}|{fset}"
        run = RUNS[rk]
        hp = run["best_params"]["xgb"]
        tr_s, va_s = partition_index(src, "train"), partition_index(src, "val")
        tr_t = partition_index(tgt, "train")
        ext = np.flatnonzero(EXTERNAL_MASKS[(src, tgt)]["strict_domain_unseen"])
        assert not np.intersect1d(tr_t, ext).size, "alignment pool overlaps the external rows"
        Xs, Xv = get_X(rk, src, tr_s), get_X(rk, src, va_s)
        Xt, Xe = get_X(rk, tgt, tr_t), get_X(rk, tgt, ext)
        y_s, y_v, y_e = (CLEAN[src]["y"].values[tr_s], CLEAN[src]["y"].values[va_s],
                         CLEAN[tgt]["y"].values[ext])
        pl_info = "none"
        Xs_for_align, Xv_for_align = Xs, Xv
        if variant == "cc-mmd":
            # pseudo-labels from the frozen source-only Stage-A model of the PRIMARY run
            # (this fset's own model is only fitted later, in Section 39R)
            m_pl = RUNS[f"{src}|{PRIMARY_FSET}"]["models"]["xgb"]
            p_pl = model_proba("xgb", m_pl, Xt)
            pl = (p_pl >= 0.5).astype(int)
            mu_t_c = np.stack([Xt[pl == c].mean(0) for c in (0, 1)])
            mu_s_c = np.stack([Xs[y_s == c].mean(0) for c in (0, 1)])
            shift = (mu_t_c - mu_s_c)[y_s]                       # per source row, TRUE class
            Xs_for_align, Xv_for_align = Xs + shift, Xv + (mu_t_c - mu_s_c)[y_v]
            pl_info = (f"pseudo-labels on target TRAIN from the frozen source-only XGB "
                       f"(estimated prevalence {pl.mean():.4f}, n={len(pl):,}); no ground truth used")
        A, mu_s_g, mu_t_g = _coral_params(Xs_for_align, Xt, REV11H["eig_floor"])
        Xs_al = (Xs_for_align - mu_s_g) @ A + mu_t_g
        Xv_al = (Xv_for_align - mu_s_g) @ A + mu_t_g
        m = make_model("xgb", hp["params"], rk, n_estimators=hp["n_estimators"])
        t0 = time.time()
        m.fit(Xs_al.astype(np.float32), y_s)
        LEDGER.record("rev11_coral_fit", f"{src}|{fset}|{variant}", src, "train", "fit_model", len(y_s),
                      f"Phase H {variant} alignment refit")
        pv = model_proba("xgb", m, Xv_al.astype(np.float32))
        thr = float(select_threshold(y_v, pv, CFG.threshold_metric))
        pe = model_proba("xgb", m, Xe)
        out = {"fit_seconds": round(time.time() - t0, 1), "val_roc_auc_aligned": float(fast_auc(y_v, pv)),
               "thr": thr, "y_ext": y_e, "p_ext": pe, "pseudo_label_info": pl_info,
               "n_train": int(len(y_s)), "n_align_target": int(len(Xt)), "n_eval": int(len(y_e))}
        del Xs, Xv, Xt, Xe, Xs_al, Xv_al, m
        gc.collect()
        return out
    return r7_cache(f"r11_phaseH_{src}_{fset}_{variant}", _compute)


for src in ["gram", "phresh"]:
    tgt = "phresh" if src == "gram" else "gram"
    direction = f"{CFG.datasets[src]['display']} -> {CFG.datasets[tgt]['display']}"
    for fset in REV11H["fsets"]:
        for variant in ["coral", "cc-mmd"]:
            nm = (f"M5 CORAL {fset.replace('R', '-R')}" if variant == "coral"
                  else f"M5C cc-MMD(mean)+CORAL {fset.replace('R', '-R')}")
            res = _phase_h_variant(src, tgt, fset, variant)
            me = classification_metrics(res["y_ext"], res["p_ext"], res["thr"])
            H_ROWS.append({"direction": direction, "model": nm, "population": "external STRICT",
                           "claim": "unsupervised domain adaptation",
                           "n_features": len(FEATURE_SETS[fset]), "n_train": res["n_train"],
                           "n_eval": res["n_eval"],
                           "target_information_used": ("target TRAIN features only, unlabelled (covariance"
                                                       + (" + own pseudo-labels for class means" if variant == "cc-mmd"
                                                          else "") + "); no target label, no target TEST"),
                           "tau": res["thr"], "estimated_target_prevalence": np.nan,
                           "true_target_prevalence": float(res["y_ext"].mean()), **me})
            CORAL_RESULTS[(src, fset, variant)] = {"name": nm, "y_ext": res["y_ext"], "p_ext": res["p_ext"],
                                                   "thr": res["thr"], "auc": float(me["roc_auc"]),
                                                   "f1": float(me["f1"]), "direction": direction,
                                                   "val_roc_auc_aligned": res["val_roc_auc_aligned"]}
            H_INFO[nm] = {"fit_seconds": res["fit_seconds"], "val_roc_auc_aligned": res["val_roc_auc_aligned"],
                          "pseudo_label_info": res["pseudo_label_info"], "threshold_tau": res["thr"]}
            P2_SCORES[(src, nm)] = (res["y_ext"], res["p_ext"])
            print(f"[Phase H] {direction} | {nm}: strict-external AUC {me['roc_auc']:.4f} "
                  f"(aligned source-val AUC {res['val_roc_auc_aligned']:.4f}, {res['fit_seconds']}s)")

PHASE_H_TABLE = pd.DataFrame(H_ROWS)
display(PHASE_H_TABLE.round(4))
save_table(PHASE_H_TABLE, "table0H_revision11_phaseH_alignment")
save_json(H_INFO, DIRS["metadata"] / "revision11_phaseH_alignment.json")

# ---- Gate 2 re-evaluated by the SAME pinned gate code (identical to the Phase E pattern) ------------
PHASE2_TABLE = pd.concat([PHASE2_TABLE, pd.DataFrame(H_ROWS)], ignore_index=True)
save_table(PHASE2_TABLE, "table0E_phase2_transfer_models")
_EXT11 = PHASE2_TABLE[PHASE2_TABLE["population"] == "external STRICT"]
_ZS11 = _EXT11[_EXT11["claim"].isin(["zero-shot", "unsupervised domain adaptation"])]
assert not _ZS11["model"].str.contains("ablation").any(), "an ablation model entered the Gate-2 pool"
P2_GATE["criterion"] = assert_gate_spec("phase2_zero_shot")
P2_GATE["by_direction"] = {}
for _d in _ZS11["direction"].unique():
    _dd = _ZS11[_ZS11["direction"] == _d]
    _b = _dd.loc[_dd["roc_auc"].idxmax()]
    P2_GATE["by_direction"][_d] = {
        "best_zero_shot_model": _b["model"], "best_zero_shot_roc_auc": float(_b["roc_auc"]),
        "best_zero_shot_accuracy": float(_b["accuracy"]),
        "M0_baseline_roc_auc": float(_dd.loc[_dd["model"].str.startswith("M0 source-only"), "roc_auc"].iloc[0]),
        "all_zero_shot_models": _dd.set_index("model")["roc_auc"].round(4).to_dict(),
        "best_model": _b["model"], "best_external_roc_auc": float(_b["roc_auc"]),
        "best_external_accuracy": float(_b["accuracy"]), "n_eval": int(_b["n_eval"]),
        "passed": bool(_b["roc_auc"] >= CFG.phase_gates["r6_p2_min_external_auc"])}
P2_GATE["passed_any_direction"] = bool(any(v["passed"] for v in P2_GATE["by_direction"].values()))
P2_GATE["passed_both_directions"] = bool(all(v["passed"] for v in P2_GATE["by_direction"].values()))
P2_GATE["passed"] = GATE_SPEC["phase2_zero_shot"]["callable"](P2_GATE)
P2_GATE["revision11_note"] = ("Revision 11 Phase H adds four gate-eligible CORAL / class-conditional-MMD "
                              "models per direction (unsupervised: target TRAIN features only). Criterion "
                              f"unchanged (SHA-256 {GATE_SPEC['phase2_zero_shot']['sha256'][:16]}).")
save_json(P2_GATE, DIRS["metadata"] / "phase2_external_auc_gate.json")
print("=" * 100)
for _d, _v in P2_GATE["by_direction"].items():
    print(f"PHASE 2 GATE [{_d}]: {'PASSED' if _v['passed'] else 'FAILED'} -- best zero-shot/UDA strict-external "
          f"ROC-AUC = {_v['best_zero_shot_roc_auc']:.4f} ({_v['best_zero_shot_model']}) vs threshold "
          f"{CFG.phase_gates['r6_p2_min_external_auc']}")
print(f"PHASE 2 GATE OVERALL (pinned criterion, BOTH directions): "
      f"{'PASSED' if P2_GATE['passed'] else 'FAILED'}")
LOG.info("Revision-11 Phase H complete in %.0fs", time.time() - P11H_T0)'''
