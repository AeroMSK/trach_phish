# -*- coding: utf-8 -*-
"""Part 6: A22 SHAP/XAI, A23 explanation stability & ERS, A24 DTS."""

CELLS_6 = []

CELLS_6.append(("code", r'''# ============ CELL: [A22] SHAP / XAI — GLOBAL + LOCAL EXPLANATIONS ============
import shap

with stage_agent("A22", "SHAP explainability (analytic interventional linear-SHAP + tree permutation)"):
    Xva_tfidf = None; gc.collect()   # released after A21 (last consumer) — 170 MB back
    shap_report = {}
    if BEST_IS_TFIDF and hasattr(BEST, "coef_"):
        # --- analytic interventional Linear SHAP: phi_j = w_j * (x_j - mu_j) ---
        # background: pre-extracted 1000-row train sample (X_BG; full Xtr released after A15)
        MU = np.asarray(X_BG.mean(axis=0)).ravel().astype(np.float64)                # E[x_j]
        W  = np.asarray(BEST.coef_).ravel().astype(np.float64)
        GRAMS = np.array([inv[i] for i in range(len(Vocab))], dtype=object)
        BASE_PHI = -W * MU                              # phi for absent features
        base_order = np.argsort(-np.abs(BASE_PHI))      # precomputed |base| order

        rs = np.random.RandomState(SEED)
        ex_idx = rs.choice(Xte_tfidf.shape[0], size=CFG["shap"]["explain"], replace=False)
        X_exp = Xte_tfidf[ex_idx]; y_exp = y_gb_te[ex_idx]

        # exact global mean|phi_j|: absent part (n - df_j)*|mu_j*w_j| + present part sum|x-w*mu|
        dfj = np.asarray((X_exp > 0).sum(axis=0)).ravel()
        present_abs = np.zeros(len(W)); n_e = X_exp.shape[0]
        for i in range(X_exp.shape[0]):
            lo, hi = X_exp.indptr[i], X_exp.indptr[i + 1]
            js, xs = X_exp.indices[lo:hi], X_exp.data[lo:hi].astype(np.float64)
            present_abs[js] += np.abs(W[js] * xs - W[js] * MU[js])
        mean_abs_phi = ((n_e - dfj) * np.abs(BASE_PHI) + present_abs) / n_e
        top_g = np.argsort(-mean_abs_phi)[:40]
        glob = pd.DataFrame({"ngram": GRAMS[top_g], "mean_abs_shap": mean_abs_phi[top_g].round(6),
                             "coef": W[top_g].round(4)})
        glob.to_csv(OUT / "shap/global_ngram_importance.csv", index=False)
        shap_report["method_linear"] = "analytic interventional Linear SHAP (phi_j = w_j*(x_j - mu_j)), background=1000 train docs, explain=2000 test docs"

        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        top20 = glob.head(20).iloc[::-1]
        ax.barh([repr(g) for g in top20.ngram], top20.mean_abs_shap, color="#16a085")
        ax.set(xlabel="mean |SHAP|", title=f"Global char-n-gram importance — {best_name}")
        fig.savefig(OUT / "figures/shap_global_ngrams.png", dpi=160); plt.close(fig)

        def shap_topk_row(row, k=10):
            """Top-k signed SHAP for one sparse row (CSR slice)."""
            lo, hi = row.indptr[0], row.indptr[1]
            js, xs = row.indices[lo:hi], row.data[lo:hi].astype(np.float64)
            cand = set(js.tolist())
            for j in base_order[:4 * k]:
                cand.add(int(j))
                if len(cand) >= 2 * k + len(js): break
            cand = np.fromiter(cand, dtype=np.int64)
            xvals = np.zeros(len(cand), dtype=np.float64)
            pos = {j: i for i, j in enumerate(js)}
            for i, j in enumerate(cand):
                xvals[i] = xs[pos[j]] if j in pos else 0.0
            phi = W[cand] * (xvals - MU[cand])
            ord_ = np.argsort(-np.abs(phi))[:k]
            return [(str(GRAMS[cand[i]]), float(phi[i])) for i in ord_]
        SHAP_READY = True
        print(f"[A22] Linear SHAP ready. Top-10 global grams: {glob.head(10)[['ngram','mean_abs_shap']].values.tolist()}")
    else:
        SHAP_READY = False
        shap_report["method_linear"] = f"best model {best_name} is not linear-coefficient model — linear SHAP skipped"

    # --- lexical tree model explanation (permutation importance, val fold) ---
    gb_name = "HistGB(lexical)"
    if gb_name in LEXMODELS:
        from sklearn.inspection import permutation_importance
        rs2 = np.random.RandomState(SEED)
        vi = permutation_importance(LEXMODELS[gb_name], FX["va"].values[:8000], y_va[:8000],
                                    n_repeats=5, random_state=SEED, scoring="roc_auc")
        imp = pd.DataFrame({"feature": LEX_COLS, "perm_importance_mean": vi.importances_mean.round(5),
                            "std": vi.importances_std.round(5)}).sort_values(
                            "perm_importance_mean", ascending=False)
        imp.to_csv(OUT / "shap/lexical_permutation_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
        t = imp.head(20).iloc[::-1]
        ax.barh(t.feature, t.perm_importance_mean, xerr=t["std"], color="#8e44ad", alpha=.85)
        ax.set(xlabel="permutation importance (delta ROC-AUC)", title="Lexical feature importance — HistGB")
        fig.savefig(OUT / "figures/shap_lexical_importance.png", dpi=160); plt.close(fig)
        shap_report["method_lexical"] = "sklearn permutation_importance on HistGB (val fold sample, ROC-AUC)"
        print(f"[A22] Lexical permutation top-8: {imp.head(8)[['feature','perm_importance_mean']].values.tolist()}")

    pd.DataFrame([{"aspect": k, "detail": v} for k, v in shap_report.items()]).to_csv(
        OUT / "shap/xai_methods.csv", index=False)'''))

CELLS_6.append(("code", r'''# ============ CELL: [A23] EXPLANATION STABILITY -> ERS ============
def _stab_pair(a, b, k):
    na = [n for n, _ in a][:k]; nb = [n for n, _ in b][:k]
    sa, sb = set(na), set(nb)
    jac = len(sa & sb) / max(1, len(sa | sb))
    shared = sa & sb
    if shared:
        da = {n: i for i, n in enumerate(na)}; db_ = {n: i for i, n in enumerate(nb)}
        va = {n: dict(a)[n] for n in shared}; vb = {n: dict(b)[n] for n in shared}
        sign_agree = float(np.mean([np.sign(va[n]) == np.sign(vb[n]) for n in shared]))
        if len(shared) >= 4:
            ra = st.rankdata([da[n] for n in shared]); rb = st.rankdata([db_[n] for n in shared])
            sp = float(spearmanr(ra, rb).statistic)
            sp = 0.0 if math.isnan(sp) else sp
        else: sp = 0.0
    else:
        sign_agree, sp = 0.0, 0.0
    return jac, sign_agree, sp

with stage_agent("A23", "Explanation stability under perturbation -> ERS"):
    assert SHAP_READY, "Explanation stability requires the linear SHAP explainer."
    Xte_tfidf = None; gc.collect()   # released after A22 (last consumer) — 270 MB back
    k = CFG["ers"]["topk"]; wJ, wS, wR = CFG["ers"]["weights"]
    rs = np.random.RandomState(SEED)
    n_ers = CFG["ers"]["sample"]
    pi = rs.choice(np.where(y_gb_te == 1)[0], n_ers // 2, replace=False)
    ni = rs.choice(np.where(y_gb_te == 0)[0], n_ers - n_ers // 2, replace=False)
    eidx = np.concatenate([pi, ni]); rs.shuffle(eidx)
    ers_urls = gb_test.url.iloc[eidx].reset_index(drop=True)

    pert5 = make_perturbations(ers_urls, seed=SEED + 1)
    rows = []
    for pname in CFG["ers"]["perturb_types"]:
        puri = pert5[pname]
        Xp = chunked_transform(vec, puri, 25000)
        Xo = chunked_transform(vec, ers_urls, 25000)
        Js, Ss, Rs = [], [], []
        for i in range(len(ers_urls)):
            a = shap_topk_row(Xo[i], k); b = shap_topk_row(Xp[i], k)
            j_, s_, r_ = _stab_pair(a, b, k)
            Js.append(j_); Ss.append(s_); Rs.append(r_)
        ers_p = wJ * float(np.mean(Js)) + wS * float(np.mean(Ss)) + wR * float(np.mean(Rs))
        rows.append({"perturbation": pname, "topk_jaccard": round(float(np.mean(Js)), 4),
                     "sign_agreement": round(float(np.mean(Ss)), 4),
                     "spearman_rank": round(float(np.mean(Rs)), 4),
                     "ERS": round(ers_p, 4)})
        print(f"    {pname}: J={np.mean(Js):.3f} sign={np.mean(Ss):.3f} rho={np.mean(Rs):.3f} -> ERS={ers_p:.3f}", flush=True)
    ers_df = pd.DataFrame(rows)
    ERS_OVERALL = float(ers_df.ERS.mean())
    ers_df.loc[len(ers_df)] = {"perturbation": "OVERALL(mean)", **{c: round(float(ers_df[c].mean()), 4)
                                                    for c in ["topk_jaccard", "sign_agreement", "spearman_rank", "ERS"]}}
    ers_df.to_csv(OUT / "ers/ers_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    e = ers_df[ers_df.perturbation != "OVERALL(mean)"]
    ax.bar(e.perturbation, e.ERS, color="#d35400", alpha=.85)
    ax.axhline(GATE_THRESHOLDS["G6_ers"]["min_ers"], color="#c0392b", ls="--", lw=1, label="gate threshold 0.60")
    ax.axhline(GATE_THRESHOLDS["G6_ers"]["warn_ers"], color="#f39c12", ls=":", lw=1, label="warn 0.45")
    ax.set(ylim=(0, 1)); ax.set(ylabel="ERS", title=f"Explanation Reliability Score — {best_name} (k={k}, n={n_ers})")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "figures/ers_by_perturbation.png", dpi=160); plt.close(fig)

    g6 = "PASS" if ERS_OVERALL >= GATE_THRESHOLDS["G6_ers"]["min_ers"] else (
          "WARN" if ERS_OVERALL >= GATE_THRESHOLDS["G6_ers"]["warn_ers"] else "FAIL")
    print(f"\n[A23] Overall ERS: {ERS_OVERALL:.4f} -> G6 {g6}")'''))

CELLS_6.append(("code", r'''# ============ CELL: [A24] DTS — DECISION TRUST SCORE ============
with stage_agent("A24", "Decision Trust Score (flip-rate + confidence-stability composite)"):
    wF, wC = CFG["dts"]["weights"]
    dts_rows = []
    for pname in FLIP:
        flip = min(FLIP[pname], 1.0); dconf = min(DCONF[pname], 1.0)
        dts_p = wF * (1 - flip) + wC * (1 - dconf)
        dts_rows.append({"perturbation": pname, "flip_rate": round(flip, 4),
                         "mean_abs_conf_delta": round(dconf, 4), "DTS": round(dts_p, 4)})
    dts_df = pd.DataFrame(dts_rows)
    DTS_OVERALL = float(dts_df.DTS.mean())
    dts_df.loc[len(dts_df)] = {"perturbation": "OVERALL(mean)",
                               **{c: round(float(dts_df[c].mean()), 4) for c in ["flip_rate", "mean_abs_conf_delta", "DTS"]}}
    dts_df.to_csv(OUT / "ers/dts_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    d = dts_df[dts_df.perturbation != "OVERALL(mean)"].sort_values("DTS")
    ax.barh(d.perturbation, d.DTS, color="#2980b9", alpha=.85)
    ax.axvline(GATE_THRESHOLDS["G7_dts"]["min_dts"], color="#c0392b", ls="--", lw=1, label="gate threshold 0.70")
    ax.set(xlabel="DTS", xlim=(0, 1)); ax.set_title(f"Decision Trust Score — {best_name}")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "figures/dts_by_perturbation.png", dpi=160); plt.close(fig)

    G7_PASS = DTS_OVERALL >= GATE_THRESHOLDS["G7_dts"]["min_dts"]
    print(dts_df.to_string(index=False))
    print(f"\n[A24] Overall DTS: {DTS_OVERALL:.4f} -> G7 {'PASS' if G7_PASS else 'FAIL'}")'''))
