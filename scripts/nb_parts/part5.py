# -*- coding: utf-8 -*-
"""Part 5: A19 transfer, A20 robustness, A21 calibration."""

CELLS_5 = []

CELLS_5.append(("code", r'''# ============ CELL: [A19] TRANSFER — PHRESHPHISH 2026 ZERO-SHOT ============
def eval_urls(urls: pd.Series):
    """Predict + score with the selected best model on new URLs (dispatches representation)."""
    if BEST_IS_TFIDF:
        X = chunked_transform(vec, urls.reset_index(drop=True), 50000)
        p = BEST.predict(X)
        s = BEST.predict_proba(X)[:, 1] if hasattr(BEST, "predict_proba") else BEST.decision_function(X)
    else:
        Fp = lexical_features(urls.reset_index(drop=True))
        p = BEST.predict(Fp.values); s = BEST.predict_proba(Fp.values)[:, 1]
    return p.astype(np.int8), s.astype(np.float32)

with stage_agent("A19", "Zero-shot transfer to PhreshPhish 2026 (URL-only, temporal-shift)"):
    p_pp, s_pp = eval_urls(pp_test.url)
    TRANSFER_METRICS = clf_metrics(y_pp_te, p_pp, s_pp)
    np.savez_compressed(OUT / "predictions/pp_test_predictions.npz", y=y_pp_te, pred=p_pp, score=s_pp)

    pd.DataFrame([{"model": best_name, "eval": "PhreshPhish-2026 zero-shot", **TRANSFER_METRICS},
                  {"model": best_name, "eval": "GB test (in-corpus)", **MAIN_METRICS}]).to_csv(
        OUT / "tables/transfer_results.csv", index=False)

    # out-of-vocabulary gram rate on a 20k sample (TF-IDF models only)
    if BEST_IS_TFIDF:
        rs = np.random.RandomState(SEED)
        smp = pp_test.url.iloc[rs.choice(len(pp_test), 20000, replace=False)]
        grams = vec.build_analyzer()
        known = set(Vocab.keys()); tot = 0; oov = 0
        for u in smp:
            gs = grams(u)
            tot += len(gs); oov += sum(1 for g in gs if g not in known)
        OOV_RATE = oov / max(tot, 1)
        print(f"    OOV char-ngram rate on PP test (20k sample): {OOV_RATE:.3%}")

    # covariate drift: KS tests on lexical features (GB test vs PP test, 20k samples each)
    ks_rows = []
    for c in LEX_COLS:
        a = FX["te"][c].iloc[:20000].values; b = FX["pp"][c].iloc[:20000].values
        stat, p = ks_2samp(a, b)
        ks_rows.append({"feature": c, "ks_stat": round(float(stat), 4), "p_value": float(p)})
    ks_df = pd.DataFrame(ks_rows).sort_values("ks_stat", ascending=False)
    ks_df.to_csv(OUT / "tables/transfer_drift_ks.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    fpr, tpr, _ = roc_curve(y_pp_te, s_pp); ax[0].plot(fpr, tpr, lw=1.8, color="#8e44ad")
    ax[0].set(xlabel="FPR", ylabel="TPR", title=f"ROC — PhreshPhish 2026 zero-shot (AUC={TRANSFER_METRICS['roc_auc']:.4f})")
    pr, rc, _ = precision_recall_curve(y_pp_te, s_pp); ax[1].plot(rc, pr, lw=1.8, color="#8e44ad")
    ax[1].set(xlabel="recall", ylabel="precision", title=f"PR — PhreshPhish (AP={TRANSFER_METRICS['pr_auc']:.4f})")
    fig.savefig(OUT / "figures/transfer_curves.png", dpi=160); plt.close(fig)

    drop = {k: round(MAIN_METRICS[k] - TRANSFER_METRICS[k], 4) for k in ("f1", "roc_auc", "accuracy")}
    G4_PASS = TRANSFER_METRICS["f1"] >= GATE_THRESHOLDS["G4_transfer_floor"]["min_pp_f1"]
    print(f"[A19] Transfer metrics: {TRANSFER_METRICS}")
    print(f"[A19] Drop (GB-test minus PP-test): {drop}")
    print(f"[A19] Top drift features: {ks_df.head(5)[['feature','ks_stat']].values.tolist()}")
    print(f"[A19] G4 transfer floor gate (F1>={GATE_THRESHOLDS['G4_transfer_floor']['min_pp_f1']}): {'PASS' if G4_PASS else 'FAIL'}")'''))

CELLS_5.append(("code", r'''# ============ CELL: [A20] ROBUSTNESS — PERTURBATION SUITE ============
def _rand_label(rs): return "".join(rs.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), 8))

def make_perturbations(urls: pd.Series, seed=SEED):
    rs = np.random.RandomState(seed)
    def case_random(u):
        return "".join(c.upper() if rs.rand() < .5 else c.lower() for c in u)
    def _split_host(u):
        m = re.match(r"^(.*://)([^/]*)(.*)$", u, re.DOTALL)
        if m: return m.group(1), m.group(2), m.group(3)
        host = u.split("/", 1)[0]
        return "", host, u[len(host):]
    def typo_swap(u):
        pre, host, rest = _split_host(u)
        if len(host) >= 4:
            i = rs.randint(1, len(host) - 2); host = host[:i] + host[i + 1] + host[i] + host[i + 2:]
        return pre + host + rest
    def typo_delete(u):
        pre, host, rest = _split_host(u)
        if len(host) >= 6:
            i = rs.randint(1, len(host) - 1); host = host[:i] + host[i + 1:]
        return pre + host + rest
    def pad_benign(u):  return u.rstrip("/") + "/blog/posts"
    def scheme_flip(u): return u.replace("https://", "http://", 1) if u.startswith("https://") else u.replace("http://", "https://", 1)
    def www_toggle(u):
        m = re.match(r"^(.*://)(www\.)?([^/]*)(.*)$", u, re.DOTALL)
        if not m: return u
        if m.group(2): return m.group(1) + m.group(3) + m.group(4)
        return m.group(1) + "www." + m.group(3) + m.group(4)
    def subdomain_junk(u):
        pre, host, rest = _split_host(u)
        if "." not in host: return u
        return pre + _rand_label(rs) + "." + host + rest
    def query_junk(u):  return u + "?utm_source=newsletter&utm_medium=email&utm_campaign=2026"
    return {"case_random": urls.map(case_random), "typo_swap": urls.map(typo_swap),
            "typo_delete": urls.map(typo_delete), "pad_benign": urls.map(pad_benign),
            "scheme_flip": urls.map(scheme_flip), "www_toggle": urls.map(www_toggle),
            "subdomain_junk": urls.map(subdomain_junk), "query_junk": urls.map(query_junk)}

with stage_agent("A20", f"Perturbation robustness (stratified {CFG['robustness']['sample']:,}-URL test sample, 8 attacks)"):
    rs = np.random.RandomState(SEED)
    n_s = CFG["robustness"]["sample"]
    per_class = n_s // 2
    pos_i = rs.choice(np.where(y_gb_te == 1)[0], per_class, replace=False)
    neg_i = rs.choice(np.where(y_gb_te == 0)[0], n_s - per_class, replace=False)
    rob_idx = np.concatenate([pos_i, neg_i]); rs.shuffle(rob_idx)
    rob_urls = gb_test.url.iloc[rob_idx].reset_index(drop=True)
    y_rob = y_gb_te[rob_idx]
    p0, s0 = eval_urls(rob_urls)
    base_m = clf_metrics(y_rob, p0, s0)

    pert = make_perturbations(rob_urls, seed=SEED)
    rob_rows = []
    FLIP, DCONF = {}, {}
    for pname, puri in pert.items():
        pp_, ss_ = eval_urls(puri)
        flip = float((pp_ != p0).mean()); dconf = float(np.abs(ss_.astype(np.float64) - s0.astype(np.float64)).mean())
        FLIP[pname], DCONF[pname] = flip, dconf
        rob_rows.append({"perturbation": pname, "flip_rate": round(flip, 4),
                         "mean_abs_conf_delta": round(dconf, 4),
                         **clf_metrics(y_rob, pp_, ss_, prefix="pert_")})
    rob_df = pd.DataFrame(rob_rows)
    rob_df["delta_f1_vs_base"] = (rob_df.pert_f1 - base_m["f1"]).round(4)
    rob_df.to_csv(OUT / "perturbation/robustness_suite.csv", index=False)
    pd.DataFrame([{"perturbation": "(base=none)", **base_m}]).to_csv(OUT / "perturbation/robustness_base.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    rr = rob_df.sort_values("flip_rate")
    ax.barh(rr.perturbation, rr.flip_rate, color="#c0392b", alpha=.8)
    ax.set(xlabel="prediction flip rate", title=f"Robustness — {best_name} (n={n_s:,} stratified GB-test sample)")
    for y_, v in enumerate(rr.flip_rate): ax.text(v, y_, f" {v:.3f}", va="center", fontsize=8)
    fig.savefig(OUT / "figures/robustness_flip_rates.png", dpi=160); plt.close(fig)

    print(rob_df[["perturbation", "flip_rate", "mean_abs_conf_delta", "delta_f1_vs_base"]].to_string(index=False))
    print(f"\n[A20] Base sample metrics: {base_m}")'''))

CELLS_5.append(("code", r'''# ============ CELL: [A21] CALIBRATION (Platt / isotonic fit on VAL; test touched once) ============
from sklearn.isotonic import IsotonicRegression

def ece_bins(y, p, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0; mce = 0.0; rows = []
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= 0) & (p <= edges[1])
        if m.sum() == 0: continue
        conf = float(p[m].mean()); acc = float(y[m].mean())
        ece += m.mean() * abs(conf - acc); mce = max(mce, abs(conf - acc))
        rows.append((f"({edges[i]:.2f},{edges[i+1]:.2f}]", int(m.sum()), round(conf, 4), round(acc, 4)))
    return float(ece), float(mce), rows

with stage_agent("A21", "Probability calibration (Platt & isotonic, validation-fit)"):
    if Xva_tfidf is None:   # released after A15 for memory headroom; lazy-reload for calibration
        Xva_tfidf = ckpt_load("tfidf_Xva", extra={"tfidf": CFG["tfidf"]})
    # raw scores on val + test for the best model
    if BEST_IS_TFIDF:
        s_va_raw = BEST.predict_proba(Xva_tfidf)[:, 1] if hasattr(BEST, "predict_proba") else BEST.decision_function(Xva_tfidf)
    else:
        s_va_raw = BEST.predict_proba(FX["va"].values)[:, 1]
    s_va_raw = s_va_raw.astype(np.float64)
    s_te_raw = SCORE_TE[best_name].astype(np.float64)
    s_pp_raw = s_pp.astype(np.float64)

    # squeeze decision values into (0,1) for Platt input
    def squeeze(s): return 1 / (1 + np.exp(-np.clip(s, -30, 30))) if s.min() < 0 or s.max() > 1 else s.copy()
    z_va = squeeze(s_va_raw)

    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(z_va.reshape(-1, 1), y_va)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(z_va, y_va)

    cand = {"raw": z_va, "platt": platt.predict_proba(z_va.reshape(-1, 1))[:, 1], "isotonic": iso.predict(z_va)}
    val_cal = {k: ece_bins(y_va, v)[:2] + (float(np.mean((v - y_va) ** 2)),) for k, v in cand.items()}
    chosen = min(val_cal, key=lambda k: val_cal[k][0])
    CALIB_METHOD = chosen

    def apply_cal(s):
        z = squeeze(s)
        if chosen == "platt": return platt.predict_proba(z.reshape(-1, 1))[:, 1]
        if chosen == "isotonic": return iso.predict(z)
        return z

    cal_rows = []
    for nm, (yv, sv) in [("GB test", (y_gb_te, s_te_raw)), ("PhreshPhish test", (y_pp_te, s_pp_raw))]:
        z = squeeze(sv); pc = apply_cal(sv)
        e_raw, m_raw, _ = ece_bins(yv, z); e_cal, m_cal, bins = ece_bins(yv, pc)
        cal_rows.append({"eval_set": nm, "method_pre": "raw", "ece": round(e_raw, 4), "mce": round(m_raw, 4),
                         "brier": round(float(np.mean((z - yv) ** 2)), 4)})
        cal_rows.append({"eval_set": nm, "method_pre": chosen, "ece": round(e_cal, 4), "mce": round(m_cal, 4),
                         "brier": round(float(np.mean((pc - yv) ** 2)), 4)})
        if nm == "GB test":
            GB_BINS_RAW, GB_BINS_CAL, P_TE_CAL = bins, ece_bins(yv, pc)[2], pc
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(OUT / "calibration/calibration_results.csv", index=False)
    POST_ECE = [r["ece"] for r in cal_rows if r["eval_set"] == "GB test" and r["method_pre"] == chosen][0]
    G5_PASS = POST_ECE <= GATE_THRESHOLDS["G5_calibration"]["max_post_cal_ece"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax_, bins_, ttl in [(ax[0], GB_BINS_RAW, "raw"), (ax[1], GB_BINS_CAL, f"{chosen} (val-fit)")]:
        if not bins_: continue
        b = pd.DataFrame(bins_, columns=["bin", "n", "conf", "acc"])
        ax_.bar(range(len(b)), b.conf, alpha=.35, color="#2980b9", label="mean confidence")
        ax_.bar(range(len(b)), b.acc, alpha=.55, color="#27ae60", label="fraction phish")
        ax_.set(xlabel="probability bin", ylabel="value", title=f"Reliability — {ttl}")
        ax_.legend(fontsize=8)
    fig.savefig(OUT / "figures/calibration_reliability.png", dpi=160); plt.close(fig)

    print(cal_df.to_string(index=False))
    print(f"\n[A21] Val calibration candidates (ece,mce,brier): {val_cal} -> chosen: {chosen}")
    print(f"[A21] G5 calibration gate (post-cal ECE<= {GATE_THRESHOLDS['G5_calibration']['max_post_cal_ece']}): {'PASS' if G5_PASS else 'FAIL'}")'''))
