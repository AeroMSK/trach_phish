# Part 6: MOD 1.1 - Section 40B base-rate evaluation (rejection resampling, evaluation-only)
#          + MOD 1.3 extension: CORAL rows appended to the Section 40 transfer table.

MD_40B = """## Section 40B — Realistic Base-Rate Evaluation (Revision 11, additive)

Both corpora are ~50/50 balanced, so every precision number above is a *balanced-prevalence* number.
A deployed phishing filter sees base rates between **0.05% and 5%**. This section re-scores the
**already-scored** full-population queues (in-domain TEST and the principal strict-external population;
identical frozen scores, **no refit, no re-selection, no threshold change**) under simulated base rates
via rejection resampling:

* For each nominal base rate the replicate fixes `n_neg = 200,000` negative scores (drawn with
  replacement from the empirical negative-score pool) and `n_pos = round(rate * n_neg / (1 - rate))`
  positives (subsampled from the real positive pool); **200 replicates**; every replicate recomputes
  ROC-AUC, average precision (AP), precision at recall 0.5/0.7/0.9, and the precision/recall/FPR at
  the **frozen** in-domain operating threshold. The effective base rate of each draw is reported.
* Reported per (run, population, rate): the replicate mean and the 2.5/97.5 percentiles. This is an
  **evaluation-only** analysis of score distributions: the AP of a replicate depends only on the score
  ranking, which resampling does not change, but precision-type metrics collapse exactly as they do in
  deployment - that collapse is the finding.
* Criteria **G** and **H** (pre-registered in `CFG.criteria` above, appended, nothing overwritten) are
  evaluated automatically in Section 48 and mirrored in the Section 54 summary.

## Section 40C — Phase H models in the transfer table

The CORAL / class-conditional-MMD models of Phase H are appended to the Section 40 transfer table as
additional rows (`branch = uda-coral / uda-cc-mmd`, `primary = False`), so the consolidated
`table05_cross_dataset_performance` carries them alongside every other transfer model."""


CODE_40B = r'''# ===================================================================================================
# SECTION 40B (REVISION 11, MOD 1.1) - base-rate evaluation by rejection resampling. Evaluation-only.
# ===================================================================================================
R11BR = CFG.revision11["baserate"]
N_NEG_FIXED = R11BR["max_neg_draw"]
BASERATE_ROWS = []
BASERATE_SUMMARY: Dict[str, Any] = {}


def _br_replicate_metrics(ps, ns, thr, recalls):
    """Metrics of one resampled replicate. Group-end step AP (as in the notebook's bootstrap,
    ties accepted/rejected together), rank AUC, and FIRST-CROSSING precision at recall r
    (the operational definition: the precision of the loosest threshold that still reaches r)."""
    s = np.concatenate([ps, ns])
    y = np.concatenate([np.ones(len(ps), dtype=np.float64), np.zeros(len(ns), dtype=np.float64)])
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    n = len(s)
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), n - 1]      # last index of each tie group
    tp = np.cumsum(ys)[ends]
    fp = np.cumsum(1.0 - ys)[ends]
    P, N = float(len(ps)), float(len(ns))
    rec_g = tp / P
    prec_g = tp / np.maximum(tp + fp, 1.0)
    out = {"roc_auc": float(fast_auc(y, s)),
           "average_precision": float(np.sum(np.diff(np.r_[0.0, rec_g]) * prec_g))}
    for r in recalls:
        k = np.flatnonzero(rec_g >= r)
        out[f"precision_at_recall_{r:g}"] = float(prec_g[k[0]]) if k.size else 0.0
    yhat = (s >= thr).astype(np.float64)
    TP = float(yhat[y == 1].sum()); FP = float(yhat[y == 0].sum())
    out["precision_at_frozen_threshold"] = TP / max(TP + FP, 1.0)
    out["recall_at_frozen_threshold"] = TP / max(P, 1.0)
    out["fpr_at_frozen_threshold"] = FP / max(N, 1.0)
    return out


def _baserate_population(rk, population):
    """(y, p_cal, threshold) for an already-scored full population queue."""
    run = RUNS[rk]; src = run["source"]
    if population == "in_domain_test":
        idx = run["test_idx"]
        return CLEAN[src]["y"].values[idx], run["test_p_cal_B"], float(run["threshold_B"])
    tgt = run["target"]
    idx = np.flatnonzero(EXTERNAL_MASKS[(src, tgt)][CFG.primary_external_view])
    p = run.get("ext_p_cal_B")
    if p is None:
        p = run["calibrator_B"].predict(model_proba(run["primary"], run["model_B"], get_X(rk, tgt, idx)))
    return CLEAN[tgt]["y"].values[idx], p, float(run["threshold_B"])


for rk in [k for k in RUN_KEYS if k.endswith(PRIMARY_FSET)]:
    run = RUNS[rk]
    for population in ["in_domain_test", "external_principal"]:
        y, p, thr = _baserate_population(rk, population)
        pos_pool, neg_pool = p[y == 1].astype(np.float64), p[y == 0].astype(np.float64)
        label = ("in-domain TEST" if population == "in_domain_test"
                 else f"strict-external ({CFG.datasets[run['target']]['display']})")
        print(f"[{rk}] base-rate simulation on {label}: {len(pos_pool):,} phishing / {len(neg_pool):,} benign "
              f"scores, frozen threshold {thr:.4f}")
        for rate in R11BR["rates"]:
            n_pos = int(round(rate * N_NEG_FIXED / (1.0 - rate)))
            n_pos = max(1, min(n_pos, len(pos_pool)))
            rep_metrics = []
            for b in range(R11BR["replicates"]):
                rng = np.random.default_rng(derived_seed("baserate", rk, population, rate, b))
                ps = rng.choice(pos_pool, n_pos, replace=len(pos_pool) < n_pos)
                ns = neg_pool[rng.integers(0, len(neg_pool), N_NEG_FIXED)]
                rep_metrics.append(_br_replicate_metrics(ps, ns, thr, R11BR["recalls"]))
            eff_rate = n_pos / (n_pos + N_NEG_FIXED)
            for metric in rep_metrics[0]:
                vals = np.array([m[metric] for m in rep_metrics])
                BASERATE_ROWS.append({
                    "run": rk, "population": population, "population_label": label,
                    "nominal_base_rate": rate, "effective_base_rate": round(eff_rate, 6),
                    "n_pos": n_pos, "n_neg": N_NEG_FIXED, "replicates": R11BR["replicates"],
                    "metric": metric, "mean": float(vals.mean()),
                    "ci_low": float(np.quantile(vals, 0.025)), "ci_high": float(np.quantile(vals, 0.975))})
            _g = [r for r in BASERATE_ROWS if r["metric"] == "precision_at_recall_0.7"][-1]
            print(f"    rate {rate:g} (effective {eff_rate:.4%}): precision@recall0.7 = "
                  f"{_g['mean']:.4f} [{_g['ci_low']:.4f}, {_g['ci_high']:.4f}]")

BASERATE_TABLE = pd.DataFrame(BASERATE_ROWS)
display(BASERATE_TABLE.pivot_table(index=["run", "population", "nominal_base_rate"],
                                   columns="metric", values="mean").round(4))
save_table(BASERATE_TABLE, "table40B_revision11_base_rate_evaluation")
BASERATE_PIVOT = (BASERATE_TABLE[BASERATE_TABLE["metric"].isin(
    ["average_precision", "precision_at_recall_0.7", "precision_at_frozen_threshold", "roc_auc"])]
    .pivot_table(index=["run", "population", "nominal_base_rate"], columns="metric", values="mean"))
BASERATE_PIVOT.to_csv(DIRS["tables"] / "table40B_revision11_base_rate_pivot.csv", index=True)

# ---- criteria G / H inputs (evaluated formally in Section 48) ---------------------------------------
_gh = {}
for rk in [k for k in RUN_KEYS if k.endswith(PRIMARY_FSET)]:
    _gh[rk] = {}
    for rate, key in [(0.01, "G_mean_precision_at_recall_0.7"), (0.001, "H_mean_precision_at_recall_0.7")]:
        sel = BASERATE_TABLE[(BASERATE_TABLE["run"] == rk) & (BASERATE_TABLE["population"] == "external_principal")
                             & np.isclose(BASERATE_TABLE["nominal_base_rate"], rate)
                             & (BASERATE_TABLE["metric"] == "precision_at_recall_0.7")]
        _gh[rk][key] = float(sel["mean"].iloc[0]) if len(sel) else np.nan
BASERATE_SUMMARY["external_precision_at_recall_0.7"] = _gh
BASERATE_SUMMARY["config"] = {"rates": R11BR["rates"], "replicates": R11BR["replicates"],
                              "n_neg_fixed": N_NEG_FIXED, "recalls": R11BR["recalls"],
                              "protocol": "evaluation-only rejection resampling on frozen full-population "
                                          "calibrated scores; no refit, no re-selection, no threshold change"}
save_json(BASERATE_SUMMARY, DIRS["metadata"] / "revision11_base_rate_summary.json")

# ---- Section 46-style figure: precision decay vs base rate ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True, constrained_layout=True)
for ax, metric, title in zip(axes, ["precision_at_recall_0.7", "precision_at_frozen_threshold"],
                              ["Precision at recall >= 0.70 (mean over replicates)",
                               "Precision at the FROZEN in-domain threshold"]):
    for (rk, pop), g in BASERATE_TABLE[BASERATE_TABLE["metric"] == metric].groupby(["run", "population"]):
        ls = "-" if pop == "external_principal" else "--"
        ax.plot(g["nominal_base_rate"], g["mean"], ls, marker="o",
                label=f"{rk} [{'external' if pop == 'external_principal' else 'test'}]")
    ax.set_xscale("log"); ax.set_xlabel("simulated phishing base rate (log)")
    ax.set_ylabel(title.split(" (")[0]); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02)
axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.0))
fig.suptitle("Revision 11 - base-rate degradation of the frozen Stage-B detector "
             "(rejection resampling, 200 replicates per rate)")
fig.savefig(DIRS["figures"] / "revision11_base_rate_decay.png", dpi=150)
plt.show()
print("Section 40B complete: base-rate table, criteria G/H inputs and degradation figure written.")

# ===================================================================================================
# SECTION 40C (REVISION 11, MOD 1.3) - Phase H alignment models appended to the transfer table.
# ===================================================================================================
_XFER_BEFORE = len(XFER_ROWS)
for (src, fset, variant), res in CORAL_RESULTS.items():
    rk_ref = f"{src}|{PRIMARY_FSET}"
    in_dom = IN_DOMAIN_TEST[(IN_DOMAIN_TEST["run"] == rk_ref) & IN_DOMAIN_TEST["primary"]
                            & (IN_DOMAIN_TEST["stage"] == "B")
                            & (IN_DOMAIN_TEST["operating_point"] == "A balanced (MCC)")].iloc[0]
    me = classification_metrics(res["y_ext"], res["p_ext"], res["thr"])
    XFER_ROWS.append({"run": f"{src}|{fset}", "direction": res["direction"],
                      "view": CFG.primary_external_view, "principal": True, "stage": "H",
                      "model": res["name"], "branch": f"uda-{variant}", "primary": False,
                      "operating_point": "tau selected on ALIGNED source validation",
                      **me,
                      "delta_roc_auc_vs_in_domain": me["roc_auc"] - in_dom["roc_auc"],
                      "delta_mcc_vs_in_domain": me["mcc"] - in_dom["mcc"],
                      "delta_f1_vs_in_domain": me["f1"] - in_dom["f1"]})
XFER_TABLE = pd.DataFrame(XFER_ROWS)
display(XFER_TABLE[XFER_TABLE["stage"] == "H"][["run", "direction", "model", "branch", "operating_point"]
                                                + CANONICAL_METRICS + ["delta_roc_auc_vs_in_domain"]].round(4))
print(f"Section 40C complete: {len(XFER_ROWS) - _XFER_BEFORE} Phase-H rows appended to the transfer table "
      f"(branch labels uda-coral / uda-cc-mmd; primary=False). table05 will export them.")'''
