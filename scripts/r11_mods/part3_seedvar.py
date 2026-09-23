# Part 3: MOD 1.5 - Section 23B five-seed variance of the modelling->evaluation chain.
# Executed AFTER Section 39R (the notebook's own section order is non-monotonic: Section 29 already
# runs after Section 34) so that every representation-increment pair is available for check 1.

MD_23B = """## Section 23B — Multi-Seed Variance of the Modelling->Evaluation Chain (Revision 11, additive)

A single-seed headline number hides how much of it is stochastic. This section refits the frozen
Stage-A configuration of every primary run (same hyper-parameters, same TRAIN partition, same TEST
evaluation - only the model seed varies) with **five seeds** and reports mean +/- standard deviation
of the canonical metrics. The feature-extraction, audit and split stages are deterministic and are
deliberately NOT re-run: the variance measured here is exactly the variance a re-run of the
modelling->evaluation chain can exhibit. (This section is executed after the Section 39R ablations -
the notebook's own section order is already non-monotonic, e.g. Section 29 runs after Section 34 -
so that all representation-increment pairs below are already computed.)

Two explicit checks are reported:

1. **Seed-vs-increment check** - the F68-R vs F54-R and F54-R vs F48 TEST ROC-AUC increments (Section
   39R, full partitions) are compared against the cross-seed standard deviation. An increment smaller
   than the seed noise is reported as *not separable from seed variance* (this addresses the ~0.0007
   AUC increments seen in the revision-8 executed run).
2. **Original-seed anchor** - the first seed of each run is exactly the seed the Stage-A model was
   fitted with, so its refit must reproduce the Section 23 Stage-A TEST row bit-for-bit; the
   reproduction error is printed as a harness validation."""

CODE_23B = r'''# ===================================================================================================
# SECTION 23B (REVISION 11, MOD 1.5) - five-seed variance of the frozen modelling->evaluation chain.
# Additive only: the primary models above are untouched; these are separate refits.
# ===================================================================================================
SEEDVAR_ROWS = []
SEEDVAR_BUNDLE: Dict[str, Dict[str, Any]] = {}


def _seedvar_refit(rk, sd, tag):
    """One refit of the frozen Stage-A config with seed `sd` (TRAIN only, evaluated on TEST)."""
    def _compute():
        run = RUNS[rk]; src = run["source"]; kind = run["primary"]
        hp = run["best_params"][kind]
        tr = partition_index(src, "train")
        Xtr, ytr = get_X(rk, src, tr), CLEAN[src]["y"].values[tr]
        m = make_model(kind, hp["params"], rk, n_estimators=hp["n_estimators"])
        sd_eff = derived_seed("seedvar", rk, sd) if not tag else sd
        try:
            m.set_params(random_state=sd_eff)
        except ValueError:
            m.named_steps["lr"].set_params(random_state=sd_eff)
        t0 = time.time()
        m.fit(Xtr, ytr)
        LEDGER.record("rev11_seedvar_fit", f"{rk}#sd{tag or sd}", src, "train", "fit_model", len(tr),
                      "seed-variance refit")
        ti = partition_index(src, "test")
        Xte, yte = get_X(rk, src, ti), CLEAN[src]["y"].values[ti]
        p = model_proba(kind, m, Xte)
        out = {"seed": (tag or sd), "is_original_seed": bool(tag == "original"),
               "fit_seconds": round(time.time() - t0, 1),
               **classification_metrics(yte, p, run["thresholds"][kind])}
        del Xtr, Xte
        gc.collect()
        return out
    return r7_cache(f"seedvar_{rk.replace('|', '_')}_{tag or sd}", _compute)


for rk in [k for k in RUN_KEYS if k.endswith(PRIMARY_FSET)]:
    run = RUNS[rk]; kind = run["primary"]
    orig_seed = derived_seed("model", rk, kind)             # exactly the seed Stage A used
    seed_list = [("original", orig_seed)] + [(None, sd) for sd in CFG.revision11["seed_variance"]["seeds"]]
    rows = []
    for tag, sd in seed_list:
        r = _seedvar_refit(rk, sd, tag)
        rows.append(r)
        SEEDVAR_ROWS.append({"run": rk, **{k: v for k, v in r.items()
                                           if k not in ("seed", "is_original_seed", "fit_seconds")},
                             "seed": r["seed"], "is_original_seed": r["is_original_seed"]})
    df = pd.DataFrame(rows).drop(columns=["fit_seconds"])
    metric_cols = [c for c in CANONICAL_METRICS if c in df]
    SEEDVAR_BUNDLE[rk] = {
        "mean": {c: float(df[c].mean()) for c in metric_cols},
        "std": {c: float(df[c].std(ddof=1)) for c in metric_cols},
        "n_seeds": len(rows), "original_seed": int(orig_seed)}
    # harness validation: the original-seed refit must reproduce the Section 23 Stage-A TEST row
    try:
        ref = IN_DOMAIN_TEST[(IN_DOMAIN_TEST["run"] == rk) & (IN_DOMAIN_TEST["stage"] == "A")
                             & (IN_DOMAIN_TEST["primary"]) & (IN_DOMAIN_TEST["model"] == MODEL_NAMES[kind])
                             & (IN_DOMAIN_TEST["operating_point"] == "A balanced (MCC)")].iloc[0]
        r0 = df[df["is_original_seed"]].iloc[0]
        repro_err = max(abs(float(r0[c]) - float(ref[c])) for c in ("roc_auc", "pr_auc", "mcc", "f1"))
        print(f"[{rk}] original-seed reproduction error vs Section 23 Stage-A row: {repro_err:.2e}")
    except IndexError:
        repro_err = float("nan")
    SEEDVAR_BUNDLE[rk]["original_seed_reproduction_error"] = repro_err
    print(f"[{rk}] five-seed TEST ROC-AUC = {SEEDVAR_BUNDLE[rk]['mean']['roc_auc']:.4f} "
          f"+/- {SEEDVAR_BUNDLE[rk]['std']['roc_auc']:.4f}")

SEEDVAR_TABLE = pd.DataFrame(SEEDVAR_ROWS)
display(SEEDVAR_TABLE.round(5))
save_table(SEEDVAR_TABLE, "table23B1_revision11_seed_variance")

# ---- seed-vs-increment check (representation increments vs the seed noise) -------------------------
def _repr_test_auc(src, fs):
    sel = REPR_TABLE[(REPR_TABLE["source"] == CFG.datasets[src]["display"])
                     & (REPR_TABLE["fset_key"] == fs)
                     & (REPR_TABLE["population"] == "in_domain_test")]
    return float(sel["roc_auc"].iloc[0]) if len(sel) else np.nan

_seed_inc_rows = []
for src in ["gram", "phresh"]:
    std_auc = SEEDVAR_BUNDLE[f"{src}|{PRIMARY_FSET}"]["std"]["roc_auc"]
    for hi, lo in [(PRIMARY_FSET, "F54R"), ("F54R", "F48")]:
        a, b = _repr_test_auc(src, hi), _repr_test_auc(src, lo)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        delta = a - b
        _seed_inc_rows.append({
            "source": CFG.datasets[src]["display"],
            "increment": f"{hi} minus {lo} (TEST ROC-AUC, full partitions)",
            "delta_auc": round(delta, 5), "seed_std_auc": round(float(std_auc), 5),
            "increment_below_seed_std": bool(abs(delta) < std_auc),
            "verdict": ("NOT SEPARABLE FROM SEED VARIANCE" if abs(delta) < std_auc
                        else "exceeds seed noise")})
SEED_INCREMENT_TABLE = pd.DataFrame(_seed_inc_rows)
display(SEED_INCREMENT_TABLE)
save_table(SEED_INCREMENT_TABLE, "table23B2_revision11_seed_vs_increment")
R11_SEED_STD = {rk: SEEDVAR_BUNDLE[rk]["std"]["roc_auc"] for rk in SEEDVAR_BUNDLE}
print("Revision-11 explicit check: any AUC increment whose absolute value is smaller than the "
      "cross-seed standard deviation is reported as not separable from seed variance.")
print(json.dumps({k: round(v, 5) for k, v in R11_SEED_STD.items()}, indent=2))'''
