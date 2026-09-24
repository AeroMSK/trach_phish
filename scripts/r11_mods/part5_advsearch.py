# Part 5: MOD 1.4 - Section 29B greedy/hill-climb adversarial composition search over existing operators.

MD_29B = """## Section 29B — Greedy Adversarial Composition Search (Revision 11, additive)

Section 29 applies each stress family *once, in isolation*. A patient adversary composes them. This
section runs a deterministic **greedy hill-climb** over the existing perturbation operators
(P5-P9 at every severity), with the **Class I identity constraint enforced at every step** (a
candidate is only accepted if its registered domain still equals the parent's - the `regdom_equal`
check of the perturbation engine, so P8/P9 variants that change the registrable identity are
automatically discarded):

* **Parents** - the first `n_parents` phishing URLs of the in-domain TEST stress sample (deterministic
  order, defined ERS not required).
* **Step** - all (family, severity) stress transforms are applied to the *current* URL; the
  identity-preserving candidate that minimises the calibrated phishing probability is accepted if it
  improves by more than `1e-4`; up to `max_steps` (5) steps.
* **Objective** - minimise calibrated phishing confidence of the **Stage-B primary detector** (the
  model actually shipped). No SHAP/ERS is recomputed (that is the Section 28/29 instrument, not the
  attack); we report what Section 29 reports: **flip rate at the frozen threshold, calibrated
  confidence shift, and per-family usage** of the accepted moves.

This is an *attack simulation on the frozen detector*, evaluated only on labelled phishing test URLs;
it never feeds back into training (the robustness augmentation of Phase B remains untouched)."""

CODE_29B = r'''# ===================================================================================================
# SECTION 29B (REVISION 11, MOD 1.4) - greedy composition search over the existing stress operators.
# Additive: the frozen Stage-B detector is attacked, never retrained. Class I identity is enforced.
# ===================================================================================================
R11ADV = CFG.revision11["adv_search"]
ADV_ROWS, ADV_EXAMPLES = [], []


def _adv_search_run(rk):
    """Greedy hill-climb per parent; batched feature extraction across parents at every step."""
    def _compute():
        run = RUNS[rk]; kind = run["primary"]; T = POPS[(rk, "test")]
        par_pool = np.flatnonzero(T["y"] == 1)
        parents = par_pool[:R11ADV["n_parents"]]
        uid_list = [str(u) for u in T["uids"][parents]]
        p_cur = run["calibrator_B"].predict(model_proba(kind, run["model_B"], T["X"][parents])).astype(np.float64)
        p_init = p_cur.copy()
        urls_cur = list(T["urls"][parents])
        pos_of = {u: i for i, u in enumerate(uid_list)}
        steps_used = np.zeros(len(parents), dtype=int)
        fam_use: Dict[str, int] = {}
        accepted_total = 0
        examples = []
        for step in range(R11ADV["max_steps"]):
            step_uids = [f"{u}#s{step}" for u in uid_list]
            recs = generate_perturbations(step_uids, urls_cur, CFG.perturbation["stress_families"],
                                          derived_seed("adv", rk, step))
            v = recs[recs["valid"] & recs["regdom_equal"]].reset_index(drop=True)   # Class I identity constraint
            if not len(v):
                break
            Xc = run["imputer"].transform(features_for_set(v["generated_url"].values, run["fset"]))
            v["pc"] = run["calibrator_B"].predict(model_proba(kind, run["model_B"], Xc))
            best_idx = v.groupby("source_uid")["pc"].idxmin()
            moved = 0
            for bi in best_idx:
                row = v.loc[bi]
                i = pos_of[str(row["source_uid"]).rsplit("#s", 1)[0]]
                if row["pc"] < p_cur[i] - R11ADV["improvement_eps"]:
                    if len(examples) < 3 and steps_used[i] == 0:
                        examples.append({"parent_url": urls_cur[i], "attacked_url": row["generated_url"],
                                         "family": row["family"], "severity": int(row["severity"]),
                                         "p_before": round(float(p_cur[i]), 4), "p_after": round(float(row["pc"]), 4)})
                    urls_cur[i] = row["generated_url"]
                    p_cur[i] = float(row["pc"])
                    steps_used[i] += 1
                    fam_use[str(row["family"])] = fam_use.get(str(row["family"]), 0) + 1
                    moved += 1
            accepted_total += moved
            if moved == 0:
                break
        thr = float(run["threshold_B"])
        yhat_fin = (p_cur >= thr).astype(int)
        flips = int((yhat_fin == 0).sum())                       # parents are all phishing: phish->benign
        _, C_init = confidence_from_p(p_init)
        _, C_fin = confidence_from_p(p_cur)
        out = {"run": rk, "n_parents": int(len(parents)),
               "prediction_flip_pct": 100.0 * flips / max(len(parents), 1),
               "phish_to_benign_flip_pct": 100.0 * flips / max(len(parents), 1),
               "mean_delta_C": float(np.mean(C_fin - C_init)),
               "mean_delta_p_cal_phish": float(np.mean(p_cur - p_init)),
               "median_steps_used": float(np.median(steps_used)),
               "mean_steps_used": float(np.mean(steps_used)),
               "steps_to_flip_median": float(np.median(steps_used[yhat_fin == 0])) if flips else np.nan,
               "accepted_moves": accepted_total, "family_usage": fam_use,
               "mean_final_p_cal": float(np.mean(p_cur)), "examples": examples}
        return out
    return r7_cache(f"r11_advsearch_{rk.replace('|', '_')}", _compute)


for rk in [k for k in RUN_KEYS if k.endswith(PRIMARY_FSET)]:
    res = _adv_search_run(rk)
    ADV_ROWS.append({k: v for k, v in res.items() if k not in ("family_usage", "examples")})
    ADV_EXAMPLES += [{"run": rk, **e} for e in res["examples"]]
    fam = pd.DataFrame([{"run": rk, "family": f, "accepted_moves": n}
                        for f, n in sorted(res["family_usage"].items(), key=lambda kv: -kv[1])])
    display(fam)
    print(f"[{rk}] greedy adversarial search: flip rate {res['prediction_flip_pct']:.1f}% "
          f"({int(round(res['prediction_flip_pct'] * res['n_parents'] / 100))}/{res['n_parents']} phishing "
          f"parents flipped), mean dC {res['mean_delta_C']:+.4f}, median steps {res['median_steps_used']:.0f}")

ADV_TABLE = pd.DataFrame(ADV_ROWS)
display(ADV_TABLE.round(4))
save_table(ADV_TABLE, "table29B_revision11_greedy_adversarial_search")
ADV_EXAMPLES_DF = pd.DataFrame(ADV_EXAMPLES)
ADV_EXAMPLES_DF.to_csv(DIRS["reports"] / "revision11_adv_search_examples.csv", index=False)
save_json({"config": R11ADV, "results": {r["run"]: r for r in ADV_ROWS}},
          DIRS["metadata"] / "revision11_adv_search.json")
print("Section 29B complete: identity-preserving greedy composition measured against the frozen "
      "Stage-B detector. Reported in the Section 29 format (flip rate + confidence shift); "
      "no training artifact was modified.")'''
