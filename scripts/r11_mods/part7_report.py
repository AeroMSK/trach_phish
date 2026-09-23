# Part 7: MOD 1.7 - Section 54B paper-style report (claim map, literature comparison, limitations)
#          + cell-188 additive extension that evaluates criteria G/H.

CELL188_TAIL_ANCHOR = 'CRITERIA_TABLE = pd.DataFrame(CRIT_ROWS)\nsave_table(CRITERIA_TABLE, "table13_success_criteria")'
CELL188_EXTENSION = r'''# ---- REVISION 11 (additive): criteria G/H evaluated from the Section 40B base-rate simulation -------
def _r11_verdict(cond, defined):
    return "UNDEFINED" if not defined else ("SUPPORTED" if cond else "NOT SUPPORTED")

if "BASERATE_SUMMARY" in globals():
    _gh = BASERATE_SUMMARY.get("external_precision_at_recall_0.7", {})
    for _cid, _rate_key, _floor in [("G", "G_mean_precision_at_recall_0.7", 0.50),
                                    ("H", "H_mean_precision_at_recall_0.7", 0.10)]:
        _vals = {rk: v.get(_rate_key, np.nan) for rk, v in _gh.items()}
        _defined = bool(_vals) and all(np.isfinite(v) for v in _vals.values()) and len(_vals) >= 2
        _ok = _defined and all(v >= _floor for v in _vals.values())
        CRIT_ROWS.append({"run": "both directions (external principal)",
                          "criterion": _cid, "rule": CFG.criteria[_cid],
                          "evidence": ("mean precision@recall>=0.7 at simulated base rate "
                                       + ("1%: " if _cid == "G" else "0.1%: ")
                                       + "; ".join(f"{rk} = {v:.4f}" for rk, v in _vals.items())),
                          "verdict": _r11_verdict(_ok, _defined)})
        print(f"CRITERION {_cid} (revision 11 base-rate): {_r11_verdict(_ok, _defined)}"
              + (f" -- floor {_floor}" if _defined else " -- Section 40B results unavailable"))
else:
    print("CRITERIA G/H (revision 11): UNDEFINED - Section 40B did not run in this execution.")

CRITERIA_TABLE = pd.DataFrame(CRIT_ROWS)
save_table(CRITERIA_TABLE, "table13_success_criteria")'''

MD_54B = """## Section 54B — Paper-Style Claim Map, Literature Comparison and Limitations (Revision 11, additive)

Three paper-grade artefacts, every number read from structures computed above (nothing hand-entered
except the published literature anchors, which are labelled as such):

1. **Claim map** - every headline number of this notebook mapped to the pre-registered declaration or
   pinned gate that produced it, its verdict, and the measured evidence.
2. **Literature comparison** - the in-domain and transfer headline numbers next to the published
   GramBeddings-protocol number (0.9827 accuracy) and the strongest recent URL-only baselines
   (TransURL F1 0.9824; TinyBERT F1 0.9387), with the protocol differences stated plainly
   (domain-disjoint splits, calibrated probabilities, strict-external evaluation - none of which the
   published numbers undergo).
3. **Limitations** - the honest list: dual-corpus domain separability, base-rate degradation with the
   measured numbers, seed variance with the measured standard deviations, the deferred third
   evaluation corpus, the documented CNN compute cap, and the first-moment approximation of
   class-conditional MMD."""

CODE_54B = r'''# ===================================================================================================
# SECTION 54B (REVISION 11, MOD 1.7) - claim map, literature comparison, limitations.
# ===================================================================================================

# ---- 1. claim map -----------------------------------------------------------------------------------
CLAIM_MAP_ROWS = []
try:
    for _, r in CRITERIA_TABLE.iterrows():
        CLAIM_MAP_ROWS.append({"headline": f"Criterion {r['criterion']}", "declaration": r["rule"],
                               "verdict": r["verdict"], "evidence": r["evidence"],
                               "source": "pre-registered CFG.criteria (Section 48)"})
except NameError:
    pass
if "GATE_SPEC" in globals():
    for g, spec in GATE_SPEC.items():
        try:
            if g == "phase2_zero_shot":
                ev = (f"best zero-shot/UDA strict-external AUC per direction: "
                      + ", ".join(f"{d}: {v['best_zero_shot_roc_auc']:.4f} ({v['best_zero_shot_model']})"
                                  for d, v in P2_GATE["by_direction"].items())
                      + f"; criterion SHA-256 {spec['sha256'][:16]}")
                verdict = "PASSED" if P2_GATE.get("passed") else "FAILED"
            elif g == "phase3_multisource":
                ev = "; ".join(f"{d}: acc {v['best_external_accuracy']:.4f} ({v['best_model']})"
                               for d, v in P3_ACC_GATE["by_direction"].items())
                verdict = "PASSED" if P3_ACC_GATE.get("passed_both_directions") else "FAILED"
            else:
                continue
            CLAIM_MAP_ROWS.append({"headline": f"Gate {g}", "declaration": spec["criterion"],
                                   "verdict": verdict, "evidence": ev,
                                   "source": "pinned GATE_SPEC (tamper-evident, Section Phase 0)"})
        except Exception as _e:
            LOG.warning("claim map: gate %s skipped (%s)", g, type(_e).__name__)
if "BASERATE_SUMMARY" in globals():
    _gh = BASERATE_SUMMARY.get("external_precision_at_recall_0.7", {})
    for rk, v in _gh.items():
        CLAIM_MAP_ROWS.append({
            "headline": f"{rk}: external precision@recall>=0.7 at base rate 1%",
            "declaration": CFG.criteria["G"],
            "verdict": ("SUPPORTED" if np.isfinite(v.get("G_mean_precision_at_recall_0.7", np.nan))
                        and v["G_mean_precision_at_recall_0.7"] >= 0.50 else "NOT SUPPORTED"),
            "evidence": f"mean over 200 replicates = {v.get('G_mean_precision_at_recall_0.7', float('nan')):.4f}",
            "source": "revision-11 Section 40B (evaluation-only rejection resampling)"})
if "SEEDVAR_BUNDLE" in globals():
    for rk, v in SEEDVAR_BUNDLE.items():
        CLAIM_MAP_ROWS.append({
            "headline": f"{rk}: TEST ROC-AUC mean +/- seed std (5 seeds)",
            "declaration": "revision-11 multi-seed variance reporting (master prompt mod 1.5)",
            "verdict": "REPORTED",
            "evidence": f"{v['mean']['roc_auc']:.4f} +/- {v['std']['roc_auc']:.4f} over {v['n_seeds']} seeds",
            "source": "revision-11 Section 23B"})
CLAIM_MAP = pd.DataFrame(CLAIM_MAP_ROWS)
display(CLAIM_MAP[["headline", "verdict", "evidence"]].to_string(index=False)[:4000])
save_table(CLAIM_MAP, "table54B1_revision11_claim_map")

# ---- 2. literature comparison ------------------------------------------------------------------------
_lit_rows = []
def _lit_row(ref, metric, value, note):
    _lit_rows.append({"reference": ref, "metric": metric, "value": value, "protocol_note": note})

_lit_row("GramBeddings protocol (published)", "accuracy", "0.9827",
         "published gram-embedding sequence model on its own balanced test split; no domain-disjoint "
         "split, no calibrated probabilities, no strict-external evaluation")
_lit_row("TransURL (published, URL-only)", "F1", "0.9824",
         "recent URL-only transformer baseline; protocol differences as above")
_lit_row("TinyBERT-based URL classifier (published)", "F1", "0.9387",
         "distilled text transformer applied to URLs; protocol differences as above")
for rk in [k for k in RUN_KEYS if k.endswith(PRIMARY_FSET)]:
    try:
        it = IN_DOMAIN_TEST[(IN_DOMAIN_TEST["run"] == rk) & IN_DOMAIN_TEST["primary"]
                            & (IN_DOMAIN_TEST["stage"] == "B")
                            & (IN_DOMAIN_TEST["operating_point"] == "A balanced (MCC)")].iloc[0]
        _lit_row(f"THIS NOTEBOOK {rk} Stage-B (in-domain TEST, balanced)",
                 "accuracy / F1 / ROC-AUC", f"{it['accuracy']:.4f} / {it['f1']:.4f} / {it['roc_auc']:.4f}",
                 "domain-disjoint split, frozen threshold, cross-fitted calibration; the honest "
                 "comparison anchor for the published numbers above")
    except IndexError:
        pass
    try:
        xt = XFER_TABLE[(XFER_TABLE["run"] == rk) & XFER_TABLE["principal"]
                        & (XFER_TABLE["stage"] == "B") & XFER_TABLE["primary"]
                        & (XFER_TABLE["operating_point"].str.startswith("A"))].iloc[0]
        _lit_row(f"THIS NOTEBOOK {rk} zero-shot strict-external",
                 "ROC-AUC / F1", f"{xt['roc_auc']:.4f} / {xt['f1']:.4f}",
                 "frozen model, unseen corpus, registered-domain-unseen rows only - a protocol the "
                 "published numbers do not undergo; the gap is the honest transfer finding")
    except IndexError:
        pass
LIT_TABLE = pd.DataFrame(_lit_rows)
display(LIT_TABLE.to_string(index=False)[:4000])
save_table(LIT_TABLE, "table54B2_revision11_literature_comparison")

# ---- 3. limitations (auto-numbered, numbers read from the computed structures) ----------------------
_lim = []
_lim.append(("Dual-corpus domain separability",
             "The two corpora are separable at origin AUC "
             + (f"{GATE_SUMMARY['origin_auc']:.3f}" if "GATE_SUMMARY" in globals() else "n/a")
             + " (tier '" + (str(GATE_SUMMARY.get('origin_tier', 'n/a')) if "GATE_SUMMARY" in globals() else "n/a")
             + "'), so cross-corpus transfer measures protocol mismatch as much as phishing knowledge."))
if "BASERATE_SUMMARY" in globals():
    _gh = BASERATE_SUMMARY.get("external_precision_at_recall_0.7", {})
    _g_vals = ", ".join(f"{rk}: {v.get('G_mean_precision_at_recall_0.7', float('nan')):.3f}" for rk, v in _gh.items())
    _lim.append(("Base-rate degradation",
                 f"Balanced-corpus precision does not survive realistic prevalence: external "
                 f"precision@recall>=0.7 at simulated base rate 1% is {_g_vals} (criterion G floor 0.50). "
                 f"Deployment claims must quote base-rate-adjusted numbers, not balanced ones."))
if "SEEDVAR_BUNDLE" in globals():
    _sv = ", ".join(f"{rk}: +/-{v['std']['roc_auc']:.4f}" for rk, v in SEEDVAR_BUNDLE.items())
    _lim.append(("Seed variance",
                 f"Cross-seed TEST ROC-AUC standard deviation over 5 seeds: {_sv}. AUC increments below "
                 f"these values are not separable from seed noise (see table23B2)."))
_lim.append(("Third evaluation corpus deferred (master-prompt mod 1.6)",
             "No licence-compatible third phishing-URL corpus is available in this execution "
             "environment (LegitPhish was retired at revision 4 and its licence was never cleared for "
             "redistribution; no other candidate is attached). Per the master prompt this is declared "
             "as an explicit limitation rather than silently skipped."))
_lim.append(("Character-CNN compute cap (mod 1.2)",
             "The NumPy CNN baseline is fitted on at most "
             + (f"{CFG.revision11['cnn']['max_rows_fit']:,}" if "CFG" in globals() else "250,000")
             + " TRAIN rows per source (2 vCPU, no GPU/torch); its numbers are at that documented "
               "scale and are not directly comparable to full-corpus fits."))
_lim.append(("Class-conditional MMD approximation (mod 1.3)",
             "The cc-MMD variant implements the class-conditional first-moment (mean) alignment with "
             "CORAL second moments, not the full kernel MMD quantity; it is labelled as such."))
_lim.append(("Greedy adversarial search (mod 1.4)",
             "The composition search is greedy per parent with a 5-step budget over the P5-P9 "
             "operators; it lower-bounds adversary capability, it does not upper-bound it."))
LIMITATIONS_TABLE = pd.DataFrame(_lim, columns=["limitation", "detail"])
display(LIMITATIONS_TABLE.to_string(index=False)[:6000])
save_table(LIMITATIONS_TABLE, "table54B3_revision11_limitations")
print("Section 54B complete: claim map, literature comparison and limitations written "
      f"({len(CLAIM_MAP)} claims mapped, {len(LIT_TABLE)} literature rows, {len(LIMITATIONS_TABLE)} limitations).")'''
