# -*- coding: utf-8 -*-
"""Part 4: A14 baselines, A15 n-gram models, A16 lexical models, A17 selection, A18 main eval."""

CELLS_4 = []

def _metrics_code():
    return r'''def clf_metrics(y_true, y_pred, score=None, prefix=""):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 roc_auc_score, average_precision_score, matthews_corrcoef,
                                 balanced_accuracy_score)
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "precision": precision_score(y_true, y_pred, zero_division=0),
         "recall": recall_score(y_true, y_pred, zero_division=0),
         "f1": f1_score(y_true, y_pred, zero_division=0),
         "mcc": matthews_corrcoef(y_true, y_pred),
         "balanced_acc": balanced_accuracy_score(y_true, y_pred)}
    if score is not None:
        m["roc_auc"] = roc_auc_score(y_true, score)
        m["pr_auc"] = average_precision_score(y_true, score)
    return {prefix + k: round(v, 6) for k, v in m.items()}'''

CELLS_4.append(("code", _metrics_code() + r'''

# ============ CELL: [A14] BASELINES (majority + lexical rule) ============
with stage_agent("A14", "Baseline models (majority class + lexical rule)"):
    majority = int(y_tr.mean() >= 0.5)
    maj_pred_va = np.full(len(y_va), majority, dtype=np.int8)
    rule_va = (FX["va"].n_susp_tokens >= 2).astype(np.int8).values
    rule_va = rule_va | (FX["va"].tld_suspicious.values.astype(bool) & (FX["va"].n_susp_tokens.values >= 1))

    val_rows = [{"model": "MajorityClass", **clf_metrics(y_va, maj_pred_va, None)},
                {"model": "LexicalRule",  **clf_metrics(y_va, rule_va, FX["va"].susp_token_ratio.values)}]
    VAL_RESULTS = val_rows
    print(pd.DataFrame(VAL_RESULTS).to_string(index=False))'''))

CELLS_4.append(("code", r'''# ============ CELL: [A15] N-GRAM MODEL TRAINING (LogReg-saga / SGD-hinge SVM / MultinomialNB) ============
# NOTE (infrastructure-forced solver choices, documented in the negative-results ledger):
# - sklearn LinearSVC caused repeated OOM kills (SIGKILL) on the 575k x 130k sparse matrix during
#   empirical probes -> replaced by SGDClassifier(loss='hinge'): same hinge-loss linear-SVM objective.
# - LogisticRegression(liblinear) forces a float64 copy of the CSR data (~1 GB transient) which OOMed
#   in the full-pipeline memory context -> solver='saga', which is float32-native (+112 MB only),
#   deterministic with random_state, and ~3x faster on this data (validated by probe: 74 s, converged).
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib, gc

with stage_agent("A15", "Char n-gram model training + validation-grid selection"):
    cached = ckpt_load("ngram_models", extra={"m": CFG["logreg"], "s": CFG["sgdsvm"]})
    if cached is None:
        MODELS = {}
        for C in CFG["logreg"]["C_grid"]:
            t0 = time.time()
            lr = LogisticRegression(C=C, solver=CFG["logreg"]["solver"],
                                    max_iter=CFG["logreg"]["max_iter"], tol=CFG["logreg"]["tol"],
                                    random_state=SEED)
            lr.fit(Xtr_tfidf, y_tr)
            MODELS[f"LogReg(C={C})"] = lr
            print(f"    LogReg saga C={C}: fitted in {time.time()-t0:.1f}s, n_iter={lr.n_iter_[0]} "
                  f"[RSS {_rss_mb():.0f} MB]", flush=True)
        for a in CFG["sgdsvm"]["alpha_grid"]:
            t0 = time.time()
            svm = SGDClassifier(loss="hinge", alpha=a, max_iter=CFG["sgdsvm"]["max_iter"],
                                tol=1e-3, random_state=SEED)
            svm.fit(Xtr_tfidf, y_tr)
            MODELS[f"LinearSVM-SGD(alpha={a})"] = svm
            print(f"    LinearSVM-SGD alpha={a}: fitted in {time.time()-t0:.1f}s, n_iter={svm.n_iter_}", flush=True)
        for a in CFG["mnb"]["alpha_grid"]:
            nb = MultinomialNB(alpha=a)
            nb.fit(Xtr_tfidf, y_tr)
            MODELS[f"MultinomialNB(alpha={a})"] = nb
        for name, m in MODELS.items():
            joblib.dump(m, OUT / f"models/{name.replace('=', '').replace('(', '_').replace(')', '')}.joblib")
        ckpt_save("ngram_models", MODELS, extra={"m": CFG["logreg"], "s": CFG["sgdsvm"]})
    else:
        MODELS = cached

    def score_of(m, X):
        return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)

    for name, m in MODELS.items():
        p = m.predict(Xva_tfidf); s = score_of(m, Xva_tfidf)
        VAL_RESULTS.append({"model": name, **clf_metrics(y_va, p, s)})
    gc.collect()

    # ---- post-training memory strategy: release the ~990 MB train matrix ----
    # Downstream stages only need small pre-extracted slices:
    #   X_BG   (1000 rows)  -> A22 SHAP background mean
    #   X_NULL (50k rows)   -> S1 label-shuffle null sanity test
    rs_bg = np.random.RandomState(SEED + 101)
    X_BG = Xtr_tfidf[rs_bg.choice(Xtr_tfidf.shape[0], CFG["shap"]["background"], replace=False)].copy()
    rs_n = np.random.RandomState(SEED + 7)
    NULL_IDX = rs_n.choice(Xtr_tfidf.shape[0], CFG["sanity"]["shuffle_n"], replace=False)
    X_NULL = Xtr_tfidf[NULL_IDX].copy()
    Xtr_tfidf = None; gc.collect()
    print(f"    [mem] released Xtr_tfidf — kept X_BG {X_BG.shape}, X_NULL {X_NULL.shape} "
          f"[RSS {_rss_mb():.0f} MB]", flush=True)

    val_df = pd.DataFrame(VAL_RESULTS).sort_values("f1", ascending=False)
    val_df.to_csv(OUT / "tables/validation_results.csv", index=False)
    print(val_df.to_string(index=False))'''))

CELLS_4.append(("code", r'''# ============ CELL: [A16] LEXICAL-FEATURE MODELS (HistGB + MLP) ============
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import joblib, gc

with stage_agent("A16", "Lexical models (HistGradientBoosting + MLP)"):
    Xva_tfidf = None; gc.collect()   # release val matrix (170 MB); A16 uses lexical features only (lazy-reloaded in A21)
    cached = ckpt_load("lexical_models", extra={"g": CFG["histgb"], "n": CFG["mlp"]})
    if cached is None:
        t0 = time.time()
        gb = HistGradientBoostingClassifier(max_iter=CFG["histgb"]["max_iter"],
                                            learning_rate=CFG["histgb"]["learning_rate"],
                                            max_leaf_nodes=CFG["histgb"]["max_leaf_nodes"],
                                            early_stopping=True, random_state=SEED)
        gb.fit(FX["tr"].values, y_tr)
        print(f"    HistGB fitted in {time.time()-t0:.1f}s", flush=True)
        t0 = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=CFG["mlp"]["hidden"], max_iter=CFG["mlp"]["max_iter"],
                            early_stopping=CFG["mlp"]["early_stopping"], random_state=SEED)
        mlp.fit(FX["tr"].values, y_tr)
        print(f"    MLP fitted in {time.time()-t0:.1f}s", flush=True)
        LEXMODELS = {"HistGB(lexical)": gb, "MLP(lexical)": mlp}
        joblib.dump(gb, OUT / "models/HistGB_lexical.joblib")
        joblib.dump(mlp, OUT / "models/MLP_lexical.joblib")
        ckpt_save("lexical_models", LEXMODELS, extra={"g": CFG["histgb"], "n": CFG["mlp"]})
    else:
        LEXMODELS = cached

    for name, m in LEXMODELS.items():
        p = m.predict(FX["va"].values); s = m.predict_proba(FX["va"].values)[:, 1]
        VAL_RESULTS.append({"model": name, **clf_metrics(y_va, p, s)})
    gc.collect()
    val_df = pd.DataFrame(VAL_RESULTS).sort_values("f1", ascending=False)
    val_df.to_csv(OUT / "tables/validation_results.csv", index=False)
    print(val_df.to_string(index=False))'''))

CELLS_4.append(("code", r'''# ============ CELL: [A17] MODEL SELECTION (decision made on VALIDATION only) ============
with stage_agent("A17", "Model selection on validation fold"):
    val_df = pd.DataFrame(VAL_RESULTS).sort_values(["f1", "roc_auc"], ascending=False)
    best_name = val_df.iloc[0]["model"]
    ALL_MODELS = {**MODELS, **LEXMODELS}
    BEST = ALL_MODELS[best_name]
    BEST_IS_TFIDF = best_name in MODELS

    sel = pd.DataFrame([{"selected_model": best_name, "selection_fold": "validation",
                         "val_f1": float(val_df.iloc[0]["f1"]), "val_roc_auc": float(val_df.iloc[0]["roc_auc"]),
                         "candidates": len(val_df)}])
    sel.to_csv(OUT / "tables/model_selection.csv", index=False)
    print(f"[A17] Selected model (by validation F1): {best_name}  |  candidates: {len(val_df)}")'''))

CELLS_4.append(("code", r'''# ============ CELL: [A18] MAIN TEST EVALUATION (GB test — touched once, after selection) ============
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve

with stage_agent("A18", "Main held-out test evaluation (GB test, 160k rows)"):
    if Xte_tfidf is None:   # checkpoint-lazy load (RAM discipline: only Xtr+Xva resident during training)
        Xte_tfidf = ckpt_load("tfidf_Xte", extra={"tfidf": CFG["tfidf"]})
    rows = []
    PRED_TE, SCORE_TE = {}, {}
    for name, m in ALL_MODELS.items():
        if name in MODELS:
            Xe = Xte_tfidf
            p = m.predict(Xe); s = m.predict_proba(Xe)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xe)
        else:
            p = m.predict(FX["te"].values); s = m.predict_proba(FX["te"].values)[:, 1]
        PRED_TE[name], SCORE_TE[name] = p.astype(np.int8), s.astype(np.float32)
        rows.append({"model": name, **clf_metrics(y_gb_te, p, s)})
    p_best, s_best = PRED_TE[best_name], SCORE_TE[best_name]
    _slug = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s)
    test_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    test_df.to_csv(OUT / "tables/main_results.csv", index=False)
    np.savez_compressed(OUT / "predictions/gb_test_predictions.npz",
                        y=y_gb_te, **{f"pred__{_slug(k)}": v for k, v in PRED_TE.items()},
                        **{f"score__{_slug(k)}": v for k, v in SCORE_TE.items()})

    cm = confusion_matrix(y_gb_te, p_best)
    cm_df = pd.DataFrame(cm, index=["true_legit", "true_phish"], columns=["pred_legit", "pred_phish"])
    cm_df.to_csv(OUT / "tables/confusion_matrix_best.csv")
    (OUT / "reports/classification_report_best.txt").write_text(
        f"Best model: {best_name}\n\n" + classification_report(y_gb_te, p_best, digits=4) + f"\nConfusion:\n{cm_df}\n")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    from sklearn.metrics import auc as sk_auc
    for nm in test_df.head(3)["model"]:
        fpr, tpr, _ = roc_curve(y_gb_te, SCORE_TE[nm]); ax[0].plot(fpr, tpr, lw=1.6,
            label=f"{nm} (AUC={sk_auc(fpr, tpr):.4f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC — GB test")
    ax[0].legend(fontsize=7)
    for nm in test_df.head(3)["model"]:
        pr, rc, _ = precision_recall_curve(y_gb_te, SCORE_TE[nm]); ax[1].plot(rc, pr, lw=1.6, label=nm)
    ax[1].set(xlabel="recall", ylabel="precision", title="PR — GB test")
    ax[1].legend(fontsize=7)
    fig.savefig(OUT / "figures/main_eval_curves.png", dpi=160); plt.close(fig)

    MAIN_METRICS = clf_metrics(y_gb_te, p_best, s_best)
    print(test_df.to_string(index=False))
    print(f"\n[A18] BEST = {best_name}: " + " ".join(f"{k}={v}" for k, v in MAIN_METRICS.items()))
    print(f"Confusion (best):\n{cm_df}")
    G3_PASS = (MAIN_METRICS["f1"] >= GATE_THRESHOLDS["G3_performance"]["min_test_f1"]) and \
              (MAIN_METRICS["roc_auc"] >= GATE_THRESHOLDS["G3_performance"]["min_test_roc_auc"])
    print(f"[A18] G3 performance gate: {'PASS' if G3_PASS else 'FAIL'}")'''))
