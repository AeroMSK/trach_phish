# Part 1: CFG cell-5 edits (criteria G/H + revision11 config block) for Revision 11 modifications.
# Purely ADDITIVE string insertions into the existing cell source.

# --- 1a. append criteria G and H right after criterion "F" in CFG.criteria -------------------------
CRITERIA_F_TAIL = (
    '"F": "Revision-5 representation/adaptation repair helps transfer: external ROC-AUC (best revision-5 model) "\n'
    '             "> external ROC-AUC (revision-4 source-only F54-R) with a DeLong 95% CI excluding 0",\n'
    '    })'
)
CRITERIA_F_NEW = (
    '"F": "Revision-5 representation/adaptation repair helps transfer: external ROC-AUC (best revision-5 model) "\n'
    '             "> external ROC-AUC (revision-4 source-only F54-R) with a DeLong 95% CI excluding 0",\n'
    '        # ---- Revision 11 (additive, pre-registered BEFORE the base-rate stage runs; nothing above is changed) ----\n'
    '        "G": "Base-rate robustness (revision 11): mean precision at recall >= 0.70 on the principal "\n'
    '             "strict-external population under a simulated phishing base rate of 1% (200 rejection-resampling "\n'
    '             "replicates, evaluation-only, no refit) is >= 0.50 for the primary Stage-B model in BOTH "\n'
    '             "transfer directions",\n'
    '        "H": "Base-rate floor (revision 11): the same mean precision at recall >= 0.70 remains >= 0.10 at a "\n'
    '             "simulated base rate of 0.1% in BOTH transfer directions",\n'
    '    })'
)

# --- 1b. add the revision-11 config block right after the char_model field -------------------------
CHAR_MODEL_TAIL = (
    '        "max_features": 300_000, "C_grid": [0.5, 2.0, 8.0], "max_rows_fit": 400_000,\n'
    '        "fusion_alpha_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})\n'
)
CHAR_MODEL_NEW = (
    '        "max_features": 300_000, "C_grid": [0.5, 2.0, 8.0], "max_rows_fit": 400_000,\n'
    '        "fusion_alpha_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})\n'
    '    # ---- Revision 11 (additive): every tunable of the seven revision-11 modifications lives here.\n'
    '    #      Nothing above this block is altered; existing gates, thresholds and hashes are untouched.\n'
    '    revision11: Dict[str, Any] = field(default_factory=lambda: {\n'
    '        "baserate": {"rates": [0.0005, 0.001, 0.005, 0.01, 0.05], "replicates": 200, "n_pos": 500,\n'
    '                     "max_neg_draw": 200_000, "recalls": [0.5, 0.7, 0.9]},\n'
    '        "cnn": {"seq_len": 140, "filters": 48, "widths": [3, 5, 7], "dense": 64, "dropout": 0.2,\n'
    '                "lr": 3e-3, "batch": 128, "epochs": 6, "patience": 2, "max_rows_fit": 250_000,\n'
    '                "max_rows_val": 50_000, "weight_decay": 1e-5},\n'
    '        "coral": {"eig_floor": 1e-10, "fsets": ["F68R", "F54R"], "fit_kind": "xgb"},\n'
    '        "adv_search": {"n_parents": 120, "max_steps": 5, "improvement_eps": 1e-4},\n'
    '        "seed_variance": {"seeds": [42, 43, 44, 45, 46]},\n'
    '    })\n'
)
