# -*- coding: utf-8 -*-
"""rev-11 infra (OOM + speed): checkpoint the Phase-A imputer-refit tail per run key.

The tail of the Phase-A cell refits a TrainOnlyImputer on the FULL train partition for
every run key (2 full-pipeline + 31 prediction-only keys). Each fit materialises the
full-corpus feature matrix via the _FEAT_NP LRU path (200-400 MB transient), and the
loop's rolling peak (observed 3439 MB) OOM-killed the kernel after the sandbox's
allocator behaviour changed with MALLOC_ARENA_MAX. It also wastes ~25 s of every chunk
prefix re-fitting identical values.

Fix: each imputer is checkpointed via the notebook's own r7_cache (deterministic medians
over the frozen TRAIN partition; keyed by the stable post-Phase-A fingerprint). On every
later chunk the tail is pure cache loads. Value-identical; infrastructure-only.
"""
import json
import ast

NB = "/home/z/my-project/trac-phish-revision11.ipynb"

OLD = """_FEAT_NP.clear()
for _rk in RUN_KEYS + PRED_RUN_KEYS:
    if _rk in RUNS:
        continue
    _s, _fs = _rk.split("|")
    _t = "phresh" if _s == "gram" else "gram"
    _tr = partition_index(_s, "train")
    _imp = TrainOnlyImputer().fit(raw_matrix(_s, _tr, _fs), CLEAN[_s]["record_id"].values[_tr])
    LEDGER.record("imputer", _rk, _s, "train", "fit_preprocessing", len(_tr), "revision-8 run")"""

NEW = """_FEAT_NP.clear()
for _rk in RUN_KEYS + PRED_RUN_KEYS:
    if _rk in RUNS:
        continue
    _s, _fs = _rk.split("|")
    _t = "phresh" if _s == "gram" else "gram"
    _tr = partition_index(_s, "train")
    # rev-11 infra (OOM + speed): each full-train imputer is checkpointed. The refit
    # materialises the full-corpus matrix for its feature set (200-400 MB transient per
    # key); re-doing that for 33 keys every chunk both wasted ~25 s of prefix and peaked
    # the kernel at 3.4 GB (OOM-kill after the allocator change). The imputer values are
    # deterministic medians over the frozen TRAIN partition - value-identical.
    def _fit_imp(_s=_s, _fs=_fs, _tr=_tr):
        return TrainOnlyImputer().fit(raw_matrix(_s, _tr, _fs), CLEAN[_s]["record_id"].values[_tr])
    _imp = r7_cache(f"r11_imputer_{_rk.replace('|', '_')}", _fit_imp)
    LEDGER.record("imputer", _rk, _s, "train", "fit_preprocessing", len(_tr), "revision-8 run")"""

nb = json.load(open(NB))
cell = nb["cells"][75]
src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
assert OLD in src, "imputer-refit loop pattern not found in cell 75"
assert "r11_imputer_" not in src, "patch already applied"
cell["source"] = src.replace(OLD, NEW)
ast.parse(cell["source"])
json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print("patched cell 75 (imputer checkpointing); compiles OK")
