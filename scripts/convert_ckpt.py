# -*- coding: utf-8 -*-
"""Convert the combined tfidf_matrices checkpoint into per-matrix checkpoints (resume optimization).
Replicates the notebook's exact fingerprint format so ckpt_load accepts them."""
import joblib, json, os

OUT = "/home/z/my-project/trac_phish_revision8_FULL_RUN"
SCHEMA_VERSION = "trac-phish-rev8-v2-canonical-dedup"
RUN_MODE = "full"
DATASET_SHA = {
    "grambeddings_archive": "4a5572edeb3d37d7563a2c98562408c837c144465ec7e0d45b838134a0e88f46",
    "phreshphish_archive": "c58b70f02eaa52fe1a7666d4e57f19a759480f53970e88d3e05a71dfd25fbbbb",
}
SEED = 20260923
TF = {"analyzer": "char", "ngram_range": [1, 5], "min_df_vocab": 8, "max_features": 300000,
      "sublinear_tf": True, "dtype": "float32", "vocab_subsample": 60000, "transform_chunk": 50000}

# verify the old fingerprint matches current expectations before converting
old = joblib.load(f"{OUT}/checkpoints/tfidf_matrices.joblib")
assert old["__fp__"] == {"schema": SCHEMA_VERSION, "run_mode": RUN_MODE, "dataset_sha": DATASET_SHA,
                         "seed": SEED, "extra": {"tfidf": TF}}, f"fp mismatch: {old['__fp__']}"
d = old["data"]
meta = {"vocab": d["Xtr"].shape[1],
        "shapes": {"Xtr": list(d["Xtr"].shape), "Xva": list(d["Xva"].shape),
                   "Xte": list(d["Xte"].shape), "Xpp": list(d["Xpp"].shape)},
        "nnz": {"Xtr": int(d["Xtr"].nnz), "Xva": int(d["Xva"].nnz),
                "Xte": int(d["Xte"].nnz), "Xpp": int(d["Xpp"].nnz)}}

def ck(nm, obj):
    joblib.dump({"__fp__": {"schema": SCHEMA_VERSION, "run_mode": RUN_MODE, "dataset_sha": DATASET_SHA,
                            "seed": SEED, "extra": {"tfidf": TF}}, "data": obj},
                f"{OUT}/checkpoints/tfidf_{nm}.joblib")
    print(f"saved tfidf_{nm}")

for nm in ["Xtr", "Xva", "Xte", "Xpp"]:
    ck(nm, d[nm])
ck("meta", meta)
os.remove(f"{OUT}/checkpoints/tfidf_matrices.joblib")
print("removed old combined checkpoint | meta:", json.dumps(meta))
