#!/usr/bin/env python3
"""Memory pass 3:
1. CLEAN's low-cardinality columns (partition, inner, original_split) -> pd.Categorical
   (~150 MB: 1.47M x 3 columns x 57-byte Python str -> 8-byte codes). Semantics of the
   usages in the notebook (== comparisons, isin, boolean masks, value_counts, groupby,
   .loc indexing) are preserved by Categorical.
2. r7_cache: trim hook on the LOAD path too.
3. Hygiene cell: apply the categorical conversion there (before any later consumer).
"""
import json

NB = "/home/z/my-project/trac-phish-revision11.ipynb"
nb = json.load(open(NB))
cells = nb["cells"]


def cell_with(anchor, what):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and anchor in "".join(c["source"])]
    assert len(hits) == 1, f"{what}: {len(hits)} hits"
    return hits[0]


# ---- 1+3. hygiene cell: categorical conversion ----
ih = cell_with("revision-11 memory hygiene applied", "hygiene")
h = "".join(cells[ih]["source"])
old_h = '''for _n in ("dev_domains", "dev_canon", "dev_raw"):
    if _n in globals():
        del globals()[_n]'''
new_h = '''for _n in ("dev_domains", "dev_canon", "dev_raw"):
    if _n in globals():
        del globals()[_n]
# rev-11 infra (memory): low-cardinality tag columns as Categorical codes (8 B/row instead of
# a Python string per row). Every later usage (== / isin / boolean masks / value_counts /
# groupby / .loc) is semantics-preserving on Categorical.
for _ds in list(CLEAN):
    for _col in ("partition", "inner", "original_split"):
        if _col in CLEAN[_ds].columns and not isinstance(CLEAN[_ds][_col].dtype, pd.CategoricalDtype):
            CLEAN[_ds][_col] = CLEAN[_ds][_col].astype("category")'''
assert old_h in h, "hygiene anchor"
h = h.replace(old_h, new_h)
compile(h, "<hygiene>", "exec")
cells[ih]["source"] = h.splitlines(keepends=True)
print(f"cell {ih}: categorical conversion added to hygiene")

# ---- 2. r7_cache load-path trim ----
i9 = cell_with("def r7_cache(name: str, compute, extra_key: str = \"\"):", "r7_cache")
s9 = "".join(cells[i9]["source"])
old_l = '''            obj = joblib.load(path)
            R7_CKPT_LOG.append({"stage": name, "status": "loaded from checkpoint", "seconds": round(time.time() - t0, 1),
                                "path": str(path)})
            LOG.info("checkpoint HIT  %s (%s)", name, path.name)
            return obj'''
new_l = '''            obj = joblib.load(path)
            R7_CKPT_LOG.append({"stage": name, "status": "loaded from checkpoint", "seconds": round(time.time() - t0, 1),
                                "path": str(path)})
            LOG.info("checkpoint HIT  %s (%s)", name, path.name)
            try:                               # rev-11 infra: reclaim fragmentation after loads too
                gc.collect()
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            return obj'''
assert old_l in s9, "r7_cache load anchor"
s9 = s9.replace(old_l, new_l)
compile(s9, "<cell9>", "exec")
cells[i9]["source"] = s9.splitlines(keepends=True)
print(f"cell {i9}: r7_cache load-path trim installed")

json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print("saved")
