#!/usr/bin/env python3
"""Final cell-51 memory fix on the clean committed state:
  A. column-wise verification (identical semantics, no ~0.8 GB transients)
  B. hygiene cell before extraction (del PHRESH_RAW, drop url_canonical_noscheme, drop overlap sets)
"""
import json

NB = "/home/z/my-project/trac-phish-revision11.ipynb"
nb = json.load(open(NB))
cells = nb["cells"]

i51 = [i for i, c in enumerate(cells) if c["cell_type"] == "code"
       and "def _extract_chunk(urls: List[str])" in "".join(c["source"])]
assert len(i51) == 1
i51 = i51[0]
s = "".join(cells[i51]["source"])

# ---------- A. column-wise verification ----------
old_asserts = '''_nan_ok = set(NAN_ALLOWED) | {ROBUST_PREFIX + c for c in NAN_ALLOWED_R} | {DINV_PREFIX + c for c in NAN_ALLOWED_D}
for ds in FEATS:
    arr = FEATS[ds].to_numpy()
    assert not np.isinf(arr).any()
    bad_nan = [c for c in ALL_FEATURE_COLUMNS if c not in _nan_ok and FEATS[ds][c].isna().any()]
    assert not bad_nan, f"unexpected NaNs in {bad_nan}"
    assert (np.nan_to_num(arr, nan=0) >= 0).all()
display(mem_report(gram_features=gram_features, phresh_features=phresh_features))'''
new_asserts = '''_nan_ok = set(NAN_ALLOWED) | {ROBUST_PREFIX + c for c in NAN_ALLOWED_R} | {DINV_PREFIX + c for c in NAN_ALLOWED_D}
for ds in FEATS:
    # rev-11 infra fix (OOM): column-wise verification - semantically identical to the original
    # whole-frame to_numpy/isinf/nan_to_num chain (no inf; no unexpected NaN; every non-NaN value
    # >= 0, with NaN excluded exactly as nan_to_num(nan=0) did) - but with zero-copy column views
    # instead of ~0.8 GB of transient full-frame copies on this memory-constrained host.
    for c in ALL_FEATURE_COLUMNS:
        v = FEATS[ds][c].to_numpy()
        assert not np.isinf(v).any(), f"infinite value in column {c}"
        if c not in _nan_ok:
            assert not FEATS[ds][c].isna().any(), f"unexpected NaNs in {c}"
        assert not (v < 0).any(), f"negative value in column {c}"
display(mem_report(gram_features=gram_features, phresh_features=phresh_features))'''
assert old_asserts in s, "assert block anchor missing"
s = s.replace(old_asserts, new_asserts)
compile(s, "<cell51>", "exec")
cells[i51]["source"] = s.splitlines(keepends=True)
print(f"cell {i51}: column-wise asserts installed, compiled OK")

# ---------- B. hygiene cell before cell 51 ----------
HYGIENE = '''# ===================================================================================================
# REVISION 11 (infrastructure-only) - memory hygiene before the feature-matrix stage.
# The execution host has 4 GB RAM; the kernel enters this stage at ~2.3 GB. Three structures
# are provably unused from here on and are released (usage was searched, not assumed):
#   * PHRESH_RAW             - last read in Section 2B (metadata consistency) and the schema preview
#   * url_canonical_noscheme - last read in Section 10B (external views, the cell above)
#   * the dev-overlap sets    - loop-local leftovers from Section 10B
# No value consumed by any later cell is affected; the determinism assert in the population
# builder re-verifies every matrix against the cached parquet anyway.
# ===================================================================================================
if "PHRESH_RAW" in globals():
    del PHRESH_RAW
for _ds in list(CLEAN):
    if "url_canonical_noscheme" in CLEAN[_ds].columns:
        CLEAN[_ds] = CLEAN[_ds].drop(columns=["url_canonical_noscheme"])
for _n in ("dev_domains", "dev_canon", "dev_raw"):
    if _n in globals():
        del globals()[_n]
gc.collect()
display(mem_report(**{f"CLEAN_{k}": v for k, v in CLEAN.items()}))
print("revision-11 memory hygiene applied: PHRESH_RAW, url_canonical_noscheme and the overlap "
      "sets released before feature extraction.")'''
compile(HYGIENE, "<hygiene>", "exec")
hyg_cell = {"cell_type": "code", "id": "r11-memhygiene", "metadata": {},
            "source": HYGIENE.splitlines(keepends=True), "execution_count": None, "outputs": []}
cells.insert(i51, hyg_cell)
print(f"hygiene cell inserted at index {i51}, compiled OK")

json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print(f"saved: {len(cells)} cells")
