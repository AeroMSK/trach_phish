#!/usr/bin/env python3
"""LRU-bound the _FEAT_NP feature-matrix cache (the accumulating-copies OOM) and extend the
hygiene cell to also release url_canonical (last used in the splitting cell).

raw_matrix() cached one full float32 matrix per (dataset, fset) - 19 combos x 100-215 MB
= up to ~5 GB on a 4 GB host. The access pattern is sequential per (dataset, fset) stage,
so an LRU of 2 covers it; an evicted combo costs only a ~0.5 s re-extraction. Values are
bit-identical (same to_numpy(dtype=float32) source, same [idx] rows).
"""
import json

NB = "/home/z/my-project/trac-phish-revision11.ipynb"
nb = json.load(open(NB))
cells = nb["cells"]

# ---- 1. LRU-bounded raw_matrix (cell 60) ----
i60 = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "_FEAT_NP: Dict" in "".join(c["source"])]
assert len(i60) == 1
i60 = i60[0]
s = "".join(cells[i60]["source"])
old = '''_FEAT_NP: Dict[Tuple[str, str], np.ndarray] = {}


def raw_matrix(ds: str, idx: np.ndarray, fset: str) -> np.ndarray:
    """Raw (un-imputed) float32 feature rows for a feature setting (column order = schema order)."""
    if (ds, fset) not in _FEAT_NP:
        _FEAT_NP[(ds, fset)] = FEATS[ds][FEATURE_SETS[fset]].to_numpy(dtype=np.float32)
    return _FEAT_NP[(ds, fset)][idx]'''
new = '''_FEAT_NP: Dict[Tuple[str, str], np.ndarray] = {}
_FEAT_NP_ORDER: List[Tuple[str, str]] = []
_FEAT_NP_MAX = 2      # rev-11 infra fix (OOM): LRU bound. Unbounded, this cache would hold up to
                      # 19 matrices x 100-215 MB (~5 GB) on a host with 4 GB RAM. The access pattern
                      # is sequential per (dataset, fset) stage, so 2 entries cover it; an eviction
                      # costs only a ~0.5 s re-extraction. Values are bit-identical.


def raw_matrix(ds: str, idx: np.ndarray, fset: str) -> np.ndarray:
    """Raw (un-imputed) float32 feature rows for a feature setting (column order = schema order)."""
    key = (ds, fset)
    if key not in _FEAT_NP:
        while len(_FEAT_NP_ORDER) >= _FEAT_NP_MAX:
            _old = _FEAT_NP_ORDER.pop(0)
            _FEAT_NP.pop(_old, None)
        _FEAT_NP[key] = FEATS[ds][FEATURE_SETS[fset]].to_numpy(dtype=np.float32)
        _FEAT_NP_ORDER.append(key)
    else:
        _FEAT_NP_ORDER.remove(key)
        _FEAT_NP_ORDER.append(key)
    return _FEAT_NP[key][idx]'''
assert old in s, "raw_matrix anchor missing"
s = s.replace(old, new)
compile(s, "<cell60>", "exec")
cells[i60]["source"] = s.splitlines(keepends=True)
print(f"cell {i60}: raw_matrix LRU-bounded, compiled OK")

# ---- 2. hygiene cell: also drop url_canonical ----
ih = [i for i, c in enumerate(cells) if c.get("id") == "r11-memhygiene"]
assert len(ih) == 1
ih = ih[0]
h = "".join(cells[ih]["source"])
old_h = '''for _ds in list(CLEAN):
    if "url_canonical_noscheme" in CLEAN[_ds].columns:
        CLEAN[_ds] = CLEAN[_ds].drop(columns=["url_canonical_noscheme"])'''
new_h = '''for _ds in list(CLEAN):
    _drop = [c for c in ("url_canonical_noscheme", "url_canonical") if c in CLEAN[_ds].columns]
    if _drop:
        CLEAN[_ds] = CLEAN[_ds].drop(columns=_drop)'''
assert old_h in h, "hygiene anchor missing"
h = h.replace(old_h, new_h)
h = h.replace("#   * url_canonical_noscheme - last read in Section 10B (external views, the cell above)",
              "#   * url_canonical_noscheme - last read in Section 10B (external views, the cell above)\n"
              "#   * url_canonical          - last read in Section 10 (the domain-disjoint split)")
compile(h, "<hygiene>", "exec")
cells[ih]["source"] = h.splitlines(keepends=True)
print(f"cell {ih}: hygiene extended with url_canonical release, compiled OK")

json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print("saved")
