#!/usr/bin/env python3
"""Comprehensive memory pass 2 (all value-identical / additive):

1. hygiene cell: also drop CLEAN columns provably unused after the data layer
   (domain_status, has_scheme, public_suffix, source_dataset) + malloc_trim after gc.
2. r7_cache: gc.collect + malloc_trim(0) after every cached stage (fragmentation reclaimer).
3. cell 72 (_domain_identifiability): float32 copies, del intermediates, in-place permutation
   in the importance loop (removes ~700 MB of transients).
4. cell 205 (final sanity): recompute url_canonical on the fly (the hygiene cell releases the
   column; canonicalize_url is deterministic so the check semantics are identical).
"""
import json

NB = "/home/z/my-project/trac-phish-revision11.ipynb"
nb = json.load(open(NB))
cells = nb["cells"]


def cell_with(anchor, what):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and anchor in "".join(c["source"])]
    assert len(hits) == 1, f"{what}: {len(hits)} hits"
    return hits[0]


# ---------- 1. hygiene cell: more drops + trim ----------
ih = cell_with("r11-memhygiene-marker-not-used" if False else "revision-11 memory hygiene applied", "hygiene")
h = "".join(cells[ih]["source"])
old_h = '''for _ds in list(CLEAN):
    _drop = [c for c in ("url_canonical_noscheme", "url_canonical") if c in CLEAN[_ds].columns]
    if _drop:
        CLEAN[_ds] = CLEAN[_ds].drop(columns=_drop)'''
new_h = '''for _ds in list(CLEAN):
    _drop = [c for c in ("url_canonical_noscheme", "url_canonical", "domain_status", "has_scheme",
                         "public_suffix", "source_dataset") if c in CLEAN[_ds].columns]
    if _drop:
        CLEAN[_ds] = CLEAN[_ds].drop(columns=_drop)'''
assert old_h in h, "hygiene drop anchor"
h = h.replace(old_h, new_h)
old_t = '''gc.collect()
display(mem_report(**{f"CLEAN_{k}": v for k, v in CLEAN.items()}))'''
new_t = '''gc.collect()
try:                                   # rev-11 infra: return freed arenas to the OS where possible
    import ctypes
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass
display(mem_report(**{f"CLEAN_{k}": v for k, v in CLEAN.items()}))'''
assert old_t in h, "hygiene trim anchor"
h = h.replace(old_t, new_t)
h = h.replace("#   * url_canonical          - last read in Section 10 (the domain-disjoint split)",
              "#   * url_canonical          - last read in Section 10 (the split); the final sanity check\n"
              "#                                recomputes it on the fly (deterministic canonicalize_url)\n"
              "#   * domain_status / has_scheme / public_suffix / source_dataset - last read in the\n"
              "#                                domain-extraction / split cells above")
compile(h, "<hygiene>", "exec")
cells[ih]["source"] = h.splitlines(keepends=True)
print(f"cell {ih}: hygiene extended (4 more columns + malloc_trim)")

# ---------- 2. r7_cache: trim hook ----------
i9 = cell_with("def r7_cache(name: str, compute, extra_key: str = \"\"):", "r7_cache")
s9 = "".join(cells[i9]["source"])
old_r = '''    obj = compute()
    try:
        joblib.dump(obj, path)
    except Exception as e:
        LOG.warning("could not persist checkpoint %s (%s)", name, type(e).__name__)
    R7_CKPT_LOG.append({"stage": name, "status": "computed", "seconds": round(time.time() - t0, 1), "path": str(path)})
    LOG.info("checkpoint MISS %s computed in %.0fs", name, time.time() - t0)
    return obj'''
new_r = '''    obj = compute()
    try:
        joblib.dump(obj, path)
    except Exception as e:
        LOG.warning("could not persist checkpoint %s (%s)", name, type(e).__name__)
    R7_CKPT_LOG.append({"stage": name, "status": "computed", "seconds": round(time.time() - t0, 1), "path": str(path)})
    LOG.info("checkpoint MISS %s computed in %.0fs", name, time.time() - t0)
    try:                                   # rev-11 infra: reclaim fragmentation between stages
        gc.collect()
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    return obj'''
assert old_r in s9, "r7_cache anchor"
s9 = s9.replace(old_r, new_r)
compile(s9, "<cell9>", "exec")
cells[i9]["source"] = s9.splitlines(keepends=True)
print(f"cell {i9}: r7_cache trim hook installed")

# ---------- 3. cell 72 _domain_identifiability: lean ----------
i72 = cell_with("def _domain_identifiability(cols: List[str], key: str)", "domclf")
s72 = "".join(cells[i72]["source"])
old_d = '''    Xg = FEATS["gram"][cols].to_numpy(dtype=np.float64)[_R7_ROWS_G]
    Xp = FEATS["phresh"][cols].to_numpy(dtype=np.float64)[_R7_ROWS_P]
    X = np.nan_to_num(np.vstack([Xg, Xp]), nan=0.0, posinf=0.0, neginf=0.0)
    d = np.concatenate([np.zeros(len(Xg)), np.ones(len(Xp))])
    mu, sd = X.mean(0), np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    Xs = (X - mu) / sd'''
new_d = '''    # rev-11 infra fix (OOM): float32 intermediates + explicit release (values identical; the
    # classifier upcasts its own working copies internally, which is where float64 belongs)
    Xg = FEATS["gram"][cols].to_numpy(dtype=np.float32)[_R7_ROWS_G]
    Xp = FEATS["phresh"][cols].to_numpy(dtype=np.float32)[_R7_ROWS_P]
    d = np.concatenate([np.zeros(len(Xg)), np.ones(len(Xp))])
    X = np.nan_to_num(np.vstack([Xg, Xp]), nan=0.0, posinf=0.0, neginf=0.0)
    del Xg, Xp
    mu, sd = X.mean(0), np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    Xs = ((X - mu) / sd).astype(np.float32)
    del X'''
assert old_d in s72, "domclf head anchor"
s72 = s72.replace(old_d, new_d)
old_i = '''    imps = []
    for j in range(Xs.shape[1]):
        Z = Xs.copy()
        Z[:, j] = rng.permutation(Z[:, j])
        imps.append(base - fast_auc(d, full.predict_proba(Z)[:, 1]))
    return {"auc": float(auc), "importance": np.asarray(imps), "cols": list(cols), "n_rows": int(len(d))}'''
new_i = '''    imps = []
    _orig_j = Xs[:, 0].copy() if Xs.shape[1] else None            # rev-11 infra: in-place permutation
    for j in range(Xs.shape[1]):                                  # (removes the per-feature full copy)
        _saved = Xs[:, j].copy()
        Xs[:, j] = rng.permutation(Xs[:, j])
        imps.append(base - fast_auc(d, full.predict_proba(Xs)[:, 1]))
        Xs[:, j] = _saved
    del _orig_j
    return {"auc": float(auc), "importance": np.asarray(imps), "cols": list(cols), "n_rows": int(len(d))}'''
assert old_i in s72, "domclf importance anchor"
s72 = s72.replace(old_i, new_i)
compile(s72, "<cell72>", "exec")
cells[i72]["source"] = s72.splitlines(keepends=True)
print(f"cell {i72}: _domain_identifiability leanified")

# ---------- 4. cell 205: on-the-fly canonical recomputation ----------
i205 = cell_with('for col, lvl in [("url_raw", "exact"), ("url_canonical", "canonical")]:', "sanity")
s205 = "".join(cells[i205]["source"])
old_s1 = '''and not (set(CLEAN[src]["url_canonical"].values[run["trainval_idx"]]) &
         set(CLEAN[src]["url_canonical"].values[partition_index(src, "test")]))'''
alt_s1 = '''and not (set(CLEAN[src]["url_canonical"].values[run["trainval_idx"]]) &
set(CLEAN[src]["url_canonical"].values[partition_index(src, "test")])))'''
new_s1 = '''and not (set(canonicalize_url(u) for u in CLEAN[src]["url_raw"].values[run["trainval_idx"]]) &
         set(canonicalize_url(u) for u in CLEAN[src]["url_raw"].values[partition_index(src, "test")]))'''
if old_s1 in s205:
    s205 = s205.replace(old_s1, new_s1)
elif alt_s1 in s205:
    s205 = s205.replace(alt_s1, new_s1.replace("         set(", "set(").replace('"test"])', '"test"])))'))
else:
    # locate the exact two-line form
    import re
    m = re.search(r'set\(CLEAN\[src\]\["url_canonical"\]\.values\[run\["trainval_idx"\]\]\)', s205)
    assert m, "sanity canonical anchor not found"
    s205 = s205.replace('set(CLEAN[src]["url_canonical"].values[run["trainval_idx"]])',
                        'set(canonicalize_url(u) for u in CLEAN[src]["url_raw"].values[run["trainval_idx"]])')
    s205 = s205.replace('set(CLEAN[src]["url_canonical"].values[partition_index(src, "test")])',
                        'set(canonicalize_url(u) for u in CLEAN[src]["url_raw"].values[partition_index(src, "test")])')
compile(s205, "<cell205>", "exec")
cells[i205]["source"] = s205.splitlines(keepends=True)
print(f"cell {i205}: final-sanity canonical recomputed on the fly")

json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print("saved")
