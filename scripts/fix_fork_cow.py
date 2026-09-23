#!/usr/bin/env python3
"""Infrastructure-only fix: fork-COW guard for the two multiprocessing fork sites.

Root cause of the repeated OOM kills (3.5 GB anon RSS in ~30 s): forked pool children run
CPython's cyclic GC, which traverses EVERY tracked object in the inherited heap and updates
gc headers -> every page of the parent's ~1.5 GB heap is copy-on-write duplicated in EACH
child -> 2 children x 1.5 GB + parent = OOM. gc.freeze() moves existing objects to the
permanent generation (excluded from traversal), so children never touch those pages; gc.unfreeze()
after the pool restores normal collection. Identical computations, identical results.
"""
import json

NB = "/home/z/my-project/trac-phish-revision11.ipynb"
nb = json.load(open(NB))
cells = nb["cells"]

# --- cell 51: the incremental extraction pool ---------------------------------------------------------
old1 = '''                try:
                    import multiprocessing as mp
                    with mp.get_context("fork").Pool(n_jobs) as pool:
                        for (ci, p), a in zip(todo, pool.imap(fn, [chunks[ci] for ci, _ in todo], 1)):
                            _consume(ci, p, a)
                except Exception as exc:'''
new1 = '''                try:
                    import multiprocessing as mp
                    # rev-11 infra fix (fork-COW guard): freeze the parent heap so the children's
                    # cyclic GC cannot copy-on-write duplicate the entire heap (the OOM root cause)
                    gc.collect()
                    gc.freeze()
                    try:
                        with mp.get_context("fork").Pool(n_jobs) as pool:
                            for (ci, p), a in zip(todo, pool.imap(fn, [chunks[ci] for ci, _ in todo], 1)):
                                _consume(ci, p, a)
                    finally:
                        gc.unfreeze()
                except Exception as exc:'''
hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and old1 in "".join(c["source"])]
assert len(hits) == 1, f"cell51: {len(hits)} hits"
i = hits[0]
s = "".join(cells[i]["source"]).replace(old1, new1)
cells[i]["source"] = s.splitlines(keepends=True)
compile(s, "<cell51>", "exec")
print(f"cell {i}: extraction pool wrapped with gc.freeze/unfreeze, compiled OK")

# --- cell 93: the RF TreeSHAP pool --------------------------------------------------------------------
old2 = '''            try:   # fork workers inherit the explainer and data; results are concatenated in order
                import multiprocessing as mp
                _RF_JOB = (ex, X)
                bounds = np.linspace(0, X.shape[0], CFG.n_jobs * 4 + 1).astype(int)
                with mp.get_context("fork").Pool(CFG.n_jobs) as pool:
                    parts = pool.map(_rf_shap_chunk, list(zip(bounds[:-1], bounds[1:])))
                sv = np.vstack(parts)'''
new2 = '''            try:   # fork workers inherit the explainer and data; results are concatenated in order
                import multiprocessing as mp
                _RF_JOB = (ex, X)
                bounds = np.linspace(0, X.shape[0], CFG.n_jobs * 4 + 1).astype(int)
                # rev-11 infra fix (fork-COW guard): same OOM root cause as the Section-13 pool
                gc.collect()
                gc.freeze()
                try:
                    with mp.get_context("fork").Pool(CFG.n_jobs) as pool:
                        parts = pool.map(_rf_shap_chunk, list(zip(bounds[:-1], bounds[1:])))
                    sv = np.vstack(parts)
                finally:
                    gc.unfreeze()'''
hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and old2 in "".join(c["source"])]
assert len(hits) == 1, f"cell93: {len(hits)} hits"
i = hits[0]
s = "".join(cells[i]["source"]).replace(old2, new2)
cells[i]["source"] = s.splitlines(keepends=True)
compile(s, "<cell93>", "exec")
print(f"cell {i}: RF TreeSHAP pool wrapped with gc.freeze/unfreeze, compiled OK")

json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print("saved")
