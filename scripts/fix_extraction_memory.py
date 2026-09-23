#!/usr/bin/env python3
"""Infrastructure-only memory/resume fix for the Section-13 feature extraction (cell 51).

The full-corpus extraction OOM-killed the kernel (3.5 GB anon RSS: float64 chunk lists +
whole-corpus vstack duplicates + fork pool). Fixes, all VALUE-IDENTICAL to the original:
  1. chunk arrays are float32 from the start (the frame was converted to float32 at the end
     anyway - same rounding, half the memory);
  2. the output matrix is PREALLOCATED and filled incrementally (no vstack duplicate);
  3. every chunk result is memoised on disk (r11_xchunk/*.npy), so a restarted chunk-execution
     resumes the extraction mid-way instead of losing all progress;
  4. pool results are consumed incrementally (imap) instead of accumulating via pool.map.

No feature definition, schema version, cache fingerprint or modelling logic is touched.
"""
import json

NB = "/home/z/my-project/trac-phish-revision11.ipynb"
nb = json.load(open(NB))
cells = nb["cells"]

# --- 1. float32 chunk arrays -------------------------------------------------------------------------
EXTRACTOR_OF = {"_extract_chunk": "extract_url_features",
                "_extract_chunk_robust": "extract_url_features_robust",
                "_extract_chunk_dinv": "extract_url_features_dinv"}
for fn_name, ex_name in EXTRACTOR_OF.items():
    old = f'''def {fn_name}(urls: List[str]) -> np.ndarray:
    return np.asarray([{ex_name}(u) for u in urls], dtype=np.float64)'''
    new = f'''def {fn_name}(urls: List[str]) -> np.ndarray:
    # revision-11 infrastructure fix: float32 from the start (the frame was converted to float32
    # at the end of _run anyway - identical values, half the intermediate memory)
    return np.asarray([{ex_name}(u) for u in urls], dtype=np.float32)'''
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and old in "".join(c["source"])]
    assert len(hits) == 1, f"{fn_name}: {len(hits)} hits"
    i = hits[0]
    s = "".join(cells[i]["source"])
    s = s.replace(old, new)
    cells[i]["source"] = s.splitlines(keepends=True)
    print(f"{fn_name} -> float32 (cell {i})")

# --- 2. preallocated incremental _run with per-chunk disk memo ---------------------------------------
old_run = '''    def _run(fn, width):
        arrays = None
        if n_jobs > 1 and len(urls) > 2 * chunk:
            try:
                import multiprocessing as mp
                with mp.get_context("fork").Pool(n_jobs) as pool:
                    arrays = pool.map(fn, chunks)
            except Exception as exc:
                LOG.warning("Parallel extraction failed (%s); using serial extraction.", exc)
                arrays = None
        if arrays is None:
            arrays = [fn(c) for c in chunks]
        X = np.vstack([a.reshape(-1, width) for a in arrays])
        X[~np.isfinite(X)] = np.nan
        return X.astype(np.float32)'''
new_run = '''    def _run(fn, width):
        # revision-11 infrastructure fix (OOM + resume): preallocated float32 output filled
        # incrementally; every chunk memoised on disk so a restarted run resumes mid-extraction.
        # Same values as the original vstack->float32 path.
        ck_dir = DIRS["cache"] / "r11_xchunk"
        ck_dir.mkdir(parents=True, exist_ok=True)
        ckey = hashlib.sha256(("|".join([schema, str(width), str(len(urls)), str(chunk),
                                          str(urls[0]) if urls else "", str(urls[-1]) if urls else ""])
                               ).encode()).hexdigest()[:16]
        X = np.empty((len(urls), width), dtype=np.float32)
        todo = []
        for ci in range(len(chunks)):
            p = ck_dir / f"{ckey}_{ci:04d}.npy"
            if p.exists():
                X[ci * chunk:(ci + 1) * chunk] = np.load(p).reshape(-1, width)
            else:
                todo.append((ci, p))

        def _consume(ci, p, a):
            a = np.asarray(a, dtype=np.float32)
            np.save(p, a)
            X[ci * chunk:(ci + 1) * chunk] = a.reshape(-1, width)

        if todo:
            if n_jobs > 1 and len(urls) > 2 * chunk:
                try:
                    import multiprocessing as mp
                    with mp.get_context("fork").Pool(n_jobs) as pool:
                        for (ci, p), a in zip(todo, pool.imap(fn, [chunks[ci] for ci, _ in todo], 1)):
                            _consume(ci, p, a)
                except Exception as exc:
                    LOG.warning("Parallel extraction failed (%s); using serial extraction.", exc)
                    for ci, p in todo:
                        if p.exists():                     # already consumed before the failure
                            X[ci * chunk:(ci + 1) * chunk] = np.load(p).reshape(-1, width)
                        else:
                            _consume(ci, p, fn(chunks[ci]))
            else:
                for ci, p in todo:
                    _consume(ci, p, fn(chunks[ci]))
        X[~np.isfinite(X)] = np.nan
        return X'''
i = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and old_run in "".join(c["source"])]
assert len(i) == 1, f"_run: {len(i)} hits"
i = i[0]
s = "".join(cells[i]["source"])
s = s.replace(old_run, new_run)
cells[i]["source"] = s.splitlines(keepends=True)
compile(s, "<cell51>", "exec")
print(f"_run -> incremental preallocated + per-chunk memo (cell {i}), compiled OK")
json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print("saved")
