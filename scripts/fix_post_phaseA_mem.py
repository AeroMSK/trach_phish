# -*- coding: utf-8 -*-
"""rev-11 infra fix (OOM): post-Phase-A release before the compat benchmark.

Chunked execution on the 4 GB host OOM-killed the kernel inside the Section-16 compat
benchmark (cell index 77) at ~3.5 GB RSS: the imputer-refit loop at the tail of the
Phase-A cell leaves the last full-corpus matrix in the _FEAT_NP LRU (~175 MB) plus
allocator-held fit transients, so the benchmark starts at ~3.25 GB with too little
headroom for its model-fitting transients.

Fix: append a release block to the Phase-A cell tail (clear LRU + scratch guard +
gc + malloc_trim). Everything released is recomputable on demand (raw_matrix
re-materialises from FEATS on the next full-partition call); no downstream value is
affected. Value-identical, infrastructure-only.
"""
import json
import sys

NB = "/home/z/my-project/trac-phish-revision11.ipynb"

PATCH = '''

# ---- rev-11 infra fix (OOM): post-Phase-A release before the compat benchmark ---------------------
# On the 4 GB execution host the Section-16 compat benchmark that follows was OOM-killed
# (kernel RSS ~3.5 GB): the imputer-refit loop above leaves the last full-corpus matrix in
# the _FEAT_NP LRU (~175 MB) plus allocator-held fit transients, so the benchmark started
# at ~3.25 GB with insufficient headroom for its model-fitting transients. Everything
# released here is recomputable on demand (raw_matrix re-materialises from FEATS on the
# next full-partition call) and holds no value consumed downstream - value-identical.
_FEAT_NP.clear()
_FEAT_NP_ORDER.clear()
_r11_mem_guard("post-Phase-A", report=False)
gc.collect()
try:
    import ctypes
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass
_krss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024
print(f"post-Phase-A release applied; kernel RSS {_krss} MB")
'''

nb = json.load(open(NB))
cell = nb["cells"][75]
src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

assert "PHASE A (REVISION 8)" in src, "cell 75 is not the Phase-A cell"
assert "RUN_KEYS:" in src, "cell 75 tail (RUN_KEYS print) not found"
assert "post-Phase-A release applied" not in src, "patch already applied"

cell["source"] = src + PATCH
nbformat_src = cell["source"]

# validate the patched cell compiles
import ast
ast.parse(nbformat_src)
json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print("patched cell 75 (post-Phase-A release); compiles OK")
