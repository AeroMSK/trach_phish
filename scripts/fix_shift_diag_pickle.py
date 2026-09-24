# -*- coding: utf-8 -*-
"""rev-11 infra fix: drop unpicklable closures from the shift-diagnostics checkpoint dict.

The generic locals-capture in cell 70 includes nested helper functions (closures) which
joblib cannot pickle -> PicklingError on every chunk -> the checkpoint never lands ->
58-59 s of recomputation on every chunk restart. A name-usage audit shows the only cell-70
names any later cell references are P1_GATE, _rows_g, _rows_p (all picklable), so
excluding callables from the persisted dict is value-identical for every downstream
consumer. Infrastructure-only.
"""
import json
import ast

NB = "/home/z/my-project/trac-phish-revision11.ipynb"

OLD = """    _loc = dict(locals())          # function locals (NOT the comprehension scope)
    return {k: v for k, v in _loc.items() if not k.startswith("__")}"""

NEW = """    _loc = dict(locals())          # function locals (NOT the comprehension scope)
    # rev-11 infra fix: nested helper closures (e.g. _dinv_shift) cannot be pickled by
    # joblib, which made this checkpoint fail with PicklingError on every chunk and cost
    # ~59 s of recomputation per restart. A name-usage audit of every later cell shows
    # only P1_GATE / _rows_g / _rows_p (all picklable) are referenced downstream, so
    # excluding callables from the persisted dict is value-identical for consumers.
    return {k: v for k, v in _loc.items() if not k.startswith("__") and not callable(v)}"""

nb = json.load(open(NB))
cell = nb["cells"][70]
src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
assert OLD in src, "expected locals-capture pattern not found in cell 70"
cell["source"] = src.replace(OLD, NEW)
ast.parse(cell["source"])
json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print("patched cell 70 (callable filter in checkpoint dict); compiles OK")
