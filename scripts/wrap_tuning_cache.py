#!/usr/bin/env python3
"""Infrastructure-only edit: wrap Stage-A tune_and_train results in the notebook's own r7_cache
so chunked execution can resume past the 1-2 hour tuning stage. NO modelling logic changes:
the original tune_and_train function is called unchanged; its results are simply persisted and
reloaded on a re-run (same pattern the notebook itself uses for every revision-7 phase stage,
and the same pattern revision 7 used when it introduced checkpointing: "infrastructure only;
no modelling logic, gate criterion or statistic is affected")."""
import json

NB = "/home/z/my-project/trac-phish-revision11.ipynb"
nb = json.load(open(NB))
cells = nb["cells"]

# anchor: the execution loop at the end of the tuning cell
ANCHOR = '''for rk in RUN_KEYS:
    tune_and_train(rk)
'''
NEW = '''# ---- REVISION 11 (infrastructure-only, same pattern as the revision-7 checkpointing): persist
# ---- the Stage-A tuning result with r7_cache so a resumed run skips the 1-2h search. The original
# ---- tune_and_train above is called UNCHANGED; only its result is cached/reloaded.
def _tune_and_train_cached(rk):
    def _compute():
        tune_and_train(rk)
        _run = RUNS[rk]
        return {"models": _run["models"], "best_params": _run["best_params"],
                "tuning_table": _run["tuning_table"]}
    _out = r7_cache(f"tune_train_{rk.replace('|', '_')}", _compute)
    RUNS[rk].update(_out)

for rk in RUN_KEYS:
    _tune_and_train_cached(rk)
'''

hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and ANCHOR in "".join(c["source"])]
assert len(hits) == 1, f"anchor matched {len(hits)} cells"
i = hits[0]
s = "".join(cells[i]["source"])
assert s.count(ANCHOR) == 1
s = s.replace(ANCHOR, NEW)
cells[i]["source"] = s.splitlines(keepends=True)
compile(s, "<tuning-cell>", "exec")
json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print(f"cell {i}: Stage-A tuning wrapped in r7_cache (infrastructure-only), compiled OK")
