# -*- coding: utf-8 -*-
"""Execute trac-phish-revision8.ipynb at FULL scale.
Saves the executed notebook after EVERY cell (crash-resilient, full visibility)."""
import os, sys, time, traceback

os.environ["TRAC_RUN_MODE"] = "full"          # directive: force full mode
os.environ.pop("TRAC_MAX_ROWS", None)         # directive: ensure no row cap
os.environ["PYTHONUNBUFFERED"] = "1"          # stream kernel output for live monitoring

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

SRC = "/home/z/my-project/trac-phish-revision8.ipynb"
DST = "/home/z/my-project/trac-phish-revision8_FULL_EXECUTED.ipynb"

nb = nbformat.read(SRC, as_version=4)

def _rss():
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS"): return int(l.split()[1]) // 1024
    except Exception: pass
    return -1

class SavingExecutor(ExecutePreprocessor):
    def preprocess_cell(self, cell, resources, index):
        head = cell.source.splitlines()[0][:64] if cell.source else ""
        print(f"[cell {index:>2}] RUN  {head} (parent RSS {_rss()} MB)", flush=True)
        t0 = time.time()
        cell, resources = super().preprocess_cell(cell, resources, index)
        nbformat.write(nb, DST)                # persist outputs after every cell
        print(f"[cell {index:>2}] DONE {head} ({time.time()-t0:.1f}s)", flush=True)
        return cell, resources

t0 = time.time()
ep = SavingExecutor(timeout=14400, kernel_name="python3",
                    interrupt_on_timeout=True, allow_errors=False)
ok = True
try:
    ep.preprocess(nb, {"metadata": {"path": "/home/z/my-project"}})
except Exception as e:
    ok = False
    print("EXECUTION ERROR:\n", "".join(traceback.format_exception(e))[-5000:], flush=True)
finally:
    try:
        nbformat.write(nb, DST)
        print(f"executed notebook saved: {DST} (elapsed {time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"FINAL SAVE FAILED: {e}", flush=True)

n_code = sum(1 for c in nb.cells if c.cell_type == "code")
n_ran = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count"))
n_err = sum(1 for c in nb.cells if c.cell_type == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error")
print(f"code cells: {n_code} | executed: {n_ran} | error outputs: {n_err}", flush=True)
sys.exit(0 if (ok and n_err == 0 and n_ran == n_code) else 1)
