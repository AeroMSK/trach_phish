# -*- coding: utf-8 -*-
"""Chunked executor for trac-phish-revision11.ipynb at FULL scale.

The sandbox kills background processes at tool-call end and caps foreground calls
at ~10 minutes, so the multi-hour run proceeds as sequential foreground chunks.
Each chunk re-executes from cell 0 in a FRESH kernel; the notebook's own
checkpointing (Section-13 feature disk cache + r7_cache for expensive stages)
skips work already persisted. The executed notebook is saved after EVERY cell.

Usage: python3 execute_r11_chunk.py [budget_seconds]   (default 540)
Exit code 0 = all cells done; 3 = budget exhausted (relaunch); 1 = real error.
"""
import os, sys, time, traceback

BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 540

os.environ["TRAC_RUN_MODE"] = "full"
os.environ.pop("TRAC_MAX_ROWS", None)
os.environ["TRAC_INPUT_ROOT"] = "/home/z/my-project/data_raw"
os.environ["TRAC_WORK_ROOT"] = "/home/z/my-project/r11_working"
os.environ["PYTHONUNBUFFERED"] = "1"

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

SRC = "/home/z/my-project/trac-phish-revision11.ipynb"
DST = "/home/z/my-project/trac-phish-revision11_FULL_EXECUTED.ipynb"
TIMING = "/home/z/my-project/logs/r11_cell_times.csv"

nb = nbformat.read(SRC, as_version=4)
t_start = time.time()


def _rss():
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS"):
                return int(l.split()[1]) // 1024
    except Exception:
        pass
    return -1


class ChunkedExecutor(ExecutePreprocessor):
    budget_out = False

    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type != "code":
            return cell, resources
        head = (cell.source.splitlines()[0][:66] if cell.source else "")[:66]
        if time.time() - t_start > BUDGET:
            self.budget_out = True
            print(f"[cell {index:>3}] SKIP (budget exhausted) {head}", flush=True)
            raise RuntimeError("__BUDGET__")
        print(f"[cell {index:>3}] RUN  {head} (RSS {_rss()} MB, t+{time.time()-t_start:.0f}s)", flush=True)
        t0 = time.time()
        cell, resources = super().preprocess_cell(cell, resources, index)
        dt = time.time() - t0
        try:
            nbformat.write(nb, DST)
        except Exception as e:
            print(f"[cell {index:>3}] SAVE-SKIP {type(e).__name__}: {e}", flush=True)
        with open(TIMING, "a") as f:
            f.write(f"{index},{dt:.1f},{_rss()},{int(time.time())}\n")
        print(f"[cell {index:>3}] DONE {head} ({dt:.1f}s)", flush=True)
        return cell, resources


ep = ChunkedExecutor(timeout=18000, kernel_name="python3",
                     interrupt_on_timeout=True, allow_errors=False)
err = None
budget_out = False
try:
    ep.preprocess(nb, {"metadata": {"path": "/home/z/my-project"}})
except RuntimeError as e:
    if "__BUDGET__" in str(e):
        budget_out = True
    else:
        err = e
        print("EXECUTION ERROR:\n", "".join(traceback.format_exception(e))[-6000:], flush=True)
except Exception as e:
    err = e
    print("EXECUTION ERROR:\n", "".join(traceback.format_exception(e))[-6000:], flush=True)
finally:
    try:
        nbformat.write(nb, DST)
    except Exception as e:
        print(f"FINAL SAVE FAILED: {e}", flush=True)

n_code = sum(1 for c in nb.cells if c.cell_type == "code")
n_ran = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count"))
n_err = sum(1 for c in nb.cells if c.cell_type == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error")
print(f"code cells: {n_code} | executed: {n_ran} | error outputs: {n_err} | "
      f"elapsed {(time.time()-t_start)/60:.1f} min", flush=True)
if err is None and n_err == 0 and n_ran == n_code:
    sys.exit(0)
if budget_out or (err is None and n_err == 0):
    sys.exit(3)
sys.exit(1)
