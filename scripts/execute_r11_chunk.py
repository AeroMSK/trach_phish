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


def _kernel_pid():
    """Find the Jupyter kernel child process pid (nbclient spawns it)."""
    try:
        import subprocess
        out = subprocess.run(["ps", "--ppid", str(os.getpid()), "-o", "pid="],
                             capture_output=True, text=True, timeout=5).stdout
        pids = [int(x) for x in out.split()]
        # deepest descendant (kernel may be a grandchild)
        for _ in range(3):
            new = []
            for p in pids:
                try:
                    o = subprocess.run(["ps", "--ppid", str(p), "-o", "pid="],
                                       capture_output=True, text=True, timeout=5).stdout
                    new += [int(x) for x in o.split()]
                except Exception:
                    pass
            if not new:
                break
            pids += new
        for p in pids:
            try:
                with open(f"/proc/{p}/cmdline") as f:
                    if "ipykernel" in f.read() or "kernel" in open(f"/proc/{p}/cmdline").read():
                        return p
            except Exception:
                pass
        return pids[-1] if pids else None
    except Exception:
        return None


def _rss(pid=None):
    pid = pid or "self"
    try:
        with open(f"/proc/{pid}/status") as f:
            for l in f:
                if l.startswith("VmRSS"):
                    return int(l.split()[1]) // 1024
    except Exception:
        pass
    return -1


import threading  # noqa: E402

_KPID = [None]


class KernelMemWatch(threading.Thread):
    """Sample the kernel's RSS every 250ms; log the peak per cell and live threshold crossings."""
    daemon = True

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.peak = 0
        self.last_reported = 0

    def run(self):
        import time as _t
        while not self.stop.is_set():
            if _KPID[0]:
                r = _rss(_KPID[0])
                if r > self.peak:
                    self.peak = r
                if r >= self.last_reported + 100 or (self.last_reported - r) >= 200:
                    print(f"    [mem] kernel {r} MB (t+{_t.time()-t_start:.0f}s)", flush=True)
                    self.last_reported = r
            _t.sleep(0.25)


watch = None


def _start_watch():
    global watch
    if watch is None:
        watch = KernelMemWatch()
        watch.start()


class ChunkedExecutor(ExecutePreprocessor):
    budget_out = False
    _r11_done = 0

    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type != "code":
            return cell, resources
        head = (cell.source.splitlines()[0][:66] if cell.source else "")[:66]
        if time.time() - t_start > BUDGET:
            self.budget_out = True
            print(f"[cell {index:>3}] SKIP (budget exhausted) {head}", flush=True)
            raise RuntimeError("__BUDGET__")
        if _KPID[0] is None or _rss(_KPID[0]) < 0:
            _KPID[0] = _kernel_pid()
        _start_watch()
        kpeak0 = watch.peak if watch else 0
        print(f"[cell {index:>3}] RUN  {head} (parent {_rss()} MB, kernel {_rss(_KPID[0])} MB, t+{time.time()-t_start:.0f}s)", flush=True)
        t0 = time.time()
        cell, resources = super().preprocess_cell(cell, resources, index)
        dt = time.time() - t0
        try:
            cell["execution_count"] = self._r11_done + 1
            self._r11_done += 1
            nbformat.write(nb, DST)
        except Exception as e:
            print(f"[cell {index:>3}] SAVE-SKIP {type(e).__name__}: {e}", flush=True)
        with open(TIMING, "a") as f:
            f.write(f"{index},{dt:.1f},{_rss(_KPID[0])},{int(time.time())}\n")
        kpeak = watch.peak if watch else 0
        print(f"[cell {index:>3}] DONE {head} ({dt:.1f}s, kernel {_rss(_KPID[0])} MB, cell-peak {kpeak} MB)", flush=True)
        if watch:
            watch.peak = _rss(_KPID[0])  # reset per-cell peak baseline
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
n_ran = ep._r11_done
n_err = sum(1 for c in nb.cells if c.cell_type == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error")
print(f"code cells: {n_code} | executed: {n_ran} | error outputs: {n_err} | "
      f"elapsed {(time.time()-t_start)/60:.1f} min", flush=True)
if err is None and n_err == 0 and n_ran == n_code:
    sys.exit(0)
if budget_out or (err is None and n_err == 0):
    sys.exit(3)
sys.exit(1)
