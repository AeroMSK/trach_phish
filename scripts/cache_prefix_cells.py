#!/usr/bin/env python3
"""Cache the two expensive uncached prefix cells (corrected splits):
- cell 66: origin diagnostic -> r7_cache("r11_origin_diagnostic"); body = loop + all 5 structures,
  tail = prints + RESULTS assignment (re-runs every restart).
- cell 77: compat benchmark -> r7_cache("r11_compat_benchmark"); body = the full loop, tail =
  frame/CSV/display/print (re-runs every restart from the cached rows).
"""
import json

NB = "/home/z/my-project/trac-phish-revision11.ipynb"
nb = json.load(open(NB))
cells = nb["cells"]

# ---------- cell 66 ----------
i66 = [i for i, c in enumerate(cells) if c["cell_type"] == "code"
       and "ORIGIN_ROWS, ORIGIN_IMP = [], {}" in "".join(c["source"])][0]
s = "".join(cells[i66]["source"])
head = "ORIGIN_ROWS, ORIGIN_IMP = [], {}\n"
assert s.startswith(head)
tail_anchor = 'print(f"\\nOrigin separability on'
ti = s.find(tail_anchor)
assert ti > 0
body, tail = s[len(head):ti], s[ti:]
# the body must assign all returned names
for name in ["ORIGIN_TABLE", "ORIGIN_GROUPS", "ORIGIN_TIER"]:
    assert f"\n{name} = " in "\n" + body or body.startswith(f"{name} = "), f"{name} not in body"
assert "_oauc" in body
new = ("# rev-11 infra (chunk-resilient): the origin diagnostic is checkpointed (r7_cache); the\n"
       "# reporting tail below re-runs on every restart from the cached structures.\n"
       "def _r11_origin_diagnostic():\n"
       "    ORIGIN_ROWS, ORIGIN_IMP = [], {}\n"
       + "\n".join("    " + ln if ln.strip() else ln for ln in body.split("\n"))
       + "\n    return ORIGIN_ROWS, ORIGIN_IMP, ORIGIN_TABLE, ORIGIN_GROUPS, ORIGIN_TIER, _oauc\n\n"
       "_origin = r7_cache(\"r11_origin_diagnostic\", _r11_origin_diagnostic)\n"
       "ORIGIN_ROWS, ORIGIN_IMP, ORIGIN_TABLE, ORIGIN_GROUPS, ORIGIN_TIER, _oauc = _origin\n"
       "del _origin\n"
       + tail)
compile(new, "<cell66>", "exec")
cells[i66]["source"] = new.splitlines(keepends=True)
print(f"cell {i66}: origin diagnostic cached (body {len(body)} ch)")

# ---------- cell 77 ----------
i77 = [i for i, c in enumerate(cells) if c["cell_type"] == "code"
       and "def _bench_sample(ds: str, part: str, cap: int, key: str)" in "".join(c["source"])][0]
s = "".join(cells[i77]["source"])
bpos = s.find("BENCH_ROWS = []\n")
assert bpos > 0
head = s[:bpos]
tpos = s.find("SMALL_BENCHMARK = pd.DataFrame(BENCH_ROWS)")
assert tpos > bpos
body, tail = s[bpos + len("BENCH_ROWS = []\n"):tpos], s[tpos:]
new = (head
       + "t0 = time.time()\n"
       + "# rev-11 infra (chunk-resilient): the whole benchmark loop is checkpointed (r7_cache); the\n"
       + "# frame, CSV write and display below re-run from the cached rows on every restart.\n"
       + "def _r11_compat_benchmark():\n"
       + "    BENCH_ROWS = []\n"
       + "\n".join("    " + ln if ln.strip() else ln for ln in body.split("\n"))
       + "\n    return BENCH_ROWS\n\n"
       + "BENCH_ROWS = r7_cache(\"r11_compat_benchmark\", _r11_compat_benchmark)\n"
       + tail)
compile(new, "<cell77>", "exec")
cells[i77]["source"] = new.splitlines(keepends=True)
print(f"cell {i77}: compat benchmark cached (body {len(body)} ch)")

json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print("saved")
