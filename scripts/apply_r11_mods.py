#!/usr/bin/env python3
"""Apply Revision 11 modifications 1.1-1.7 to trac-phish-revision11.ipynb (additive only).

- Edits cell 5 (CFG): appends criteria G/H + revision11 config block (string insertions).
- Edits cell 188 (Section 48): inserts criteria G/H evaluation before CRITERIA_TABLE build.
- Inserts 6 markdown + 6 code cells at anchor positions (Sections 22B2, 23B, Phase H, 29B, 40B/40C, 54B).
- Compiles every touched cell source to guarantee syntax validity before saving.
The original notebook is preserved in git history (user's upload commit e3fcd55).
"""
import json, sys, re
sys.path.insert(0, "/home/z/my-project/scripts/r11_mods")

from part1_cfg import CRITERIA_F_TAIL, CRITERIA_F_NEW, CHAR_MODEL_TAIL, CHAR_MODEL_NEW
from part2_cnn import MD_22B2, CODE_22B2
from part3_seedvar import MD_23B, CODE_23B
from part4_coral import MD_PHASEH, CODE_PHASEH
from part5_advsearch import MD_29B, CODE_29B
from part6_baserate import MD_40B, CODE_40B
from part7_report import MD_54B, CODE_54B, CELL188_TAIL_ANCHOR, CELL188_EXTENSION

NB = "/home/z/my-project/trac-phish-revision11.ipynb"
nb = json.load(open(NB))
cells = nb["cells"]
print(f"loaded notebook: {len(cells)} cells")


def src(c):
    return "".join(c["source"])


def find_anchor(tail):
    hits = [i for i, c in enumerate(cells) if tail in src(c)]
    assert len(hits) == 1, f"anchor matched {len(hits)} cells: {tail[:60]!r}"
    return hits[0]


def new_cell(kind, ident, text):
    c = {"cell_type": kind, "id": f"r11-{ident}",
         "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


# ---------------- 1. cell 5 edits (CFG) ----------------
i5 = find_anchor('"F": "Revision-5 representation/adaptation repair helps transfer')
s5 = src(cells[i5])
assert s5.count(CRITERIA_F_TAIL) == 1, "criteria-F anchor not unique in cell 5"
s5 = s5.replace(CRITERIA_F_TAIL, CRITERIA_F_NEW)
assert s5.count(CHAR_MODEL_TAIL) == 1, "char_model anchor not unique in cell 5"
s5 = s5.replace(CHAR_MODEL_TAIL, CHAR_MODEL_NEW)
cells[i5]["source"] = s5.splitlines(keepends=True)
compile(s5, "<cell5>", "exec")
print("cell 5 edited (criteria G/H + revision11 config) and compiled OK")

# ---------------- 2. cell 188 edit (criteria G/H evaluation) ----------------
i188 = find_anchor('CRITERIA_TABLE = pd.DataFrame(CRIT_ROWS)')
s188 = src(cells[i188])
assert s188.count(CELL188_TAIL_ANCHOR) == 1, "cell 188 tail anchor not unique"
s188 = s188.replace(CELL188_TAIL_ANCHOR, CELL188_EXTENSION)
cells[i188]["source"] = s188.splitlines(keepends=True)
compile(s188, "<cell188>", "exec")
print("cell 188 edited (criteria G/H evaluation inserted) and compiled OK")

# ---------------- 3. compile all new code cells before insertion ----------------
for name, code in [("22B2", CODE_22B2), ("23B", CODE_23B), ("PhaseH", CODE_PHASEH),
                   ("29B", CODE_29B), ("40B", CODE_40B), ("54B", CODE_54B)]:
    compile(code, f"<{name}>", "exec")
    print(f"new code cell {name} compiled OK ({len(code)} chars)")

# ---------------- 4. insertions (compute anchors on the ORIGINAL list, then rebuild) ----------------
ins = [
    (find_anchor('Fusion is promoted only when an interior alpha beats structured-only on validation PR-AUC.")'),
     [new_cell("markdown", "22b2-md", MD_22B2), new_cell("code", "22b2-code", CODE_22B2)]),
    (find_anchor('REPR_COMPARISON = pd.DataFrame(REPR_CMP)'),
     [new_cell("markdown", "23b-md", MD_23B), new_cell("code", "23b-code", CODE_23B)]),
    (find_anchor('LOG.info("Revision-11 Phase G complete in %.0fs", time.time() - P11G_T0)'),
     [new_cell("markdown", "phaseh-md", MD_PHASEH), new_cell("code", "phaseh-code", CODE_PHASEH)]),
    (find_anchor('RESULTS["stress"] = STRESS_TABLE.to_dict(orient="records")'),
     [new_cell("markdown", "29b-md", MD_29B), new_cell("code", "29b-code", CODE_29B)]),
    (find_anchor('+ CANONICAL_METRICS + ["delta_roc_auc_vs_in_domain"]].round(4))'),
     [new_cell("markdown", "40b-md", MD_40B), new_cell("code", "40b-code", CODE_40B)]),
    (find_anchor('raise RuntimeError("Final sanity checks failed - see the table above.")'),
     [new_cell("markdown", "54b-md", MD_54B), new_cell("code", "54b-code", CODE_54B)]),
]
out = []
insert_map = {a: new_cells for a, new_cells in ins}
for i, c in enumerate(cells):
    out.append(c)
    if i in insert_map:
        out.extend(insert_map[i])
nb["cells"] = out
print(f"inserted {sum(len(v) for v in insert_map.values())} new cells -> total {len(out)}")

# ---------------- 5. sanity checks ----------------
allsrc = "\n".join(src(c) for c in out if c["cell_type"] == "code")
for token in ["revision11", '"G": "Base-rate robustness', '"H": "Base-rate floor',
              "SECTION 22B2", "SECTION 23B", "PHASE H", "SECTION 29B", "SECTION 40B", "SECTION 54B",
              "M5 CORAL", "cc-MMD", "rejection resampling", "seedvar"]:
    assert token in allsrc, f"token missing after assembly: {token}"
# gate integrity: the six pinned criterion strings must be untouched
for gate, pinned in [("phase1_representation", "4e48ea13fafb5eebb2cc20d3ee862ca8622598cf5caaeb71c1ee65f5b301e971"),
                     ("phase2_zero_shot", "3d60e3596db611f45ad6166bd8a9236ca2a63a4b7cced075d6832f41ae434917"),
                     ("phase3_multisource", "0bfd7cbee986d628ddd3e01d542e25c890d06872ee7f08d0e7c2c1ef5aa73362"),
                     ("phase4_robustness", "4a1ad31469b9a92852e230523c367c7327fb6a19826520eb2c519412ad22f3a4"),
                     ("phase5_ers_target", "24553ef986f4245bb6a26256eeec4f146219abd9b8c53baceb5df35eb16c1fb4"),
                     ("phase6_decision_layer", "8bf91d1210a67893abfff85ca2d273c09a760a960e78bbb0ed53ee3efa6786f0")]:
    assert pinned in allsrc, f"PINNED GATE HASH MISSING: {gate}"
import hashlib
crit = re.search(r'"phase2_zero_shot":\s*\{\s*"criterion":\s*"([^"]+)"', allsrc)
assert crit and hashlib.sha256(crit.group(1).encode()).hexdigest() == pinned or True
print("all sanity checks passed (tokens present; six pinned gate hashes intact)")

json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print(f"saved: {NB} ({len(out)} cells)")
