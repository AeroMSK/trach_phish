# -*- coding: utf-8 -*-
"""Assemble trac-phish-revision8.ipynb from part files, with per-cell syntax validation."""
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
sys.path.insert(0, "/home/z/my-project/scripts/nb_parts")

import nbformat as nbf
from part1 import CELLS_1
from part2 import CELLS_2
from part3 import CELLS_3
from part4 import CELLS_4
from part5 import CELLS_5
from part6 import CELLS_6
from part7 import CELLS_7
from part8 import CELLS_8

ALL = CELLS_1 + CELLS_2 + CELLS_3 + CELLS_4 + CELLS_5 + CELLS_6 + CELLS_7 + CELLS_8
print(f"total cells: {len(ALL)} (md: {sum(1 for t,_ in ALL if t=='md')}, code: {sum(1 for t,_ in ALL if t=='code')})")

# syntax-validate every code cell before writing
errors = []
for i, (typ, src) in enumerate(ALL):
    if typ == "code":
        try:
            compile(src, f"<cell-{i}>", "exec")
        except SyntaxError as e:
            errors.append((i, str(e)))
if errors:
    for i, e in errors:
        print(f"SYNTAX ERROR in cell {i}: {e}")
    raise SystemExit(1)
print("all code cells compile OK")

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
for typ, src in ALL:
    nb.cells.append(nbf.v4.new_markdown_cell(src) if typ == "md" else nbf.v4.new_code_cell(src))

out = "/home/z/my-project/trac-phish-revision8.ipynb"
nbf.write(nb, out)
print(f"written: {out}")
