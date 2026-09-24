#!/usr/bin/env python3
"""Audit trac-phish-revision11.ipynb structure: cells, sections, CFG, gates."""
import json, re, sys

NB = '/home/z/my-project/trac-phish-revision11.ipynb'
nb = json.load(open(NB))
cells = nb['cells']
print(f"TOTAL CELLS: {len(cells)}")
print(f"NB FORMAT: nbformat={nb.get('nbformat')}.{nb.get('nbformat_minor')}")

kinds = {}
for c in cells:
    kinds[c['cell_type']] = kinds.get(c['cell_type'], 0) + 1
print(f"CELL TYPES: {kinds}")

# any cell with outputs?
n_out = sum(1 for c in cells if c.get('outputs'))
n_exec = sum(1 for c in cells if c.get('execution_count'))
print(f"CELLS WITH OUTPUTS: {n_out}  EXECUTION_COUNTS: {n_exec}")

# find section headers
print("\n=== SECTION HEADERS (first 80 chars of markdown H1/H2 + code comments 'Section N') ===")
sec_re = re.compile(r'^#+\s*(Section\s*\d+.*)', re.I)
sec_re2 = re.compile(r'#\s*(SECTION\s*\d+[\s:–—-].{0,70})', re.I)
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    first = src.strip().split('\n')[0][:90] if src.strip() else ''
    m = sec_re.match(src.strip())
    m2 = sec_re2.match(src.strip())
    if m:
        print(f"[cell {i:3d}] MD   : {m.group(1)[:85]}")
    elif m2:
        print(f"[cell {i:3d}] CODE : {m2.group(1)[:85]}")
