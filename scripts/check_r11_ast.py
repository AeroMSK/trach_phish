#!/usr/bin/env python3
"""AST-based free-name audit for the revision-11 code cells."""
import json, ast, builtins

NB = "/home/z/my-project/trac-phish-revision11.ipynb"
nb = json.load(open(NB))
cells = nb["cells"]
r11_ids = {"r11-22b2-code", "r11-23b-code", "r11-phaseh-code", "r11-29b-code", "r11-40b-code", "r11-54b-code"}

# names assigned/defined/imported anywhere in the notebook BEFORE each r11 cell
defined = set(dir(builtins))
assign_re = ast.Assign, ast.AnnAssign, ast.For, ast.With, ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom

def collect_defined(tree, into):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            into.add(node.name)
            for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                into.add(a.arg)
            if node.args.vararg:
                into.add(node.args.vararg.arg)
            if node.args.kwarg:
                into.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            into.add(node.name)
        elif isinstance(node, ast.Lambda):
            for a in node.args.args + node.args.kwonlyargs:
                into.add(a.arg)
            if node.args.vararg:
                into.add(node.args.vararg.arg)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        into.add(n.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            into.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                into.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            into.add(node.name)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    into.add(n.id)
        elif isinstance(node, ast.comprehension):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    into.add(n.id)
        elif isinstance(node, ast.With) or isinstance(node, getattr(ast, "AsyncWith", ast.With())):
            for item in node.items:
                if item.optional_vars:
                    for n in ast.walk(item.optional_vars):
                        if isinstance(n, ast.Name):
                            into.add(n.id)

ok = True
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"cell {i}: SYNTAX ERROR {e}")
        ok = False
        continue
    if c.get("id") in r11_ids:
        own_local = set()
        collect_defined(tree, own_local)
        defined |= own_local          # r11 cells also define names for the cells after them
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)
        missing = sorted(n for n in used if n not in defined and n not in own_local)
        if missing:
            print(f"cell {i:3d} [{c['id']}]: UNRESOLVED free names: {missing}")
            ok = False
        else:
            print(f"cell {i:3d} [{c['id']}]: all free names resolve OK")
    else:
        collect_defined(tree, defined)
        # r11 cells that run BEFORE this cell already added their names to `defined`

print("\nRESULT:", "PASS" if ok else "FAIL - fix the unresolved names above")
