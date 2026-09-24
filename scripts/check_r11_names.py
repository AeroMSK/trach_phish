#!/usr/bin/env python3
"""Static name-resolution audit: every free name used in the revision-11 code cells must be
defined earlier in execution order (or be a builtin / stdlib / previously-imported symbol)."""
import json, re, builtins, sys

NB = "/home/z/my-project/trac-phish-revision11.ipynb"
nb = json.load(open(NB))
cells = nb["cells"]
r11_ids = {"r11-22b2-code", "r11-23b-code", "r11-phaseh-code", "r11-29b-code", "r11-40b-code", "r11-54b-code"}

# collect defined names from all earlier cells + imports
defined = set(dir(builtins))
defined |= {"np", "pd", "json", "os", "sys", "time", "re", "math", "hashlib", "subprocess",
            "random", "glob", "shutil", "zipfile", "warnings", "gc", "traceback", "platform",
            "scipy", "sklearn", "xgb", "lgb", "shap", "joblib", "matplotlib", "plt",
            "Dict", "List", "Tuple", "Optional", "Any", "Sequence", "Callable", "dataclass", "field",
            "tldextract_mod", "Path", "defaultdict", "display", "Markdown", "LOG", "np", "st"}
def_re = re.compile(r"^(?:def|class)\s+(\w+)|^(\w+)\s*(?::[^=]+)?=|^(\w+)(?:,\s*\w+)*\s*(?::[^=]+)?=|^from\s+\S+\s+import\s+(.+)|^import\s+(.+)|^\s{4,}(\w+)\s*[:=]", re.M)

for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    for m in def_re.finditer(s):
        for g in (m.group(1), m.group(2), m.group(3)):
            if g:
                defined.add(g)
        if m.group(4):
            defined |= {x.strip() for x in m.group(4).split(",") if x.strip().isidentifier()}
        if m.group(5):
            defined |= {x.strip().split(".")[0] for x in m.group(5).split(",") if x.strip()}
    # for-loop targets and with/as also define names
    for m in re.finditer(r"^\s*for\s+([\w,\s]+)\s+in", s, re.M):
        for name in re.split(r"[,\s]+", m.group(1).strip()):
            if name:
                defined.add(name)
    if c.get("id") in r11_ids:
        # audit THIS cell against everything defined so far (plus its own names)
        own = set()
        for m in def_re.finditer(s):
            for g in (m.group(1), m.group(2), m.group(3)):
                if g:
                    own.add(g)
        for m in re.finditer(r"^\s*for\s+([\w,\s]+)\s+in", s, re.M):
            for name in re.split(r"[,\s]+", m.group(1).strip()):
                if name:
                    own.add(name)
        # names in comprehensions / lambda args / function params are local enough for this audit
        names = set(re.findall(r"\b([a-zA-Z_]\w{2,})\b", s))
        params = set()
        for m in re.finditer(r"def\s+\w+\(([^)]*)\)", s):
            params |= {p.split("=")[0].strip().lstrip("*") for p in m.group(1).split(",") if p.strip()}
        for m in re.finditer(r"lambda\s+([^:]+):", s):
            params |= {p.strip() for p in m.group(1).split(",") if p.strip()}
        # strings inside f-string braces
        missing = sorted(n for n in names if n not in defined and n not in own and n not in params
                         and not n[0].isupper() and n not in
                         {"len", "min", "max", "abs", "int", "float", "str", "bool", "list", "dict",
                          "set", "tuple", "range", "print", "round", "sum", "enumerate", "sorted",
                          "zip", "isinstance", "getattr", "super", "open", "type", "repr", "map",
                          "filter", "iter", "next", "all", "any", "True", "False", "None"})
        upper = sorted(n for n in names if n[0].isupper() and n not in defined and n not in own)
        print(f"cell {i:3d} [{c['id']}]: missing lowercase={missing[:12] or 'NONE'} "
              f"| unchecked-caps={upper[:10] or 'NONE'}")
print("\nNOTE: names flagged above must be verified manually if not NONE.")
