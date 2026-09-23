import pandas as pd, numpy as np, json

BASE = "/home/z/my-project/data_raw/grambeddings/grambeddings_dataset_main"

def load(path):
    labels, urls = [], []
    bad_lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n").rstrip("\r")
            if line == "":
                labels.append(-1); urls.append(pd.NA); bad_lines.append(i); continue
            c = line.find(",")
            if c == -1:
                labels.append(-1); urls.append(line); bad_lines.append(i); continue
            lab = line[:c]; url = line[c + 1:]
            labels.append(int(lab) if lab in ("1", "2") else -1)
            urls.append(url)
            if labels[-1] == -1:
                bad_lines.append(i)
    return pd.DataFrame({"label": np.array(labels, dtype=np.int8),
                         "url": pd.array(urls, dtype="string")}), bad_lines

tr, tr_bad = load(f"{BASE}/train.csv")
te, te_bad = load(f"{BASE}/test.csv")
R = {"train_rows": len(tr), "test_rows": len(te),
     "train_unparseable_lines": tr_bad[:20], "train_n_unparseable": len(tr_bad),
     "test_unparseable_lines": te_bad[:20], "test_n_unparseable": len(te_bad)}
print("rows:", R["train_rows"], R["test_rows"], "bad:", R["train_n_unparseable"], R["test_n_unparseable"])

# --- 2. Class counts & imbalance ---
def class_counts(df, name):
    c = df["label"].value_counts().sort_index().to_dict()
    R[f"{name}_class_counts"] = {int(k): int(v) for k, v in c.items()}
    vals = list(c.values())
    R[f"{name}_imbalance_ratio"] = round(max(vals) / min(vals), 6) if min(vals) > 0 else float("inf")
    return c

class_counts(tr, "train"); class_counts(te, "test")

# --- 3. Duplicates ---
def dup_analysis(df, name):
    pair_dup = df.duplicated(subset=["label", "url"], keep=False)
    R[f"{name}_dup_label_url_rows"] = int(pair_dup.sum())
    R[f"{name}_dup_label_url_extra_rows"] = int(df.duplicated(subset=["label", "url"]).sum())  # rows beyond first
    # duplicate URLs with conflicting labels
    url_counts = df["url"].value_counts()
    multi_urls = url_counts[url_counts > 1].index
    sub = df[df["url"].isin(multi_urls)]
    conflict_urls = sub.groupby("url")["label"].nunique()
    n_conflict_urls = int((conflict_urls > 1).sum())
    R[f"{name}_urls_multi_label_conflict"] = n_conflict_urls
    R[f"{name}_rows_in_multi_url_conflict"] = int(sub[sub["url"].isin(conflict_urls[conflict_urls > 1].index)].shape[0])

dup_analysis(tr, "train"); dup_analysis(te, "test")

# Cross train ∩ test
common = set(tr["url"]) & set(te["url"])
R["cross_common_urls"] = len(common)
tr_c = tr[tr["url"].isin(common)][["url", "label"]].drop_duplicates()
te_c = te[te["url"].isin(common)][["url", "label"]].drop_duplicates()
m = tr_c.merge(te_c, on="url", suffixes=("_tr", "_te"))
R["cross_url_label_agree"] = int((m["label_tr"] == m["label_te"]).sum())
R["cross_url_label_conflict"] = int((m["label_tr"] != m["label_te"]).sum())
R["train_rows_with_common_url"] = int(tr["url"].isin(common).sum())
R["test_rows_with_common_url"] = int(te["url"].isin(common).sum())
# exact (label,url) pairs shared across splits
pair_tr = set(zip(tr["label"], tr["url"]))
pair_te = set(zip(te["label"], te["url"]))
R["cross_shared_label_url_pairs"] = len(pair_tr & pair_te)

# --- 4. URL length stats ---
def len_stats(df, name):
    L = df["url"].str.len().astype(int)
    d = {"mean": round(float(L.mean()), 2), "median": float(L.median()),
         "p95": float(np.percentile(L, 95)), "max": int(L.max()), "min": int(L.min())}
    R[f"{name}_len"] = d
    for lab in sorted(df["label"].unique()):
        Lc = df.loc[df["label"] == lab, "url"].str.len().astype(int)
        R[f"{name}_len_class{int(lab)}"] = {"mean": round(float(Lc.mean()), 2), "median": float(Lc.median()),
                                            "p95": float(np.percentile(Lc, 95)), "max": int(Lc.max()), "min": int(Lc.min())}

len_stats(tr, "train"); len_stats(te, "test")

# --- 5. Malformed URLs ---
def malformed(df, name):
    u = df["url"]
    R[f"{name}_malformed"] = {
        "empty": int((u.fillna("").str.len() == 0).sum()),
        "whitespace_only": int((u.fillna("").str.strip().str.len() == 0).sum()),
        "missing_scheme": int((~u.fillna("").str.strip().str.lower().str.startswith(("http://", "https://"))).sum()),
        "contains_space": int(u.fillna("").str.contains(" ", regex=False).sum()),
        "contains_internal_whitespace": int(u.fillna("").str.strip().str.contains(r"\s", regex=True).sum()),
        "null_na": int(u.isna().sum()),
    }

malformed(tr, "train"); malformed(te, "test")

print(json.dumps(R, indent=2))
with open("/home/z/my-project/trac_phish_revision8_FULL_RUN/logs/agents/A04_raw.json", "w") as f:
    json.dump(R, f, indent=2)
