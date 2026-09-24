#!/usr/bin/env python3
"""A05: Cross-dataset overlap pre-scan (memory-safe: exact-string sets, no cross joins)."""
import pandas as pd
import sys

GB_DIR = "/home/z/my-project/data_raw/grambeddings/grambeddings_dataset_main"
PP_DIR = "/home/z/my-project/data_raw/phreshphish/phreshphish_transfer"
OUT = "/home/z/my-project/trac_phish_revision8_FULL_RUN/logs/agents/A05.json"


def norm(s: pd.Series) -> set:
    """Minimal normalization: strip whitespace, lowercase. Return set of exact URL strings."""
    s = s.astype("string").str.strip().str.lower()
    return set(s.dropna().unique())

def load_gb(split):
    """Stream CSV; split on FIRST comma only (URLs contain commas). Memory-safe."""
    urls = set()
    n = 0
    with open(f"{GB_DIR}/{split}.csv", "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            label, _, url = line.partition(",")
            n += 1
            url = url.strip().lower()
            if url:
                urls.add(url)
    return urls, n


def load_pp(split):
    df = pd.read_parquet(f"{PP_DIR}/phreshphish_{split}_url_only.parquet", columns=["url"])
    return norm(df["url"]), len(df)


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    res = {}

    gb_train, gb_train_n = load_gb("train")
    gb_test, gb_test_n = load_gb("test")
    pp_train, pp_train_n = load_pp("train")
    pp_test, pp_test_n = load_pp("test")

    res["sizes"] = {
        "GB_train_rows": gb_train_n, "GB_train_unique": len(gb_train),
        "GB_test_rows": gb_test_n, "GB_test_unique": len(gb_test),
        "PP_train_rows": pp_train_n, "PP_train_unique": len(pp_train),
        "PP_test_rows": pp_test_n, "PP_test_unique": len(pp_test),
    }

    # ---- exact-URL overlaps ----
    pairs = [
        ("GB_train_vs_GB_test", gb_train, gb_test, gb_test_n),
        ("GB_train_vs_PP_train", gb_train, pp_train, gb_train_n),
        ("GB_train_vs_PP_test", gb_train, pp_test, gb_test_n),
        ("GB_test_vs_PP_test", gb_test, pp_test, gb_test_n),
        ("GB_test_vs_PP_train", gb_test, pp_train, gb_test_n),
        ("PP_train_vs_PP_test", pp_train, pp_test, pp_test_n),
    ]
    res["exact_url_overlap"] = {}
    for name, a, b, denom_rows in pairs:
        inter = a & b
        res["exact_url_overlap"][name] = {
            "count": len(inter),
            "pct_of_second_set_unique": round(pct(len(inter), len(b)), 4),
            "denominator_note": f"% of {name.split('_vs_')[1]} unique URLs",
        }

    # ---- hostname-level overlap: GB_train ∩ GB_test only ----
    from urllib.parse import urlparse

    def hosts(urls, cache={}):
        out = set()
        for u in urls:
            if u in cache:
                out.add(cache[u])
                continue
            h = urlparse(u if "://" in u else "//" + u).hostname
            if h is not None:
                h = h.strip().lower().rstrip(".")
                if h:
                    out.add(h)
        return out

    gb_train_hosts = hosts(gb_train)
    gb_test_hosts = hosts(gb_test)
    host_inter = gb_train_hosts & gb_test_hosts
    res["hostname_overlap_GB_train_vs_GB_test"] = {
        "GB_train_unique_hosts": len(gb_train_hosts),
        "GB_test_unique_hosts": len(gb_test_hosts),
        "shared_hosts": len(host_inter),
        "pct_of_test_hosts": round(pct(len(host_inter), len(gb_test_hosts)), 4),
    }

    # a few example shared hosts (evidence)
    res["example_shared_hosts"] = sorted(host_inter)[:10]

    import json
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    sys.exit(main())
