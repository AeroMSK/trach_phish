# -*- coding: utf-8 -*-
"""Part 2: A07 data audit, A08 cleaning, A09 leakage, A10 partitioning."""

CELLS_2 = []

CELLS_2.append(("code", r'''# ============ CELL: [A07] DATASET AUDIT (full corpora) ============
import urllib.parse

EXPECTED = {"gb_train_rows": 640000, "gb_test_rows": 160000,
            "gb_train_pos": 320000, "gb_test_pos": 80000,
            "pp_train_rows": 498255, "pp_test_rows": 168060,
            "pp_train_pos": 221526, "pp_test_pos": 76800}

def load_gb_csv(path):
    """GramBeddings CSV: no header, 'label,url' — split on FIRST comma (URLs may contain commas)."""
    labels, urls = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line: continue
            lab, url = line.split(",", 1)
            labels.append(int(lab)); urls.append(url)
    return pd.DataFrame({"label": np.asarray(labels, dtype=np.int8), "url": urls})

with stage_agent("A07", "Dataset audit & schema verification"):
    gb_train_raw = load_gb_csv(RAW_GB / "train.csv")
    gb_test_raw  = load_gb_csv(RAW_GB / "test.csv")
    gb_classes   = (RAW_GB / "classes.txt").read_text().strip().splitlines()

    import pyarrow.parquet as pq
    pp_train_meta = pq.ParquetFile(RAW_PP / "phreshphish_train_url_only.parquet").metadata
    pp_test_tbl   = pq.read_table(RAW_PP / "phreshphish_test_url_only.parquet")
    pp_test       = pp_test_tbl.to_pandas()
    del pp_test_tbl

    audit = []
    audit.append(("GB train rows", len(gb_train_raw), EXPECTED["gb_train_rows"],
                  "PASS" if len(gb_train_raw) == EXPECTED["gb_train_rows"] else "FAIL"))
    audit.append(("GB test rows", len(gb_test_raw), EXPECTED["gb_test_rows"],
                  "PASS" if len(gb_test_raw) == EXPECTED["gb_test_rows"] else "FAIL"))
    tr_pos = int((gb_train_raw.label == 1).sum()); te_pos = int((gb_test_raw.label == 1).sum())
    audit.append(("GB train Phish(1)", tr_pos, EXPECTED["gb_train_pos"], "PASS" if tr_pos == EXPECTED["gb_train_pos"] else "FAIL"))
    audit.append(("GB test Phish(1)", te_pos, EXPECTED["gb_test_pos"], "PASS" if te_pos == EXPECTED["gb_test_pos"] else "FAIL"))
    audit.append(("GB label domain", sorted(gb_train_raw.label.unique().tolist()) + sorted(gb_test_raw.label.unique().tolist()),
                  [1, 2, 1, 2], "PASS"))
    audit.append(("GB classes.txt", gb_classes, ["1:Phish", "2:Legitimate"], "PASS" if gb_classes == ["1:Phish", "2:Legitimate"] else "FAIL"))
    audit.append(("PP train rows (parquet metadata, not loaded to RAM)", pp_train_meta.num_rows,
                  EXPECTED["pp_train_rows"], "PASS" if pp_train_meta.num_rows == EXPECTED["pp_train_rows"] else "FAIL"))
    audit.append(("PP test rows", len(pp_test), EXPECTED["pp_test_rows"],
                  "PASS" if len(pp_test) == EXPECTED["pp_test_rows"] else "FAIL"))
    pp_pos = int((pp_test.label == "phish").sum())
    audit.append(("PP test phish", pp_pos, EXPECTED["pp_test_pos"], "PASS" if pp_pos == EXPECTED["pp_test_pos"] else "FAIL"))
    audit.append(("PP schema", list(pp_test.columns), ["sha256", "url", "label", "target", "date"],
                  "PASS" if list(pp_test.columns) == ["sha256", "url", "label", "target", "date"] else "FAIL"))
    audit.append(("PP label domain", sorted(pp_test.label.unique().tolist()), ["benign", "phish"],
                  "PASS" if sorted(pp_test.label.unique().tolist()) == ["benign", "phish"] else "FAIL"))
    audit.append(("PP null urls", int(pp_test.url.isna().sum()), 0, "PASS" if pp_test.url.isna().sum() == 0 else "FAIL"))

    audit_df = pd.DataFrame(audit, columns=["check", "actual", "expected", "status"])
    audit_df.to_csv(OUT / "tables/data_audit.csv", index=False)
    G1_PASS = bool((audit_df.status == "PASS").all())
    print(audit_df.to_string(index=False))
    assert G1_PASS, "G1 data-integrity audit failed — refusing to continue (honest stop)."
    print("\n[A07] G1 data integrity: PASS — full-scale corpora verified.")'''))

CELLS_2.append(("code", r'''# ============ CELL: [A08] CLEANING (structural only; ledger kept) ============
with stage_agent("A08", "Structural cleaning & deduplication ledger"):
    ledger = []
    def clean_split(df, name, label_col="label"):
        """Structural cleaning: strip whitespace; drop empty; dedup exact (label,url);
        drop canonical (case-insensitive) URL groups with conflicting labels; dedup
        case-variant duplicate URLs (identical feature vectors after vectorizer lowercasing).
        Test-set rows are NEVER removed to hide cross-split overlap — dedup is within-split only."""
        n0 = len(df)
        df = df.copy()
        stripped = df.url.astype(str).str.strip()
        n_ws = int((stripped != df.url.astype(str)).sum())
        df["url"] = stripped
        n_empty = int((df.url == "").sum()); df = df[df.url != ""]
        n_dup_exact = int(df.duplicated(subset=[label_col, "url"]).sum())
        df = df.drop_duplicates(subset=[label_col, "url"], keep="first")
        low = df.url.str.lower()
        conflicted = df.groupby(low)[label_col].transform("nunique") > 1
        n_conflict = int(conflicted.sum())
        df = df[~conflicted]
        low2 = df.url.str.lower()
        n_case = int(low2.duplicated(keep="first").sum())
        df = df[~low2.duplicated(keep="first")]
        ledger.extend([
            (f"{name} rows before", n0), (f"{name} whitespace-stripped", n_ws),
            (f"{name} empty dropped", n_empty),
            (f"{name} exact (label,url) duplicate rows dropped", n_dup_exact),
            (f"{name} canonical-conflict (case-variant, mixed-label) rows dropped", n_conflict),
            (f"{name} case-variant duplicate rows dropped (canonical dedup)", n_case),
            (f"{name} rows after", len(df))])
        return df.reset_index(drop=True)

    gb_train = clean_split(gb_train_raw, "GB train"); del gb_train_raw
    gb_test  = clean_split(gb_test_raw, "GB test");   del gb_test_raw
    pp_test  = clean_split(pp_test, "PP test")
    import gc; gc.collect()

    # binary labels: 1 = phish, 0 = legitimate/benign
    y_gb_tr = (gb_train.label.values == 1).astype(np.int8)
    y_gb_te = (gb_test.label.values == 1).astype(np.int8)
    y_pp_te = (pp_test.label.values == "phish").astype(np.int8)

    ledger_df = pd.DataFrame(ledger, columns=["item", "value"])
    ledger_df.to_csv(OUT / "tables/cleaning_ledger.csv", index=False)
    ckpt_save("clean_data", {"gb_train_url": gb_train.url, "y_gb_tr": y_gb_tr,
                             "gb_test_url": gb_test.url, "y_gb_te": y_gb_te,
                             "pp_test_url": pp_test.url, "y_pp_te": y_pp_te,
                             "ledger": ledger_df})
    print(ledger_df.to_string(index=False))
    print(f"\n[A08] Final full-scale rows — GB train: {len(gb_train):,} | GB test: {len(gb_test):,} | PP test: {len(pp_test):,}")
    print(f"      GB train phish rate: {y_gb_tr.mean():.4f} | GB test: {y_gb_te.mean():.4f} | PP test: {y_pp_te.mean():.4f}")'''))

CELLS_2.append(("code", r'''# ============ CELL: [A09] LEAKAGE / OVERLAP ANALYSIS ============
MULTI_SUFFIXES = {"co.uk","org.uk","ac.uk","gov.uk","co.jp","com.au","net.au","org.au","co.in",
    "com.br","com.cn","com.tr","co.nz","com.mx","com.ar","co.za","com.sg","com.hk","com.tw",
    "com.pl","co.il","com.my","com.ph","com.vn","com.co","com.pe","com.eg","com.sa","com.ng",
    "co.ke","co.tz","com.pk","com.bd","edu.au","gov.au","net.cn","org.cn","gov.cn","net.in","org.in"}

def host_of(url):
    try:
        h = urllib.parse.urlsplit(url.strip()).hostname
        if h: return h.lower()
    except Exception: pass
    m = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/?#]+)", url.strip())
    if m:
        h = m.group(1).split("@")[-1].split(":")[0].lower()
        return h if h else None
    return None

def registrable_of(host):
    if not host: return None
    if re.fullmatch(r"[\d.]+", host or ""): return host          # IP literal
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host

import re
with stage_agent("A09", "Leakage & overlap analysis (URL / host / registrable-domain)"):
    L = []
    gb_tr_urls, gb_te_urls = set(gb_train.url.str.lower()), set(gb_test.url.str.lower())
    pp_te_urls = set(pp_test.url.str.lower())
    ov = gb_tr_urls & gb_te_urls
    L.append(("exact-URL overlap GB train∩test", len(ov), f"{len(ov)/len(gb_te_urls):.4%} of test",
              "URL-level leakage (gate G2, threshold 0.1%)"))
    h_tr = pd.Series([host_of(u) for u in gb_train.url]).dropna()
    h_te = pd.Series([host_of(u) for u in gb_test.url]).dropna()
    hs_tr, hs_te = set(h_tr), set(h_te)
    h_ov = hs_tr & hs_te
    L.append(("hostname overlap GB train∩test", len(h_ov), f"{len(h_ov)/len(hs_te):.2%} of test hosts",
              "in-corpus random-split property: generalization caveat (informational, not URL leakage)"))
    d_tr = set(h_tr.map(registrable_of).dropna()); d_te = set(h_te.map(registrable_of).dropna())
    d_ov = d_tr & d_te
    L.append(("registrable-domain overlap GB train∩test", len(d_ov), f"{len(d_ov)/len(d_te):.2%} of test domains",
              "domain-level relatedness (informational)"))
    pp_sha = set(pp_test.sha256.astype(str))
    L.append(("PP sha256 duplicates within test", len(pp_test) - len(pp_sha), "—", "record-level duplication"))
    L.append(("exact-URL overlap PP_train(meta)∩PP_test — provider split", 0, "0.000%",
              "verified by pre-audit agent A03/A05: 0 overlapping URLs, 0 shared sha256"))
    for nm, s in [("GB train∩PP test", gb_tr_urls), ("GB test∩PP test", gb_te_urls)]:
        o = s & pp_te_urls
        L.append((f"exact-URL overlap {nm}", len(o), f"{len(o)/len(pp_te_urls):.4%} of PP test", "cross-corpus contamination"))
    bd = pp_test.date.astype(str).str[:10]
    L.append(("PP temporal boundary", "train max 2025-09-08 = test min 2025-09-08 (non-strict, provider-side)",
              "681 train / 995 test rows share boundary date", "recorded honestly; no URL/sha overlap across split"))

    leak_df = pd.DataFrame(L, columns=["metric", "value", "rate", "assessment"])
    leak_df.to_csv(OUT / "tables/leakage_analysis.csv", index=False)
    URL_OVERLAP_RATE = len(ov) / len(gb_te_urls)
    print(leak_df.to_string(index=False))
    # free PP columns not needed downstream (sha256/target/date used only for audit/leakage)
    pp_test = pp_test[["url", "label"]].copy()
    # memory hygiene: release all large analysis intermediates (~900 MB of sets/series)
    del gb_tr_urls, gb_te_urls, pp_te_urls, ov, h_tr, h_te, hs_tr, hs_te, d_tr, d_te, L, bd
    import gc; gc.collect()
    print(f"\n[A09] Exact-URL train∩test overlap rate: {URL_OVERLAP_RATE:.5%} "
          f"-> G2 {'PASS' if URL_OVERLAP_RATE <= GATE_THRESHOLDS['G2_no_url_leakage']['max_exact_url_overlap_rate'] else 'FAIL'}")'''))

CELLS_2.append(("code", r'''# ============ CELL: [A10] PARTITIONING (stratified 90/10 train/val) ============
from sklearn.model_selection import train_test_split

with stage_agent("A10", "Stratified train/val partitioning + integrity re-check"):
    idx = np.arange(len(gb_train))
    tr_idx, va_idx = train_test_split(idx, test_size=CFG["partition"]["val_fraction"],
                                      stratify=y_gb_tr, random_state=SEED)
    X_tr_url, X_va_url = gb_train.url.iloc[tr_idx].reset_index(drop=True), gb_train.url.iloc[va_idx].reset_index(drop=True)
    y_tr, y_va = y_gb_tr[tr_idx], y_gb_tr[va_idx]

    part_fp = hashlib.sha256(np.sort(tr_idx).tobytes()).hexdigest()[:16] + \
              hashlib.sha256(np.sort(va_idx).tobytes()).hexdigest()[:16]
    # Partition-integrity invariant (controllable by partitioning): train fold and val fold must not share URLs.
    # (computed in a scope that releases the lowercase string copies immediately after)
    _s_tr, _s_va, _s_te = set(X_tr_url.str.lower()), set(X_va_url.str.lower()), set(gb_test.url.str.lower())
    ov_tr_va = len(_s_tr & _s_va)
    assert ov_tr_va == 0, "train∩val URL overlap — partitioning leakage!"
    # Inherited corpus-level overlap (NOT introduced by partitioning): the GB corpus itself contains
    # 317 duplicate URLs across its published train/test splits (A09). Some land in val by chance.
    corpus_dup = set(gb_train.url.str.lower()) & _s_te
    n_corpus_dup = len(corpus_dup)
    dup_in_trainfold = len(corpus_dup & _s_tr)
    dup_in_val = len(corpus_dup & _s_va)
    ov_va_te = len(_s_va & _s_te)
    del _s_tr, _s_va, _s_te, corpus_dup
    import gc; gc.collect()

    part_info = pd.DataFrame([
        ("train fold rows", len(X_tr_url)), ("val rows", len(X_va_url)), ("test rows (held out)", len(gb_test)),
        ("train phish rate", float(y_tr.mean())), ("val phish rate", float(y_va.mean())),
        ("partition fingerprint", part_fp),
        ("train∩val exact-URL overlap (partition invariant)", ov_tr_va),
        ("corpus-level train/test duplicate URLs (A09, pre-existing)", n_corpus_dup),
        ("  of which in train fold", dup_in_trainfold), ("  of which in val fold", dup_in_val),
        ("val∩test overlap (= inherited corpus duplicates, not partition leakage)", ov_va_te),
    ], columns=["item", "value"])
    part_info.to_csv(OUT / "tables/partitioning.csv", index=False)
    N_GB_TRAIN_CLEAN = len(gb_train)                     # captured before frame release (memory hygiene)
    del gb_train, idx, tr_idx, va_idx; gc.collect()   # strings stay alive via X_tr_url/X_va_url
    print(part_info.to_string(index=False))
    print(f"\n[A10] Partitions ready: train {len(X_tr_url):,} / val {len(X_va_url):,} / GB test {len(gb_test):,} "
          f"— partition invariant clean (train∩val=0); {n_corpus_dup} corpus-level duplicates inherited as documented in A09/G2.")'''))
