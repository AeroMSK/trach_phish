# -*- coding: utf-8 -*-
"""Part 3: A11 lexical features, A12 char n-gram TF-IDF, A13 representation analysis."""

CELLS_3 = []

CELLS_3.append(("code", r'''# ============ CELL: [A11] LEXICAL FEATURE EXTRACTION (full scale) ============
SUSPICIOUS_TOKENS = ["login","signin","sign-in","logon","verify","verification","verify-account",
    "secure","security","account","update","confirm","banking","password","credential","webscr",
    "session","token","auth","otp","gift","bonus","invoice","payment","billing","alert","suspend",
    "unlock","limited","offer","prize","winner","free","casino","paypal","apple","google","microsoft",
    "amazon","netflix","facebook","bank","chase","wells","fargo","crypto","wallet","authorize",
    "recovery","reset","support","helpdesk","delivery","shipment","tracking","docusign","dropbox"]
SUSPICIOUS_TLDS = {"tk","ml","ga","cf","gq","xyz","top","buzz","click","link","work","rest","fit",
    "icu","cyou","cam","surf","monster","zip","mov","review","stream","download","gdn","kim","pw"}
_SUS_PAT = "(?i)(" + "|".join(sorted(SUSPICIOUS_TOKENS, key=len, reverse=True)) + ")"

def fast_host(url):
    m = re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://([^/?#]*)", url)
    host = m.group(1) if m else url.split("/", 1)[0]
    host = host.split("@")[-1].split(":", 1)[0]
    return host if "." in host or re.fullmatch(r"[\d.]+", host or "") else (host if host else None)

def _entropy(s):
    if not s: return 0.0
    c = {}
    for ch in s: c[ch] = c.get(ch, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())

def lexical_features(urls: pd.Series) -> pd.DataFrame:
    u = urls.astype(str).str.strip()
    host = u.map(fast_host).fillna("")
    host_l = host.str.lower()
    pathq = u.str.extract(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://[^/?#]*(.*)", expand=False).fillna("")
    F = pd.DataFrame(index=urls.index)
    F["url_len"]        = u.str.len()
    F["host_len"]       = host.str.len()
    F["path_len"]       = pathq.str.split("?").str[0].fillna("").str.len()
    F["query_len"]      = pathq.str.split("?", n=1).str[1].fillna("").str.len()
    F["n_dots"]         = u.str.count(r"\.")
    F["n_hyphens"]      = u.str.count(r"-")
    F["n_underscores"]  = u.str.count(r"_")
    F["n_slashes"]      = u.str.count(r"/")
    F["n_colons"]       = u.str.count(r":")
    F["n_question"]     = u.str.count(r"\?")
    F["n_equal"]        = u.str.count(r"=")
    F["n_amp"]          = u.str.count(r"&")
    F["n_percent"]      = u.str.count(r"%")
    F["n_at"]           = u.str.count(r"@")
    F["n_digits"]       = u.str.count(r"\d")
    F["digit_ratio"]    = F.n_digits / F.url_len.clip(lower=1)
    F["n_special"]      = u.str.count(r"[^a-zA-Z0-9]")
    F["special_ratio"]  = F.n_special / F.url_len.clip(lower=1)
    F["https"]          = u.str.startswith("https://").astype(np.int8)
    F["has_ip"]         = host_l.str.fullmatch(r"(\d{1,3}\.){3}\d{1,3}").fillna(False).astype(np.int8)
    F["n_subdomains"]   = (host_l.str.count(r"\.") - host_l.str.startswith("www.").astype(int)).clip(lower=0)
    F["has_port"]       = host.str.contains(":", regex=False).astype(np.int8)
    F["has_fragment"]   = u.str.contains("#", regex=False).astype(np.int8)
    F["n_query_params"] = pathq.str.split("?", n=1).str[1].fillna("").str.count(r"&") + \
                         pathq.str.split("?", n=1).str[1].fillna("").str.count(r"=")
    F["n_path_seg"]     = pathq.str.split("?").str[0].fillna("").str.count(r"/")
    F["n_susp_tokens"]  = u.str.count(_SUS_PAT)
    n_tok = u.str.count(r"[./\-_?=&@:]+").clip(lower=1)
    F["susp_token_ratio"] = F.n_susp_tokens / n_tok
    F["tld_suspicious"] = host_l.str.rsplit(".", n=1).str[-1].isin(SUSPICIOUS_TLDS).astype(np.int8)
    F["char_entropy"]   = u.map(_entropy)
    F["max_token_len"]  = u.str.split(r"[./\-_?=&@:~,+%]+").map(lambda t: max((len(x) for x in t), default=0))
    F["n_unique_chars"] = u.map(lambda s: len(set(s)))
    return F.astype(np.float32)

with stage_agent("A11", "Lexical feature extraction (27 features, all rows)"):
    cached = ckpt_load("lexical_feats", extra="lex-v2")
    if cached is None:
        FX = {}
        for nm, s in [("tr", X_tr_url), ("va", X_va_url), ("te", gb_test.url), ("pp", pp_test.url)]:
            t0 = time.time(); FX[nm] = lexical_features(s)
            print(f"    lexical[{nm}]: {FX[nm].shape} in {time.time()-t0:.1f}s", flush=True)
        ckpt_save("lexical_feats", FX, extra="lex-v2")
    else:
        FX = cached
    LEX_COLS = list(FX["tr"].columns)
    print(f"[A11] Lexical matrix: {len(LEX_COLS)} features x rows "
          f"tr={len(FX['tr']):,} va={len(FX['va']):,} te={len(FX['te']):,} pp={len(FX['pp']):,}")
    FX["tr"].describe().T[["mean","std","min","max"]].round(3).to_csv(OUT / "tables/lexical_summary.csv")'''))

CELLS_3.append(("code", r'''# ============ CELL: [A12] CHAR N-GRAM TF-IDF (two-stage vocab, memory-safe) ============
from sklearn.feature_extraction.text import TfidfVectorizer
import gc, joblib

def _rss_mb():
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS"): return int(l.split()[1]) / 1024
    except Exception: pass
    return -1

def chunked_transform(vec, texts, chunk):
    parts = []
    for i in range(0, len(texts), chunk):
        parts.append(vec.transform(texts.iloc[i:i + chunk]))
        gc.collect()
    return sp.vstack(parts, format="csr")

with stage_agent("A12", "Char n-gram TF-IDF (1-5 grams, two-stage vocabulary, float32)"):
    tf = CFG["tfidf"]
    meta = ckpt_load("tfidf_meta", extra={"tfidf": tf})
    if meta is None:
        t0 = time.time()
        vsel = TfidfVectorizer(analyzer=tf["analyzer"], ngram_range=tuple(tf["ngram_range"]),
                               min_df=tf["min_df_vocab"], max_features=tf["max_features"],
                               sublinear_tf=tf["sublinear_tf"], lowercase=True, dtype=np.float32)
        vsel.fit(X_tr_url.sample(n=tf["vocab_subsample"], random_state=SEED))
        Vocab = vsel.vocabulary_
        print(f"    stage-1 vocab: {len(Vocab):,} features (min_df={tf['min_df_vocab']} on "
              f"{tf['vocab_subsample']:,}-doc subsample) [RSS {_rss_mb():.0f} MB]", flush=True)
        del vsel; gc.collect()

        vec = TfidfVectorizer(analyzer=tf["analyzer"], ngram_range=tuple(tf["ngram_range"]),
                              vocabulary=Vocab, sublinear_tf=tf["sublinear_tf"],
                              lowercase=True, dtype=np.float32, norm="l2")
        Xtr_tfidf = vec.fit_transform(X_tr_url)              # IDF estimated on FULL train fold only (leakage-safe)
        print(f"    stage-2 full fit: {Xtr_tfidf.shape}, nnz={Xtr_tfidf.nnz:,} "
              f"({Xtr_tfidf.data.nbytes/1e6:.0f} MB) in {time.time()-t0:.1f}s [RSS {_rss_mb():.0f} MB]", flush=True); gc.collect()
        Xva_tfidf = chunked_transform(vec, X_va_url, tf["transform_chunk"])
        Xte_tfidf = chunked_transform(vec, gb_test.url, tf["transform_chunk"])
        Xpp_tfidf = chunked_transform(vec, pp_test.url, tf["transform_chunk"])
        joblib.dump(vec, OUT / "models/tfidf_vectorizer.joblib")
        # per-matrix checkpoints: RAM keeps only Xtr + Xva; Xte loaded on demand in A18;
        # Xpp never needed downstream (transfer eval re-transforms URLs) — checkpoint-only.
        ckpt_save("tfidf_Xtr", Xtr_tfidf, extra={"tfidf": tf})
        ckpt_save("tfidf_Xva", Xva_tfidf, extra={"tfidf": tf})
        ckpt_save("tfidf_Xte", Xte_tfidf, extra={"tfidf": tf})
        ckpt_save("tfidf_Xpp", Xpp_tfidf, extra={"tfidf": tf})
        meta = {"vocab": len(Vocab),
                "shapes": {"Xtr": list(Xtr_tfidf.shape), "Xva": list(Xva_tfidf.shape),
                           "Xte": list(Xte_tfidf.shape), "Xpp": list(Xpp_tfidf.shape)},
                "nnz": {"Xtr": int(Xtr_tfidf.nnz), "Xva": int(Xva_tfidf.nnz),
                        "Xte": int(Xte_tfidf.nnz), "Xpp": int(Xpp_tfidf.nnz)}}
        ckpt_save("tfidf_meta", meta, extra={"tfidf": tf})
        del Xte_tfidf, Xpp_tfidf; gc.collect()
        Xte_tfidf = None; Xpp_tfidf = None
    else:
        vec = joblib.load(OUT / "models/tfidf_vectorizer.joblib")
        Vocab = vec.vocabulary_
        Xtr_tfidf = ckpt_load("tfidf_Xtr", extra={"tfidf": tf})
        Xva_tfidf = ckpt_load("tfidf_Xva", extra={"tfidf": tf})
        Xte_tfidf = None; Xpp_tfidf = None

    sparsity = 1.0 - Xtr_tfidf.nnz / (Xtr_tfidf.shape[0] * Xtr_tfidf.shape[1])
    print(f"[A12] TF-IDF: {meta['shapes']} (train/val/GB-test/PP-test); vocab {len(Vocab):,}; "
          f"sparsity {sparsity:.5f} [RSS {_rss_mb():.0f} MB — Xtr+Xva resident, Xte/Xpp lazy]")
    with open(OUT / "diagnostics/tfidf_memory.json", "w") as f:
        json.dump({"shapes": meta["shapes"], "nnz": meta["nnz"], "vocab": len(Vocab),
                   "sparsity": sparsity, "resident": "Xtr+Xva (Xte/Xpp checkpoint-lazy)"}, f, indent=2)'''))

CELLS_3.append(("code", r'''# ============ CELL: [A13] REPRESENTATION ANALYSIS ============
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier

with stage_agent("A13", "Representation analysis (sparsity, class log-odds, SVD geometry, kNN probe)"):
    rep = []
    rep.append(("vocabulary size", len(Vocab)))
    rep.append(("avg active grams per doc (train)", round(Xtr_tfidf.nnz / Xtr_tfidf.shape[0], 1)))
    rep.append(("matrix sparsity (train)", round(1 - Xtr_tfidf.nnz / (Xtr_tfidf.shape[0] * Xtr_tfidf.shape[1]), 6)))
    # document-frequency per class via chunked bincount over CSR indices (no boolean-sparse copies)
    V_ = Xtr_tfidf.shape[1]
    _ind, _indptr = Xtr_tfidf.indices, Xtr_tfidf.indptr
    def _df_rows(rows):
        acc = np.zeros(V_, dtype=np.int64); CH = 100000
        for s in range(0, len(rows), CH):
            seg = np.concatenate([_ind[_indptr[r]:_indptr[r + 1]] for r in rows[s:s + CH]]) \
                  if len(rows[s:s + CH]) else np.empty(0, dtype=_ind.dtype)
            acc += np.bincount(seg, minlength=V_)
        return acc
    df_pos = _df_rows(np.where(y_tr == 1)[0]); df_neg = _df_rows(np.where(y_tr == 0)[0])
    gc.collect()
    rep.append(("grams present in >=1% of phish docs", int((df_pos >= 0.01 * (y_tr == 1).sum()).sum())))
    rep.append(("grams present in >=1% of legit docs", int((df_neg >= 0.01 * (y_tr == 0).sum()).sum())))

    # class log-odds (Jeffreys-smoothed document-frequency log-ratio)
    n_pos, n_neg = float((y_tr == 1).sum()), float((y_tr == 0).sum())
    lo = np.log((df_pos + 0.5) / (n_pos - df_pos + 0.5)) - np.log((df_neg + 0.5) / (n_neg - df_neg + 0.5))
    inv = {v: k for k, v in Vocab.items()}
    grams = np.array([inv[i] for i in range(len(Vocab))], dtype=object)
    order = np.argsort(-lo)
    top_phish = [(repr(grams[i]).strip("'"), round(float(lo[i]), 2)) for i in order[:30]]
    top_legit = [(repr(grams[i]).strip("'"), round(float(lo[i]), 2)) for i in order[::-1][:30]]
    pd.DataFrame(top_phish, columns=["ngram", "log_odds"]).to_csv(OUT / "tables/top_grams_logodds_phish.csv", index=False)
    pd.DataFrame(top_legit, columns=["ngram", "log_odds"]).to_csv(OUT / "tables/top_grams_logodds_legit.csv", index=False)

    # DF distribution figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].hist(df_pos, bins=80, log=True, color="#c0392b", alpha=.7, label="phish")
    ax[0].hist(df_neg, bins=80, log=True, color="#27ae60", alpha=.55, label="legit")
    ax[0].set(xlabel="document frequency", ylabel="count (log)", title="Gram document-frequency distribution")
    ax[0].legend()
    rs = np.random.RandomState(SEED)
    sidx = rs.choice(Xtr_tfidf.shape[0], size=min(5000, Xtr_tfidf.shape[0]), replace=False)
    sX, sy = Xtr_tfidf[sidx], y_tr[sidx]
    svd2 = TruncatedSVD(n_components=2, random_state=SEED).fit(sX)
    Z = svd2.transform(sX)
    ax[1].scatter(Z[sy == 0, 0], Z[sy == 0, 1], s=3, c="#27ae60", alpha=.25, label="legit", rasterized=True)
    ax[1].scatter(Z[sy == 1, 0], Z[sy == 1, 1], s=3, c="#c0392b", alpha=.25, label="phish", rasterized=True)
    ax[1].set(xlabel="SVD-1", ylabel="SVD-2", title=f"TruncatedSVD projection (evr={svd2.explained_variance_ratio_.sum():.3f})")
    ax[1].legend(markerscale=4)
    fig.savefig(OUT / "figures/representation_analysis.png", dpi=160); plt.close(fig)

    # k-NN separability probe (SVD-100, 10k sample)
    kidx = rs.choice(Xtr_tfidf.shape[0], size=10000, replace=False)
    kX = TruncatedSVD(n_components=100, random_state=SEED).fit_transform(Xtr_tfidf[kidx])
    ky = y_tr[kidx]
    h = kX.shape[0] // 2
    knn = KNeighborsClassifier(n_neighbors=5).fit(kX[:h], ky[:h])
    knn_acc = float(knn.score(kX[h:], ky[h:]))
    rep.append(("5-NN probe accuracy (SVD-100, 10k sample)", round(knn_acc, 4)))

    rep_df = pd.DataFrame(rep, columns=["metric", "value"])
    rep_df.to_csv(OUT / "tables/representation_analysis.csv", index=False)
    print(rep_df.to_string(index=False))
    print("\nTop-10 phish-leaning grams:", [g for g, _ in top_phish[:10]])
    print("Top-10 legit-leaning grams:", [g for g, _ in top_legit[:10]])'''))
