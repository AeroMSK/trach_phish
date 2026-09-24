# Part 2: MOD 1.2 - Section 22B2 modern sequence baseline (pure-NumPy character CNN).
# Memory-safe design: gather-based conv forward (no one-hot materialisation at eval scale),
# one-hot window matmul backward only on small train batches; batched evaluation helper.

MD_22B2 = """## Section 22B2 — Modern Sequence Baseline: Character-Level CNN (Revision 11, additive)

The B1 challenger above is a linear bag-of-n-grams. The published GramBeddings protocol reports
0.9827 accuracy for a gram-embedding sequence model, so a *sequence-aware* baseline is required to
contextualise the structured branch: this section fits a small character-level CNN (DistilBERT-scale
is unreachable on this CPU-only environment - no torch/GPU - so the CNN is implemented from scratch
in NumPy with hand-written backpropagation, which is fully deterministic and auditable).

* **Input** - fixed-length byte-level character indices (140 positions, 100-symbol alphabet + OOV),
  truncation/padding documented per dataset.
* **Architecture** - 3 parallel conv widths (3/5/7) x 48 filters, ReLU, global max-pool, dense-64 ReLU,
  dropout 0.2, single sigmoid output. ~175K parameters.
* **Training** - Adam, batch 128, early stopping on source-VALIDATION PR-AUC (patience 2), fitted ONLY
  on source TRAIN URLs (same hygiene as B1: never the target, never TEST).
* **Compute cap (honest, pre-declared)** - the fit uses at most `CFG.revision11.cnn.max_rows_fit`
  (250,000) TRAIN rows per source. Full-corpus CNN training on 2 vCPU was measured as infeasible in
  the available wall-clock budget; the cap, and the fact that every comparison number below is at that
  documented scale, is recorded in the negative-results ledger and repeated in the limitations table.

This is a *reported baseline for context*, not a candidate primary detector (no TreeSHAP, so it can
never enter the explainability pipeline); it is evaluated in-domain (TEST) and zero-shot on the strict
external population of the opposite corpus, with the identical frozen evaluation code used everywhere
else (`classification_metrics`)."""

CODE_22B2 = r'''# ===================================================================================================
# SECTION 22B2 (REVISION 11, MOD 1.2) - character-level CNN baseline in pure NumPy.
# Additive only: nothing above is modified; this model can never be primary (no TreeSHAP).
# ===================================================================================================
R11CNN = CFG.revision11["cnn"]
_CNN_ALPHABET = ("abcdefghijklmnopqrstuvwxyz0123456789"
                 "-._~:/?#[]@!$&'()*+,;=% \"\\^`{}|<>")
_CNN_A = len(_CNN_ALPHABET) + 1                       # +1 = OOV byte
_CNN_BYTE2IDX = np.full(256, _CNN_A - 1, dtype=np.int16)
for _i, _ch in enumerate(_CNN_ALPHABET):
    _CNN_BYTE2IDX[ord(_ch.encode("latin-1", "ignore"))] = _i
_CNN_L = R11CNN["seq_len"]
_CNN_F = R11CNN["filters"]


def _cnn_encode(urls):
    """URLs -> (n, L) int16 alphabet indices (byte-level; non-ASCII -> OOV). Deterministic."""
    n = len(urls)
    out = np.full((n, _CNN_L), _CNN_A - 1, dtype=np.int16)
    for i, u in enumerate(urls):
        b = u.encode("utf-8", "ignore")[:_CNN_L]
        if b:
            out[i, :len(b)] = _CNN_BYTE2IDX[np.frombuffer(b, dtype=np.uint8)]
    return out


class _NumPyCNN:
    """3-width parallel conv1d (gather-based) + global max-pool + dense-64 + sigmoid. Manual backprop, Adam."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.W = [rng.normal(0, 0.05, (w * _CNN_A, _CNN_F)).astype(np.float32) for w in R11CNN["widths"]]
        self.b = [np.zeros(_CNN_F, dtype=np.float32) for _ in R11CNN["widths"]]
        nf = _CNN_F * len(R11CNN["widths"])
        self.Wd = rng.normal(0, np.sqrt(2.0 / nf), (nf, R11CNN["dense"])).astype(np.float32)
        self.bd = np.zeros(R11CNN["dense"], dtype=np.float32)
        self.Wo = rng.normal(0, np.sqrt(2.0 / R11CNN["dense"]), (R11CNN["dense"], 1)).astype(np.float32)
        self.bo = np.zeros(1, dtype=np.float32)
        self.params = [self.Wd, self.bd, self.Wo, self.bo] + self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.wd = R11CNN["weight_decay"]
        self.drop_rng = np.random.default_rng(seed + 1)

    def _conv_gather(self, Xi, W, b):
        """conv1d via embedding gather: conv[i,t,f] = sum_k W[k*A + Xi[i,k+t], f]. No one-hot copy."""
        w = W.shape[0] // _CNN_A
        Wk = W.reshape(w, _CNN_A, _CNN_F)
        T = Xi.shape[1] - w + 1
        acc = np.zeros((Xi.shape[0], T, _CNN_F), dtype=np.float32)
        for k in range(w):
            acc += Wk[k][Xi[:, k:k + T]]
        return np.maximum(acc + b, 0.0)                 # ReLU

    def forward(self, Xi, train=False):
        convs, argm, pooled = [], [], []
        for W, b in zip(self.W, self.b):
            c = self._conv_gather(Xi, W, b)
            convs.append(c)
            argm.append(c.argmax(axis=1))            # (n, F): time index of the max per filter
            pooled.append(c.max(axis=1))             # (n, F): global max over time
        Z = np.concatenate(pooled, axis=1)
        if train:
            drop = (self.drop_rng.rand(Z.shape[0], 1) > R11CNN["dropout"]).astype(np.float32) / (1.0 - R11CNN["dropout"])
        else:
            drop = 1.0
        Zd = Z * drop
        A1 = np.maximum(Zd @ self.Wd + self.bd, 0.0)
        logit = (A1 @ self.Wo + self.bo)[:, 0]
        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        return p, (Xi, convs, argm, Z, drop, Zd, A1)

    def loss(self, p, y):
        eps = 1e-7
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    def backward(self, p, y, cache):
        from numpy.lib.stride_tricks import sliding_window_view
        Xi, convs, argm, Z, drop, Zd, A1 = cache
        n = len(y)
        dlogit = (p - y) / n
        gWo = A1.T @ dlogit[:, None] + self.wd * self.Wo
        gbo = dlogit.sum().astype(np.float32).reshape(1)
        dA1 = dlogit[:, None] @ self.Wo.T
        dH = dA1 * (A1 > 0)                       # grad wrt dense pre-activation
        dWd = Zd.T @ dH + self.wd * self.Wd
        dbd = dH.sum(0)
        dZ = (dH @ self.Wd.T) * drop              # (n, nf): grad wrt pooled features
        grads = [dWd.astype(np.float32), dbd.astype(np.float32), gWo.astype(np.float32), gbo]
        gWs, gbs = [], []
        # one-hot only for the small train batch (n<=batch), then window-matmul for exact gW
        H = np.zeros((n, _CNN_L, _CNN_A), dtype=np.float32)
        H[np.arange(n)[:, None], np.arange(_CNN_L)[None, :], Xi] = 1.0
        for j, (W, b) in enumerate(zip(self.W, self.b)):
            w = W.shape[0] // _CNN_A
            dc = np.zeros_like(convs[j])
            dc[np.arange(n)[:, None], argm[j], np.arange(_CNN_F)[None, :]] = dZ[:, j * _CNN_F:(j + 1) * _CNN_F]
            dc *= (convs[j] > 0)
            win = sliding_window_view(H, (w, _CNN_A), axis=(1, 2)).reshape(n, _CNN_L - w + 1, w * _CNN_A)
            gW = win.reshape(-1, w * _CNN_A).T @ dc.reshape(-1, _CNN_F)
            gb = dc.sum((0, 1))
            gWs.append(gW.astype(np.float32))
            gbs.append(gb.astype(np.float32))
        return grads + gWs + gbs

    def step(self, grads, lr):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g.shape != p.shape:
                g = g.reshape(p.shape).astype(np.float32)
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

    def proba(self, Xi, bs=4096):
        """Batched inference; never materialises one-hot or window copies at eval scale."""
        out = np.empty(len(Xi), dtype=np.float64)
        for k in range(0, len(Xi), bs):
            p, _ = self.forward(Xi[k:k + bs], train=False)
            out[k:k + len(p)] = p
        return out


def _cnn_fit_source(src):
    """Fit the CNN on source TRAIN (capped, documented), select on source VAL. Returns bundle."""
    def _compute():
        tr = _sub_rows(partition_index(src, "train"), R11CNN["max_rows_fit"], src)
        vi = partition_index(src, "val")
        if len(vi) > R11CNN["max_rows_val"]:
            rng = np.random.default_rng(derived_seed("cnn_val", src))
            vi = np.sort(rng.choice(vi, R11CNN["max_rows_val"], replace=False))
        u_tr, y_tr = CLEAN[src]["url_raw"].values[tr], CLEAN[src]["y"].values[tr].astype(np.float32)
        u_v, y_v = CLEAN[src]["url_raw"].values[vi], CLEAN[src]["y"].values[vi]
        Xtr, Xv = _cnn_encode(u_tr), _cnn_encode(u_v)
        net = _NumPyCNN(derived_seed("cnn_init", src))
        LEDGER.record("rev11_cnn_fit", f"{src}|CNN", src, "train", "fit_model", len(tr),
                      "NumPy char CNN (documented cap)")
        n, bs = len(Xtr), R11CNN["batch"]
        best, hist, patience = (-1.0, None), [], 0
        for ep in range(R11CNN["epochs"]):
            order = np.random.default_rng(derived_seed("cnn_ep", src, ep)).permutation(n)
            t0 = time.time()
            for k in range(0, n, bs):
                idx = order[k:k + bs]
                p, cache = net.forward(Xtr[idx], train=True)
                net.step(net.backward(p, y_tr[idx], cache), R11CNN["lr"])
            pv = net.proba(Xv)
            v_auc, v_ap = fast_auc(y_v, pv), average_precision_score(y_v, pv)
            hist.append({"epoch": ep + 1, "val_roc_auc": round(float(v_auc), 5),
                         "val_pr_auc": round(float(v_ap), 5), "seconds": round(time.time() - t0, 1)})
            print(f"  [CNN {src}] epoch {ep+1}: val AUC {v_auc:.4f}  val PR-AUC {v_ap:.4f}", flush=True)
            if v_ap > best[0] + 1e-5:
                best, patience = (v_ap, [np.copy(q) for q in net.params]), 0
            else:
                patience += 1
                if patience >= R11CNN["patience"]:
                    break
        if best[1] is not None:
            for p_, q_ in zip(net.params, best[1]):
                p_[...] = q_
        thr = float(select_threshold(y_v, net.proba(Xv), CFG.threshold_metric))
        return {"net": net, "thr": thr, "history": hist, "n_fit": int(len(tr)),
                "n_val": int(len(vi)), "seq_len": _CNN_L, "alphabet": _CNN_A}
    return r7_cache(f"cnn_fit_{src}", _compute)


CNN_ROWS = []
CNN_BUNDLES: Dict[str, Any] = {}
for src in ["gram", "phresh"]:
    bundle = _cnn_fit_source(src)
    CNN_BUNDLES[src] = bundle
    net, thr = bundle["net"], bundle["thr"]
    ti = partition_index(src, "test")
    pte = net.proba(_cnn_encode(CLEAN[src]["url_raw"].values[ti]))
    yte = CLEAN[src]["y"].values[ti]
    m = classification_metrics(yte, pte, thr)
    print_classification_report(f"[{src}] R11 char CNN - IN-DOMAIN TEST (fit cap={bundle['n_fit']:,} rows)",
                                yte, pte, thr)
    CNN_ROWS.append({"source": CFG.datasets[src]["display"], "population": "in-domain TEST",
                     "n_fit_rows": bundle["n_fit"], "n_eval": int(len(yte)), **m})
    tgt = "phresh" if src == "gram" else "gram"
    idx = np.flatnonzero(EXTERNAL_MASKS[(src, tgt)]["strict_domain_unseen"])
    if len(idx):
        pe = net.proba(_cnn_encode(CLEAN[tgt]["url_raw"].values[idx]))
        ye = CLEAN[tgt]["y"].values[idx]
        me = classification_metrics(ye, pe, thr)
        print_classification_report(f"[{src}] R11 char CNN - ZERO-SHOT strict-external ({CFG.datasets[tgt]['display']})",
                                    ye, pe, thr)
        CNN_ROWS.append({"source": CFG.datasets[src]["display"],
                         "population": f"zero-shot strict-external ({CFG.datasets[tgt]['display']})",
                         "n_fit_rows": bundle["n_fit"], "n_eval": int(len(ye)), **me})
CNN_TABLE = pd.DataFrame(CNN_ROWS)
display(CNN_TABLE.round(4))
save_table(CNN_TABLE, "table22B2_revision11_char_cnn_baseline")
display(pd.DataFrame(CNN_BUNDLES["gram"]["history"] + CNN_BUNDLES["phresh"]["history"]))
save_json({s: {"history": CNN_BUNDLES[s]["history"], "n_fit": CNN_BUNDLES[s]["n_fit"],
               "n_val": CNN_BUNDLES[s]["n_val"], "seq_len": CNN_BUNDLES[s]["seq_len"],
               "alphabet_size": CNN_BUNDLES[s]["alphabet"], "threshold": CNN_BUNDLES[s]["thr"]}
           for s in CNN_BUNDLES}, DIRS["metadata"] / "revision11_char_cnn.json")
R11_LEDGER_NOTES = [{
    "item": "Mod 1.2 char CNN compute cap",
    "note": (f"Full-corpus CNN training (gram TRAIN={len(partition_index('gram', 'train')):,} rows) was "
             f"infeasible on 2 vCPU without GPU/torch; the CNN is fitted on a documented capped subset "
             f"of at most {R11CNN['max_rows_fit']:,} TRAIN rows per source (deterministic sub-selection). "
             f"Every CNN number is at that scale and is labelled with it."),
}]
print("Section 22B2 complete: NumPy char-CNN baseline fitted (documented capped subset) and evaluated "
      "in-domain and zero-shot. Reported for context; the CNN is NOT a primary candidate.")'''
