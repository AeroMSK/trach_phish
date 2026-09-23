#!/usr/bin/env python3
"""Numerical validation of the revision-11 algorithm implementations BEFORE full execution."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

R11CNN = {"seq_len": 40, "filters": 6, "widths": [3, 5], "dense": 8, "dropout": 0.0,
          "lr": 1e-3, "batch": 16, "epochs": 1, "patience": 2, "weight_decay": 0.0}
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=% \"\\^`{}|<>"
A = len(ALPHABET) + 1
L, F = R11CNN["seq_len"], R11CNN["filters"]
rng = np.random.default_rng(0)
FAILS = []


class Net:                                   # float64 twin of the notebook cell
    def __init__(self, seed):
        r = np.random.default_rng(seed)
        self.W = [r.normal(0, 0.05, (w * A, F)) for w in R11CNN["widths"]]
        self.b = [np.zeros(F) for _ in R11CNN["widths"]]
        nf = F * len(R11CNN["widths"])
        self.Wd = r.normal(0, np.sqrt(2 / nf), (nf, R11CNN["dense"]))
        self.bd = np.zeros(R11CNN["dense"])
        self.Wo = r.normal(0, np.sqrt(2 / R11CNN["dense"]), (R11CNN["dense"], 1))
        self.bo = np.zeros(1)
        self.params = [self.Wd, self.bd, self.Wo, self.bo] + self.W + self.b
        self.wd = 0.0

    def _conv(self, Xi, W, b):
        w = W.shape[0] // A
        Wk = W.reshape(w, A, F)
        T = Xi.shape[1] - w + 1
        acc = np.zeros((Xi.shape[0], T, F))
        for k in range(w):
            acc += Wk[k][Xi[:, k:k + T]]
        return np.maximum(acc + b, 0.0)

    def forward(self, Xi, train=False):
        convs, argm, pooled = [], [], []
        for W, b in zip(self.W, self.b):
            c = self._conv(Xi, W, b)
            convs.append(c)
            argm.append(c.argmax(axis=1))
            pooled.append(c.max(axis=1))
        Z = np.concatenate(pooled, axis=1)
        drop = 1.0
        Zd = Z * drop
        A1 = np.maximum(Zd @ self.Wd + self.bd, 0.0)
        logit = (A1 @ self.Wo + self.bo)[:, 0]
        p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        return p, (Xi, convs, argm, Z, drop, Zd, A1)

    def backward(self, p, y, cache):
        Xi, convs, argm, Z, drop, Zd, A1 = cache
        n = len(y)
        dlogit = (p - y) / n
        gWo = A1.T @ dlogit[:, None]
        gbo = dlogit.sum().reshape(1)
        dA1 = dlogit[:, None] @ self.Wo.T
        dH = dA1 * (A1 > 0)
        dWd = Zd.T @ dH
        dbd = dH.sum(0)
        dZ = (dH @ self.Wd.T) * drop
        grads = [dWd, dbd, gWo, gbo]
        gWs, gbs = [], []
        H = np.zeros((n, L, A))
        H[np.arange(n)[:, None], np.arange(L)[None, :], Xi] = 1.0
        for j, (W, b) in enumerate(zip(self.W, self.b)):
            w = W.shape[0] // A
            dc = np.zeros_like(convs[j])
            dc[np.arange(n)[:, None], argm[j], np.arange(F)[None, :]] = dZ[:, j * F:(j + 1) * F]
            dc *= (convs[j] > 0)
            win = sliding_window_view(H, (w, A), axis=(1, 2)).reshape(n, L - w + 1, w * A)
            gWs.append(win.reshape(-1, w * A).T @ dc.reshape(-1, F))
            gbs.append(dc.sum((0, 1)))
        return grads + gWs + gbs


# ---------------- 1. gradient check (float64, magnitude-aware) ----------------
net = Net(7)
Xi = rng.integers(0, A, size=(9, L)).astype(np.int16)
y = rng.integers(0, 2, size=9).astype(np.float64)
p, cache = net.forward(Xi, train=True)
grads = net.backward(p, y, cache)
bce = lambda pp: float(-np.mean(y * np.log(pp + 1e-12) + (1 - y) * np.log(1 - pp + 1e-12)))

def num_grad(pi, eps=1e-6):
    g = np.zeros_like(net.params[pi])
    flat, gf = net.params[pi].ravel(), g.ravel()
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        lp = bce(net.forward(Xi, train=True)[0])
        flat[i] = orig - eps
        lm = bce(net.forward(Xi, train=True)[0])
        flat[i] = orig
        gf[i] = (lp - lm) / (2 * eps)
    return g

worst = 0.0
for pi in range(len(net.params)):
    ng = num_grad(pi)
    ag = grads[pi].reshape(net.params[pi].shape)
    na, nn_ = np.linalg.norm(ag), np.linalg.norm(ng)
    if max(na, nn_) < 1e-8:            # dead filter / zero gradient: skip (noise-dominated)
        continue
    rel = np.linalg.norm(ag - ng) / max(na, nn_)
    worst = max(worst, rel)
    if rel > 5e-3:
        FAILS.append(f"grad param {pi} rel {rel:.2e}")
print(f"1. CNN gradient check (float64, magnitude-aware): worst relative error = {worst:.2e}",
      "PASS" if worst < 5e-3 else "FAIL")

# ---------------- 2. CORAL (exact w.r.t. the SHRUNKEN covariance) ----------------
def sym_sqrt_inv(C, eig_floor=1e-10):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, max(w.max(), 1e-30) * eig_floor, None)
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T

def coral_check(Xs, Xt, eig_floor):
    d = Xs.shape[1]
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Ct = np.cov(Xt, rowvar=False)
    _, Cs_ih = sym_sqrt_inv(np.cov(Xs, rowvar=False), eig_floor)
    Ct_h, _ = sym_sqrt_inv(Ct, eig_floor)
    Xal = (Xs - mu_s) @ (Cs_ih @ Ct_h) + mu_t
    err_cov = np.abs(np.cov(Xal, rowvar=False) - Ct).max() / np.abs(Ct).max()
    err_mu = np.abs(Xal.mean(0) - mu_t).max()
    return err_cov, err_mu, Xal

d = 12
# well-conditioned (realistic): full-rank + isotropic floor
Ms = rng.normal(0, 1, (d, d)); Ms = Ms @ Ms.T + 0.5 * np.eye(d)
Mt = rng.normal(0, 1, (d, d)) * 1.7; Mt = Mt @ Mt.T + 0.5 * np.eye(d)
Xs = rng.multivariate_normal(np.full(d, 3.0), Ms, 5000)
Xt = rng.multivariate_normal(np.full(d, -1.0), Mt, 6000)
err_cov, err_mu, _ = coral_check(Xs, Xt, 1e-10)
print(f"2a. CORAL well-conditioned, rel eig-floor 1e-10: cov rel err {err_cov:.2e}, mean err {err_mu:.2e}",
      "PASS" if err_cov < 1e-6 and err_mu < 1e-9 else "FAIL")
if not (err_cov < 1e-6 and err_mu < 1e-9):
    FAILS.append("coral-well")
# ill-conditioned: output must stay finite and bounded (stability), alignment best-effort
Mi = np.diag(np.logspace(-6, 3, d))
Xi_s = rng.multivariate_normal(np.zeros(d), Mi @ rng.normal(0, .1, (d, d)).T @ Mi + 1e-8 * np.eye(d), 4000)
Xi_t = rng.multivariate_normal(np.ones(d), Mi * 3.0, 4000)
err_covi, err_mui, Xal_i = coral_check(Xi_s, Xi_t, 1e-10)
stable = np.all(np.isfinite(Xal_i)) and np.abs(Xal_i).max() < 1e6
print(f"2b. CORAL ill-conditioned stability: finite&bounded={stable}, cov rel err {err_covi:.2e} "
      "(best-effort by design)")
if not stable:
    FAILS.append("coral-ill")

# ---------------- 3. base-rate metrics vs sklearn (group-end AP, first-crossing p@r) ----------------
from sklearn.metrics import roc_auc_score, average_precision_score

def br_metrics(ps, ns, thr, recalls):
    s = np.concatenate([ps, ns])
    yy = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
    order = np.argsort(-s, kind="stable")
    ys, ss = yy[order], s[order]
    n = len(s)
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), n - 1]
    tp = np.cumsum(ys)[ends]
    fp = np.cumsum(1.0 - ys)[ends]
    P = float(len(ps))
    rec_g = tp / P
    prec_g = tp / np.maximum(tp + fp, 1.0)
    out = {"roc_auc": roc_auc_score(yy, s),
           "average_precision": float(np.sum(np.diff(np.r_[0.0, rec_g]) * prec_g))}
    for r in recalls:
        k = np.flatnonzero(rec_g >= r)
        out[f"precision_at_recall_{r:g}"] = float(prec_g[k[0]]) if k.size else 0.0
    return out

ps = np.clip(rng.beta(5, 2, 300), 0, 1)
ns = np.clip(rng.beta(2, 5, 2000), 0, 1)
mm = br_metrics(ps, ns, 0.5, [0.5, 0.7, 0.9])
ap_ref = average_precision_score(np.concatenate([np.ones(300), np.zeros(2000)]),
                                 np.concatenate([ps, ns]))
ap_err = abs(mm["average_precision"] - ap_ref)
print(f"3. step AP vs sklearn: {mm['average_precision']:.6f} vs {ap_ref:.6f} (err {ap_err:.2e})",
      "PASS" if ap_err < 1e-10 else "FAIL")
if ap_err >= 1e-10:
    FAILS.append("ap")

# ---------------- 4. precision@recall hand check (ties included) ----------------
ps2 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25])
ns2 = np.concatenate([np.full(12, 0.65), [0.1]])
mm2 = br_metrics(ps2, ns2, 0.5, [0.7])
# first threshold reaching recall>=0.7: after .4P -> TP=7, FP=12 -> 7/19
exp = 7 / 19
got = mm2["precision_at_recall_0.7"]
print(f"4. precision@recall0.7 hand check (tie group of 12): got {got:.4f}, expected {exp:.4f}",
      "PASS" if abs(got - exp) < 1e-12 else "FAIL")
if abs(got - exp) >= 1e-12:
    FAILS.append("p@r")

print("\nRESULT:", "ALL PASS" if not FAILS else f"FAILURES: {FAILS}")
