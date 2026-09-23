# -*- coding: utf-8 -*-
"""Empirical memory probe: measure RSS of model fits on the checkpointed TF-IDF matrices."""
import os, sys, time, gc, resource
import numpy as np
import joblib

def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) / 1024
    return -1

print(f"baseline RSS: {rss():.0f} MB")
blob = joblib.load("/home/z/my-project/trac_phish_revision8_FULL_RUN/checkpoints/tfidf_matrices.joblib")["data"]
Xtr, Xva = blob["Xtr"], blob["Xva"]
print(f"loaded Xtr {Xtr.shape} nnz={Xtr.nnz:,} data={Xtr.data.dtype} | RSS {rss():.0f} MB")
del blob; gc.collect()

y = np.random.RandomState(0).randint(0, 2, Xtr.shape[0]).astype(np.int8)  # dummy labels for memory test only

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

for name, make in [
    ("LogReg liblinear C=1", lambda: LogisticRegression(C=1.0, solver="liblinear", max_iter=2000)),
    ("LinearSVC C=1", lambda: LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=0)),
    ("MultinomialNB a=0.25", lambda: MultinomialNB(alpha=0.25)),
]:
    t0 = time.time()
    try:
        m = make(); m.fit(Xtr, y)
        print(f"{name}: FIT OK in {time.time()-t0:.1f}s | peak RSS {rss():.0f} MB | val acc {m.score(Xva, np.random.RandomState(1).randint(0,2,Xva.shape[0])):.3f}")
        del m; gc.collect()
        print(f"   after del: RSS {rss():.0f} MB")
    except MemoryError:
        print(f"{name}: MEMORYError at RSS {rss():.0f} MB")
        sys.exit(1)
print("PROBE COMPLETE — all fits OK")
