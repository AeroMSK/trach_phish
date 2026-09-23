# -*- coding: utf-8 -*-
"""Probe 2: MNB + SGD-hinge memory safety on checkpointed TF-IDF."""
import gc, time, sys
import numpy as np, joblib

def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) / 1024
    return -1

blob = joblib.load("/home/z/my-project/trac_phish_revision8_FULL_RUN/checkpoints/tfidf_matrices.joblib")["data"]
Xtr, Xva = blob["Xtr"], blob["Xva"]
del blob; gc.collect()
print(f"loaded | RSS {rss():.0f} MB", flush=True)
y = np.random.RandomState(0).randint(0, 2, Xtr.shape[0]).astype(np.int8)

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

t0 = time.time()
m = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=15, tol=1e-3, random_state=0)
m.fit(Xtr, y)
print(f"SGD-hinge a=1e-4: FIT OK in {time.time()-t0:.1f}s | RSS {rss():.0f} MB", flush=True)
del m; gc.collect()

t0 = time.time()
m = MultinomialNB(alpha=0.25)
m.fit(Xtr, y)
print(f"MultinomialNB a=0.25: FIT OK in {time.time()-t0:.1f}s | RSS {rss():.0f} MB", flush=True)
del m; gc.collect()
print("PROBE2 COMPLETE")
