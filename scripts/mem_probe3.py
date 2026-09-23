# -*- coding: utf-8 -*-
"""Probe 3: saga solver memory/time on full TF-IDF (does it avoid the float64 copy?)."""
import gc, time
import numpy as np, joblib

def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) / 1024
    return -1

def vmaxrss():
    try:
        return int(open("/proc/self/status").read().split("VmHWM:")[1].split()[0]) / 1024
    except Exception:
        return -1

blob = joblib.load("/home/z/my-project/trac_phish_revision8_FULL_RUN/checkpoints/tfidf_Xtr.joblib")["data"]
Xtr = blob
print(f"loaded Xtr | RSS {rss():.0f} MB | HWM {vmaxrss():.0f} MB", flush=True)
y = np.random.RandomState(0).randint(0, 2, Xtr.shape[0]).astype(np.int8)

from sklearn.linear_model import LogisticRegression
t0 = time.time()
m = LogisticRegression(C=1.0, solver="saga", max_iter=30, tol=1e-4, random_state=0)
m.fit(Xtr, y)
print(f"saga C=1: FIT in {time.time()-t0:.1f}s, n_iter={m.n_iter_} | RSS {rss():.0f} MB | HWM {vmaxrss():.0f} MB", flush=True)
print("PROBE3 COMPLETE")
