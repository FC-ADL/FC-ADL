"""Baseline anomaly detectors. Each returns the change point as an index into the evaluated
stream (samples from `start` onwards), or None, and the runtime in seconds."""
import time

import ads_evt as spot
import numpy as np
from baro.anomaly_detection import bocpd, nsigma as baro_nsigma
from sklearn.cluster import Birch

START = 360


def _first(flags):
    flags = list(flags)
    return flags.index(1) if 1 in flags else None


def detect_spot(df, q=1e-2, depth=180, start=START, end=1080):
    t0 = time.time()
    alarms = []
    for col in df.columns:
        series = df[col].values[:end].astype(float)
        model = spot.dSPOT(q=q, depth=depth)
        model.fit(init_data=series[:start], data=series[start:end])
        model.initialize()
        found = model.run().get('alarms', [])
        if found:
            alarms.append(found[0])
    return (min(alarms) if alarms else None), time.time() - t0


def detect_nsigma(df, start=START):
    t0 = time.time()
    df = df.copy()
    df['time'] = np.arange(len(df))
    alarms = baro_nsigma(df, startsfrom=start)
    return (min(alarms) - start if alarms else None), time.time() - t0


def detect_birch(df, start=START):
    t0 = time.time()
    X_train, X_pred = df.iloc[:start].values, df.iloc[start:].values
    clf = Birch(threshold=0.5, branching_factor=50, n_clusters=3).fit(X_train)
    labels = np.concatenate([clf.predict(X_train), clf.predict(X_pred)])
    flips = np.concatenate([[False], labels[1:] != labels[:-1]])
    return _first(flips[start:].astype(int)), time.time() - t0


def detect_baro(df, start=START):
    t0 = time.time()
    cps = bocpd(df.iloc[start:])
    cp = cps[0] if cps and cps[0] < len(df) - start else None
    return cp, time.time() - t0


DETECTORS = {'baro': detect_baro, 'nsigma': detect_nsigma, 'spot': detect_spot, 'birch': detect_birch}
