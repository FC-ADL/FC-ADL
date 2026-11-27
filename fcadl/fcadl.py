"""FC-ADL: anomaly detection and localisation from time-varying functional connectivity."""
import time

import hdbscan
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .correlation import tapered_correlations
from .embedding import affinity_sqrt, landmarks, lmds_fit

WINDOW = 360
THRESHOLD = 0.05
EPS = 1e-2
LANDMARK_FRACTION = 0.25
EMBED_DIM = 10


def fc_graphs(df, window=WINDOW, threshold=THRESHOLD):
    """Thresholded correlations of the differenced metrics for every window.

    Window w covers samples w+1 .. w+window, so the first window containing a fault that starts
    at sample t is t - window.
    """
    df = df.replace(0, np.nan).interpolate(method='linear', axis=0, limit_direction='both')
    X = np.array(df.ffill().bfill().diff().iloc[1:])
    return tapered_correlations(X, window, threshold)


def embed(graphs, eps=EPS, landmark_fraction=LANDMARK_FRACTION, dim=EMBED_DIM):
    """LMDS embedding of the DeltaCon distances between FC graphs."""
    flat = affinity_sqrt((graphs + 1) / 2, eps).reshape(len(graphs), -1)
    lands = landmarks(len(graphs), landmark_fraction)
    coords, _ = lmds_fit(cdist(flat[lands], flat), lands, dim)
    return coords


def cluster(coords):
    return hdbscan.HDBSCAN(min_cluster_size=max(5, len(coords) // 5), min_samples=5, metric='euclidean',
                           prediction_data=True, core_dist_n_jobs=-1).fit_predict(coords)


def changepoint_from_labels(labels):
    """Earliest window assigned to a different cluster from the preceding non-noise window."""
    labels = np.asarray(labels)
    idx = np.where(labels != -1)[0]
    if idx.size < 2:
        return None
    change = np.where(labels[idx[:-1]] != labels[idx[1:]])[0]
    return int(idx[change[0] + 1]) if change.size else None


def change_graph(graphs, changepoint, threshold=THRESHOLD):
    """|G_a - G_b| between the mean FC graphs before and from the change point."""
    centroids = []
    for part in (graphs[:changepoint], graphs[changepoint:]):
        g = np.zeros_like(part[0])
        for m in part:
            g += m
        g /= len(part)
        g /= np.max(g)
        g[(g < threshold) & (g > -threshold)] = 0
        centroids.append(g)
    return np.abs(centroids[1] - centroids[0])


def detect(df, window=WINDOW, threshold=THRESHOLD):
    """Change point (window index) of the FC structure, or None, and the runtime in seconds."""
    start = time.time()
    labels = cluster(embed(fc_graphs(df, window, threshold)))
    return changepoint_from_labels(labels), time.time() - start


def localise(df, window=WINDOW, threshold=THRESHOLD, top_k=5, onset_window=None):
    """Top-k candidate root causes ranked by total change in FC dependencies, and the runtime.

    If onset_window is given, it replaces FC-ADL's own change point.
    """
    start = time.time()
    graphs = fc_graphs(df, window, threshold)
    changepoint = onset_window if onset_window is not None else changepoint_from_labels(cluster(embed(graphs)))
    if changepoint is None:
        return [], time.time() - start
    strength = pd.Series(change_graph(graphs, changepoint, threshold).sum(axis=1), index=df.columns)
    return strength.nlargest(top_k).index.tolist(), time.time() - start
