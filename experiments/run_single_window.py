"""Time for FC-ADL to process one new window, per step, against the number of microservices
-> results/single_window.json. The pipeline is fitted on all windows but the last (not timed).

    python -m experiments.run_single_window
"""
import json
import time
import warnings

import hdbscan
import numpy as np
from scipy.spatial.distance import cdist

from experiments.alibaba import load_trace, sample
from experiments.data import RESULTS
from fcadl.correlation import TaperedCorrelation, tapered_correlations
from fcadl.embedding import affinity_sqrt, landmarks, lmds_fit, lmds_project
from fcadl.fcadl import EMBED_DIM, EPS, LANDMARK_FRACTION, THRESHOLD, WINDOW

warnings.filterwarnings('ignore')
SIZES = [5, 10, 50, 100, 250, 500, 1000]
REPEATS = 3


def run(df):
    X = np.array(df.diff().iloc[1:])
    graphs = tapered_correlations(X[:-1], WINDOW, THRESHOLD)
    flat = affinity_sqrt((graphs + 1) / 2, EPS).reshape(len(graphs), -1)
    lands = landmarks(len(graphs), LANDMARK_FRACTION)
    coords, model = lmds_fit(cdist(flat[lands], flat), lands, EMBED_DIM)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, len(coords) // 5), min_samples=5,
                                prediction_data=True, core_dist_n_jobs=-1).fit(coords)
    hdbscan.approximate_predict(clusterer, coords[-1:])
    last = len(graphs) - 1
    stream = TaperedCorrelation(X[last:last + WINDOW])
    block = X[last + 1:last + 1 + WINDOW]

    t = {}
    t0 = time.perf_counter()
    stream.update(X[last + WINDOW], X[last])
    r = stream.correlation(block.max(axis=0) == block.min(axis=0), THRESHOLD)
    t['correlation'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sq = affinity_sqrt(((r + 1) / 2)[None], EPS)[0]
    t['deltacon'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    c = lmds_project(model, cdist(sq.reshape(1, -1), flat[lands]))
    t['lmds'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    hdbscan.approximate_predict(clusterer, c)
    t['hdbscan'] = time.perf_counter() - t0
    t['total'] = sum(t.values())
    return t


def main():
    out_path = RESULTS / 'single_window.json'
    runs = json.load(open(out_path)) if out_path.exists() else []
    done = {(r['size'], r['repeat']) for r in runs}
    trace = load_trace()
    for size in SIZES:
        for repeat in range(REPEATS):
            if (size, repeat) not in done:
                runs.append(dict(size=size, repeat=repeat, **run(sample(trace, size, repeat))))
                print(runs[-1], flush=True)
                with open(out_path, 'w') as f:
                    json.dump(runs, f, indent=1)


if __name__ == '__main__':
    main()
