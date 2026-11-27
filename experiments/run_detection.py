"""Detection on every dataset: change point per method -> results/detection.jsonl.

    python -m experiments.run_detection [method ...]
"""
import json
import sys
import warnings

import fcadl
from baselines.detection import DETECTORS
from experiments.data import RESULTS, WINDOW, datasets, load

warnings.filterwarnings('ignore')
METHODS = {'fcadl': fcadl.detect, **DETECTORS}


def main(methods):
    out_path = RESULTS / 'detection.jsonl'
    rows = [json.loads(l) for l in open(out_path)] if out_path.exists() else []
    rows = [r for r in rows if r['method'] not in methods]
    for ds in datasets():
        df, fault_sample = load(ds['name'], detection=True)
        for method in methods:
            changepoint, runtime = METHODS[method](df)
            changepoint = None if changepoint is None else int(changepoint)
            rows.append(dict(dataset=ds['name'], deployment=ds['deployment'], fault=ds['fault'], method=method,
                             changepoint=changepoint, fault_window=fault_sample - WINDOW, runtime=runtime))
        print(ds['name'], flush=True)
        with open(out_path, 'w') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)


if __name__ == '__main__':
    main(sys.argv[1:] or list(METHODS))
