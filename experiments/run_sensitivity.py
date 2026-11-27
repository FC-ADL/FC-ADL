"""FC-ADL over window size W and threshold tau on social network -> results/sensitivity_{detection,rca}.jsonl.

    python -m experiments.run_sensitivity detection|rca
"""
import json
import sys
import warnings

import fcadl
from experiments.data import RESULTS, datasets, load

warnings.filterwarnings('ignore')
WINDOWS = [60, 120, 180, 240, 300, 360, 420, 480]
THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95]


def main(task):
    sets = datasets('social_network')
    data = {ds['name']: load(ds['name'], detection=task == 'detection') for ds in sets}
    with open(RESULTS / f'sensitivity_{task}.jsonl', 'w') as out:
        for window in WINDOWS:
            for threshold in THRESHOLDS:
                for ds in sets:
                    df, fault_sample = data[ds['name']]
                    row = dict(window=window, threshold=threshold, dataset=ds['name'], fault=ds['fault'])
                    try:
                        if task == 'detection':
                            changepoint, runtime = fcadl.detect(df, window, threshold)
                            row.update(changepoint=changepoint, fault_window=fault_sample - window)
                        else:
                            ranking, runtime = fcadl.localise(df, window, threshold)
                            rc = ds['root_cause']
                            row.update(rank=ranking.index(rc) + 1 if rc in ranking else None)
                        row['runtime'] = runtime
                    except Exception as e:
                        row['error'] = repr(e)
                    out.write(json.dumps(row) + '\n')
                print(window, threshold, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
