"""Root cause analysis on every dataset: ranking per method -> results/rca.jsonl.

CIRCA, epsilon-Diagnosis and fcadl_true_onset are given the ground-truth fault onset; PC and
LiNGAM use the whole recording; the other methods use their own detected change point.

    python -m experiments.run_rca [method ...]
"""
import json
import sys
import warnings

import fcadl
from baselines import rca
from experiments.data import RESULTS, WINDOW, datasets, load

warnings.filterwarnings('ignore')


def run(method, df, fault_sample):
    normal = df.iloc[5:fault_sample]
    abnormal = df.iloc[fault_sample:fault_sample + len(normal)]
    if method == 'fcadl':
        return fcadl.localise(df)
    if method == 'fcadl_true_onset':
        return fcadl.localise(df, onset_window=fault_sample - WINDOW)
    if method == 'circa':
        return rca.rca_circa(normal, abnormal)
    if method == 'epsilon':
        return rca.rca_epsilon(normal, abnormal)
    return {'baro': rca.rca_baro, 'nsigma': rca.rca_nsigma, 'pc': rca.rca_pc, 'lingam': rca.rca_lingam}[method](df)


METHODS = ['fcadl', 'fcadl_true_onset', 'circa', 'nsigma', 'baro', 'epsilon', 'pc', 'lingam']


def main(methods):
    out_path = RESULTS / 'rca.jsonl'
    rows = [json.loads(l) for l in open(out_path)] if out_path.exists() else []
    rows = [r for r in rows if r['method'] not in methods]
    for ds in datasets():
        df, fault_sample = load(ds['name'])
        for method in methods:
            ranking, runtime = run(method, df, fault_sample)
            rank = ranking.index(ds['root_cause']) + 1 if ds['root_cause'] in ranking else None
            rows.append(dict(dataset=ds['name'], deployment=ds['deployment'], fault=ds['fault'], method=method,
                             ranking=list(ranking), rank=rank, runtime=runtime))
        print(ds['name'], flush=True)
        with open(out_path, 'w') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)


if __name__ == '__main__':
    main(sys.argv[1:] or METHODS)
