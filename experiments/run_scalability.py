"""End-to-end RCA runtime against the number of microservices -> results/scalability.json.

Completed measurements are kept, so an interrupted run can be resumed.

    python -m experiments.run_scalability
"""
import json
import warnings

import fcadl
from baselines import rca
from experiments.alibaba import load_trace, sample
from experiments.data import RESULTS

warnings.filterwarnings('ignore')
SIZES = [5, 10, 50, 100, 150, 250]
METHODS = {'fcadl': (fcadl.localise, 3), 'nsigma': (rca.rca_nsigma, 3), 'pc': (rca.rca_pc, 1),
           'lingam': (rca.rca_lingam, 1), 'baro': (rca.rca_baro, 1)}


def main():
    out_path = RESULTS / 'scalability.json'
    runs = json.load(open(out_path)) if out_path.exists() else []
    done = {(r['size'], r['repeat'], r['method']) for r in runs}
    trace = load_trace()
    for size in SIZES:
        for repeat in range(max(r for _, r in METHODS.values())):
            df = sample(trace, size, repeat)
            for method, (fn, repeats) in METHODS.items():
                if repeat >= repeats or (size, repeat, method) in done:
                    continue
                _, runtime = fn(df)
                runs.append(dict(size=size, repeat=repeat, method=method, runtime=runtime))
                print(runs[-1], flush=True)
                with open(out_path, 'w') as f:
                    json.dump(runs, f, indent=1)


if __name__ == '__main__':
    main()
