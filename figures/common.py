import json

import matplotlib as mpl
import numpy as np

from experiments.data import RESULTS, ROOT

OUTPUT = ROOT / 'output'
DEPLOYMENTS = {'social_network': 'Social network', 'hotel_reservation': 'Hotel reservation'}
FAULTS = {'cpu': 'CPU', 'mem': 'Mem.', 'mem_nn': 'Mem. (NN)', 'net': 'Net.'}

# label, colour (Okabe-Ito), marker; shared by every figure
STYLE = {
    'fcadl': ('FC-ADL', '#D55E00', 'o'),
    'baro': ('BARO', '#0072B2', 's'),
    'nsigma': ('N-Sigma', '#009E73', '^'),
    'spot': ('SPOT', '#56B4E9', 'v'),
    'birch': ('BIRCH', '#CC79A7', 'D'),
    'circa': ('CIRCA*', '#E69F00', 'P'),
    'epsilon': ('ε-Diagnosis*', '#882255', 'h'),
    'pc': ('PC+PR', '#000000', 'X'),
    'lingam': ('LiNGAM+PR', '#7F7F7F', 'D'),
}


def use_style(**extra):
    mpl.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, **extra})


def save(fig, name, dpi=300):
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUTPUT / f'{name}.png', dpi=dpi, bbox_inches='tight')


def read_jsonl(name):
    return [json.loads(line) for line in open(RESULTS / name) if line.strip()]


def f1(rows):
    """Per-dataset scoring as BARO: TP if a change point is found, FN if not, and an additional FP
    if the first change point is before the fault."""
    found = [r for r in rows if r.get('changepoint') is not None]
    tp, fp = len(found), sum(r['changepoint'] < r['fault_window'] for r in found)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / len(rows)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


def top3(rows):
    return sum(r.get('rank') is not None and r['rank'] <= 3 for r in rows) / len(rows)


def bootstrap(rows, score, n=5000, seed=0):
    """Score and 95% percentile bootstrap interval, resampling datasets."""
    rng = np.random.default_rng(seed)
    samples = [score([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(n)]
    return score(rows), *np.percentile(samples, [2.5, 97.5])
