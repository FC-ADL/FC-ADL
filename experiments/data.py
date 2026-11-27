"""Datasets: data/<service>_<fault>_sin_<run>/{cpu_usage.csv, fault_inject.txt}."""
import fnmatch
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

# Detection is evaluated from sample 360 onwards on all but the last 360 samples; a fault at
# sample t is at index t - 360 of the evaluated stream (and of FC-ADL's windows).
WINDOW = 360

DEPLOYMENTS = {
    'social_network': {'cp': 'compose-post-service', 'hts': 'home-timeline-service',
                       'uts': 'user-timeline-service', 'ums': 'user-mention-service'},
    'hotel_reservation': {s: f'{s}-hotelres' for s in ('search', 'recommendation', 'reservation', 'user')},
}
FAULTS = {'cpuhog': 'cpu', 'memhog': 'mem', 'memhog_redo': 'mem_nn', 'memhog_nn': 'mem_nn',
          'netdelay': 'net', 'networkdelay': 'net'}
EXCLUDED = ['nginx*', 'jaeger*', 'socialnetwork-elasticsearch*']


def datasets(deployment=None):
    pattern = re.compile(r'([a-z]+)_([a-z_]+)_sin_(\d+)$')
    out = []
    for path in sorted(DATA.iterdir()) if DATA.exists() else []:
        m = pattern.match(path.name)
        if not m or not (path / 'cpu_usage.csv').exists():
            continue
        for name, services in DEPLOYMENTS.items():
            if m.group(1) in services and (deployment is None or deployment == name):
                out.append(dict(name=path.name, deployment=name, fault=FAULTS[m.group(2)],
                                root_cause=services[m.group(1)]))
    return out


def _rename_pods(df):
    """Strip Kubernetes pod hashes; replicas of a service are numbered -1, -2, ..."""
    df = df[[c for c in df.columns if not any(fnmatch.fnmatchcase(c, p) for p in EXCLUDED)]]
    bases = ['-'.join(c.split('-')[:-2]) for c in df.columns]
    totals, seen, names = Counter(bases), defaultdict(int), []
    for base in bases:
        if totals[base] > 1:
            seen[base] += 1
            names.append(f'{base}-{seen[base]}')
        else:
            names.append(base)
    df.columns = names

    def key(name):
        parts = name.split('-')
        return parts[:-1] + [int(parts[-1]) if parts[-1].isdigit() else parts[-1]]

    return df.reindex(columns=sorted(df.columns, key=key))


def load(name, detection=False):
    """CPU usage per MS (zeros treated as missing, constant series dropped) and the fault sample."""
    path = DATA / name
    df = pd.read_csv(path / 'cpu_usage.csv')
    if detection:
        df = df.iloc[:-WINDOW]
    if 'time' in df.columns:
        df = _rename_pods(df.drop('time', axis=1))
    df = df.replace(0, np.nan).interpolate(method='linear', axis=0, limit_direction='both')
    df = df.ffill().bfill().dropna(axis=1, how='all')
    return df.loc[:, df.var() != 0], int((path / 'fault_inject.txt').read_text())
