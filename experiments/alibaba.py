"""Alibaba 2022 microservice CPU trace: data/alibaba/MSCPU-48h.csv, one row per microservice at 60 s."""
import numpy as np
import pandas as pd

from experiments.data import DATA

N_SAMPLES = 1080


def load_trace():
    df = pd.read_csv(DATA / 'alibaba' / 'MSCPU-48h.csv').T.iloc[1:N_SAMPLES + 1].fillna(0)
    df.columns = [str(c) for c in df.columns]
    return df.loc[:, df.var() > 0]


def sample(trace, size, repeat):
    df = trace.sample(n=size, axis=1, random_state=1000 * repeat + size)
    df = df.replace(0, np.nan).interpolate(method='linear', axis=0, limit_direction='both')
    df = df.ffill().bfill().dropna(axis=1, how='all')
    return df.loc[:, df.var() != 0]
