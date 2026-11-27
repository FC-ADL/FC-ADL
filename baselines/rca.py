"""Baseline root cause analysis methods. Each returns the top-k candidates and the runtime in seconds."""
import time

import networkx as nx
import numpy as np
import pandas as pd
from baro.anomaly_detection import bocpd, nsigma as baro_detect_nsigma
from baro.root_cause_analysis import nsigma as baro_rca_nsigma, robust_scorer
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased import lingam
from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis
from pyrca.analyzers.ht import HT


def _pc_adjacency(data, alpha):
    """Binary adjacency from the PC graph; undirected and bidirected edges point both ways."""
    raw = np.array(pc(np.array(data), alpha=alpha, show_progress=False).G.graph)
    n = raw.shape[0]
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if raw[j, i] == 1 and raw[i, j] == -1:
                adj[i, j] = 1
            elif (raw[i, j] == -1 and raw[j, i] == -1) or (raw[i, j] == 1 and raw[j, i] == 1):
                adj[i, j] = adj[j, i] = 1
    return adj


def _pagerank_on_reversed(adj, data, alpha, top_k):
    """PageRank over reversed causal edges weighted by absolute correlation (as MicroDiag)."""
    corr = data.corr().abs()
    metrics = list(data.columns)
    G = nx.DiGraph()
    G.add_nodes_from(metrics)
    reversed_edges = adj.T != 0
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            if reversed_edges[i, j] and corr.iat[i, j] > 0:
                G.add_edge(metrics[i], metrics[j], weight=corr.iat[i, j])
    for u in G:
        total = sum(d['weight'] for _, _, d in G.out_edges(u, data=True))
        if total > 0:
            for _, _, d in G.out_edges(u, data=True):
                d['weight'] /= total
    scores = nx.pagerank(G, alpha=alpha, weight='weight')
    return [node for node, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def rca_pc(data, top_k=5, pc_alpha=0.05, pr_alpha=0.85):
    t0 = time.time()
    ranking = _pagerank_on_reversed(_pc_adjacency(data, pc_alpha), data, pr_alpha, top_k)
    return ranking, time.time() - t0


def rca_lingam(data, top_k=5, pr_alpha=0.85):
    t0 = time.time()
    model = lingam.DirectLiNGAM()
    model.fit(data.values)
    ranking = _pagerank_on_reversed(model.adjacency_matrix_, data, pr_alpha, top_k)
    return ranking, time.time() - t0


def rca_circa(normal, abnormal, top_k=5, alpha=0.05):
    t0 = time.time()
    graph = pd.DataFrame(_pc_adjacency(normal, alpha), normal.columns, normal.columns)
    model = HT(config=HT.config_class(graph=graph, root_cause_top_k=top_k))
    model.train(normal)
    ranking = [rc[0] for rc in model.find_root_causes(abnormal).root_cause_nodes]
    return ranking, time.time() - t0


def rca_epsilon(normal, abnormal, top_k=5):
    n = min(len(normal), len(abnormal))
    normal, abnormal = normal.iloc[:n], abnormal.iloc[:n]
    t0 = time.time()
    model = EpsilonDiagnosis(config=EpsilonDiagnosis.config_class(alpha=0.05, root_cause_top_k=top_k))
    model.train(normal)
    ranking = [rc[0] for rc in model.find_root_causes(abnormal).root_cause_nodes]
    return ranking, time.time() - t0


def rca_baro(data, top_k=5):
    t0 = time.time()
    cps = bocpd(data)
    # robust_scorer needs at least two samples either side of the onset
    if not cps or cps[0] < 2 or cps[0] > len(data) - 2:
        return [], time.time() - t0
    return robust_scorer(data, anomalies=cps)['ranks'][:top_k], time.time() - t0


def rca_nsigma(data, top_k=5, start=360):
    t0 = time.time()
    data = data.copy()
    data['time'] = np.arange(len(data))
    alarms = baro_detect_nsigma(data, startsfrom=start)
    if not alarms:
        return [], time.time() - t0
    return baro_rca_nsigma(data, min(alarms))['ranks'][:top_k], time.time() - t0
