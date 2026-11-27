"""Fig. 2: FC-ADL on a social network recording where two of the three home-timeline-service pods
are faulted after 3600 s. A: response time and CPU usage; B: 2-D LMDS embedding of the FC graphs,
coloured by time; C: change in FC between the normal and faulty states, nodes sized by edge change."""
import itertools
from collections import defaultdict

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist

import fcadl
from experiments.data import RESULTS
from fcadl.embedding import affinity_sqrt, landmarks, lmds_fit
from fcadl.fcadl import (EPS, LANDMARK_FRACTION, WINDOW, change_graph, changepoint_from_labels, cluster, embed,
                         fc_graphs)
from figures.common import save

STEP, DURATION, FAULT = 5, 7200, 3600
FAULTY_PODS = ['home-timeline-service-65b4dd89b5-t6prg', 'home-timeline-service-65b4dd89b5-q8dn2']
N_EDGES = 40


def service_type(pod):
    return '-'.join(pod.split('-')[:-2])


def draw_graph(ax, dG, pods, keep, pos, groups, colours, diam):
    sub = np.abs(dG[np.ix_(keep, keep)])
    np.fill_diagonal(sub, 0)
    iu = np.triu_indices_from(sub, 1)
    sub[sub < np.sort(sub[iu])[-N_EDGES]] = 0
    G = nx.from_numpy_array(sub)
    xy = np.array([pos[n] for n in range(len(keep))])

    def limits():
        ax.set_xlim(xy[:, 0].min() - 1, xy[:, 0].max() + 1)
        ax.set_ylim(xy[:, 1].min() - 1, xy[:, 1].max() + 1)

    limits()
    ax.set_aspect('equal')
    ax.apply_aspect()
    pts_per_unit = (ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]) * 72 / ax.figure.dpi
    for stype, nodes in groups.items():
        p = xy[nodes]
        c = p.mean(axis=0)
        radius = (np.linalg.norm(p - c, axis=1) + diam[nodes] / 2 / pts_per_unit).max() + 0.12
        ax.add_patch(Circle(c, radius, color=colours[stype], alpha=0.2, zorder=0))
    w = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    nx.draw(G, pos, ax=ax, node_size=diam ** 2, linewidths=0.8, edgecolors='k',
            node_color=[colours[service_type(pods[keep[n]])] for n in G.nodes()],
            width=2 * w / w.max(), edge_color='k')
    limits()
    for n, i in enumerate(keep):
        if pods[i] in FAULTY_PODS:
            ax.add_patch(Circle(pos[n], (diam[n] / 2 - 0.6) / pts_per_unit, fill=False, hatch='////',
                                edgecolor='crimson', lw=0, zorder=5))


def main():
    mpl.rcParams.update({'font.family': 'serif', 'hatch.linewidth': 1.0})
    cpu = pd.read_csv(RESULTS / 'fig2' / 'cpu_usage.csv')
    rtime = pd.read_csv(RESULTS / 'fig2' / 'response_time.csv', index_col='t')['response_time_ms']
    df = cpu.replace(0, np.nan).interpolate(method='linear', axis=0, limit_direction='both')
    df = df.ffill().bfill().dropna(axis=1, how='all')
    df = df.loc[:, df.var() != 0]

    graphs = fc_graphs(df)
    changepoint = changepoint_from_labels(cluster(embed(graphs)))
    dG = change_graph(graphs, changepoint)
    strength = dG.sum(axis=1)
    ranking = pd.Series(strength, index=df.columns).nlargest(5).index.tolist()
    assert ranking == fcadl.localise(df)[0]
    flat = affinity_sqrt((graphs + 1) / 2, EPS).reshape(len(graphs), -1)
    lands = landmarks(len(graphs), LANDMARK_FRACTION)
    coords, _ = lmds_fit(cdist(flat[lands], flat), lands, 2)
    win_t = (np.arange(len(graphs)) + WINDOW + 2) * STEP
    print('top-5:', ranking)
    print(f'change point: window {changepoint}, t = {win_t[changepoint]} s (fault at {FAULT} s)')

    fig = plt.figure(figsize=(11, 6.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.06)
    left = outer[0].subgridspec(6, 1, height_ratios=[0.42, 0.42, 0.30, 0.2, 0.12, 2.4], hspace=0.12)
    axA1 = fig.add_subplot(left[0])
    axA2 = fig.add_subplot(left[1], sharex=axA1)
    axCB = fig.add_subplot(left[3])
    axB = fig.add_subplot(left[5])
    axC = fig.add_subplot(outer[1])

    for ax in (axA1, axA2):
        ax.axvspan(0, FAULT, color='#9fdf9f', alpha=0.75, lw=0, zorder=0)
        ax.axvspan(FAULT, DURATION, color='#f4a582', alpha=0.75, lw=0, zorder=0)
        ax.axvline(FAULT, color='k', ls='--', lw=0.8)
        ax.set_xlim(WINDOW * STEP, DURATION)
        ax.yaxis.grid(True, linestyle='--', color='k', lw=0.6)
        ax.set_axisbelow(False)
    axA1.plot(rtime.index, rtime.values / 1000, color='crimson', lw=0.7)
    axA1.set_ylabel('RT (s)', fontsize=8)
    axA1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, integer=True))
    axA1.tick_params(labelbottom=False, labelsize=7)
    cmap = plt.get_cmap('ocean_r', cpu.shape[1] + 4)
    t_cpu = (np.arange(len(cpu)) + 1) * STEP
    for i, pod in enumerate(cpu.columns):
        axA2.plot(t_cpu, cpu[pod].values / STEP, lw=0.5, alpha=0.7, color=cmap(i + 2))
    axA2.set_ylabel('CPU usage\n(cores)', fontsize=8)
    axA2.set_xlabel('Time (s)', fontsize=8)
    axA2.tick_params(labelsize=7)
    axA1.set_title('A', loc='left', fontweight='bold')

    sc = axB.scatter(coords[:, 0], coords[:, 1], c=win_t, cmap='viridis', s=14)
    centres = [coords[i:i + 25].mean(axis=0) for i in range(0, len(coords), 25)]
    for p0, p1 in zip(centres[:-1], centres[1:]):
        axB.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='->', color='k', alpha=0.9, lw=1.2))
    sc.set_clim(WINDOW * STEP, DURATION)
    cb = fig.colorbar(sc, cax=axCB, orientation='horizontal')
    cb.set_ticks([])
    cb.outline.set_linewidth(0.5)
    axB.set_xticks([])
    axB.set_yticks([])
    axB.set_title('B', loc='left', fontweight='bold')

    pods = list(df.columns)
    keep = [i for i, p in enumerate(pods) if service_type(p).endswith('-service')]
    groups = defaultdict(list)
    for n, i in enumerate(keep):
        groups[service_type(pods[i])].append(n)
    colours = dict(zip(groups, itertools.cycle(plt.cm.tab20.colors)))
    pos = {}
    for g, nodes in enumerate(groups.values()):
        a = 2 * np.pi * g / len(groups)
        centre = 3.5 * np.array([np.cos(a), np.sin(a)])
        for j, n in enumerate(nodes):
            b = 2 * np.pi * j / len(nodes)
            pos[n] = centre + 0.5 * np.array([np.cos(b), np.sin(b)]) * (len(nodes) > 1)
    s = strength[keep]
    diam = 22 * s / s.max()
    draw_graph(axC, dG, pods, keep, pos, groups, colours, diam)
    axC.set_title('Change in FC between normal and faulty states', fontsize=9)
    for rank, n in enumerate(np.argsort(s)[::-1][:3], 1):
        axC.annotate(str(rank), pos[n], ha='center', va='center', fontsize=9, fontweight='bold', zorder=6,
                     path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    axC.set_title('C', loc='left', fontweight='bold')
    faulted = (mpl.patches.Patch(facecolor='white', edgecolor='k', lw=0.8),
               mpl.patches.Patch(facecolor='none', edgecolor='crimson', hatch='////', lw=0))
    axC.legend(handles=[Circle((0, 0), 0.5, color=c) for c in colours.values()] + [faulted],
               labels=[s.replace('-service', '') for s in colours] + ['faulted pod'],
               handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=1)},
               loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8, frameon=False,
               columnspacing=1.0, handlelength=1.2)
    save(fig, 'fig2_method')


if __name__ == '__main__':
    main()
