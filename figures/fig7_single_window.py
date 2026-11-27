"""Fig. 7: time for FC-ADL to process one new window, per step (mean and std over repeats)."""
import json

import matplotlib.pyplot as plt
import pandas as pd

from experiments.data import RESULTS
from figures.common import save, use_style

STEPS = {
    'correlation': ('Correlation update', '#56B4E9'),
    'deltacon': ('DeltaCon', '#CC79A7'),
    'lmds': ('LMDS projection', '#E69F00'),
    'hdbscan': ('HDBSCAN predict', '#009E73'),
    'total': ('Total', '#000000'),
}


def main():
    use_style()
    runs = pd.DataFrame(json.load(open(RESULTS / 'single_window.json')))
    stats = runs.groupby('size')[list(STEPS)].agg(['mean', 'std'])
    print(stats.round(6).to_string())
    fig = plt.figure(figsize=(3.5, 2.625))
    for step, (label, colour) in STEPS.items():
        plt.errorbar(stats.index, stats[(step, 'mean')], yerr=stats[(step, 'std')].fillna(0), marker='o',
                     linestyle='--' if step == 'total' else '-', color=colour, label=label,
                     markersize=4, linewidth=1, capsize=2)
    plt.xlabel('Number of microservices')
    plt.ylabel('Time (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='major', ls=':', lw=0.5, color='0.85')
    plt.legend(fontsize=6.5, frameon=False)
    plt.tight_layout()
    save(fig, 'fig7_single_window')


if __name__ == '__main__':
    main()
