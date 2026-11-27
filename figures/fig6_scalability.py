"""Fig. 6: end-to-end RCA runtime against the number of microservices (mean and std over repeats)."""
import json

import matplotlib.pyplot as plt
import pandas as pd

from experiments.data import RESULTS
from figures.common import STYLE, save, use_style


def main():
    use_style()
    runs = pd.DataFrame(json.load(open(RESULTS / 'scalability.json')))
    stats = runs.groupby(['method', 'size'])['runtime'].agg(['mean', 'std'])
    print(stats.round(4).to_string())
    fig = plt.figure(figsize=(3.5, 2.625))
    for m in ['fcadl', 'baro', 'nsigma', 'pc', 'lingam']:
        label, colour, marker = STYLE[m]
        s = stats.loc[m]
        plt.errorbar(s.index, s['mean'], yerr=s['std'].fillna(0), marker=marker, color=colour, label=label,
                     markersize=4, linewidth=1, capsize=2)
    plt.xlabel('Number of microservices')
    plt.ylabel('Time (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='major', ls=':', lw=0.5, color='0.85')
    plt.ylim(top=3e5)
    plt.legend(fontsize=6.5, frameon=False, loc='upper left', ncol=2)
    plt.tight_layout()
    save(fig, 'fig6_scalability')


if __name__ == '__main__':
    main()
