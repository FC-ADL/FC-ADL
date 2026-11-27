"""Fig. 5: FC-ADL F1-score and top-3 accuracy over window size and threshold (social network)."""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from experiments.run_sensitivity import THRESHOLDS, WINDOWS
from figures.common import f1, read_jsonl, save, top3, use_style


def grid(task, score):
    cells = defaultdict(list)
    for r in read_jsonl(f'sensitivity_{task}.jsonl'):
        cells[(r['window'], r['threshold'])].append(r)
    return np.array([[score(cells[(w, t)]) for w in WINDOWS] for t in THRESHOLDS])


def main():
    use_style(**{'axes.spines.top': True, 'axes.spines.right': True, 'font.size': 10, 'axes.labelsize': 11,
                 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'mathtext.fontset': 'dejavuserif'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (task, score, title) in zip(axes, [('detection', f1, 'F1-score'), ('rca', top3, 'Top-3 accuracy')]):
        M = grid(task, score)
        im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(WINDOWS)), labels=WINDOWS)
        ax.set_yticks(range(len(THRESHOLDS)), labels=[f'{t:g}' for t in THRESHOLDS])
        ax.set_xlabel('Window size $W$')
        ax.set_ylabel(r'Threshold $\tau$')
        ax.set_title(title, fontsize=11)
        for i in range(len(THRESHOLDS)):
            for j in range(len(WINDOWS)):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                        color='white' if M[i, j] < 0.6 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        print(title)
        for t, row in zip(THRESHOLDS, M):
            print(f'{t:<6g}', ' '.join(f'{x:.2f}' for x in row))
    fig.tight_layout()
    save(fig, 'fig5_sensitivity')


if __name__ == '__main__':
    main()
