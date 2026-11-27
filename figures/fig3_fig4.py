"""Fig. 3 (detection F1-score) and Fig. 4 (top-3 accuracy) against mean runtime, per deployment."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from figures.common import DEPLOYMENTS, STYLE, bootstrap, f1, read_jsonl, save, top3, use_style

SMALL = {'epsilon', 'pc', 'lingam'}


def summarise(rows, methods, score):
    out = {}
    for m in methods:
        out[m] = {}
        for dep in DEPLOYMENTS:
            sel = [r for r in rows if r['method'] == m and r['deployment'] == dep]
            value, lo, hi = bootstrap(sel, score)
            out[m][dep] = value, np.mean([r['runtime'] for r in sel]), lo, hi
    return out


def draw(ax, res, offsets, key_loc):
    for m, v in res.items():
        label, colour, marker = STYLE[m]
        (y1, x1, lo1, hi1), (y2, x2, lo2, hi2) = v['social_network'], v['hotel_reservation']
        x2 *= 1.12
        for x, lo, hi in ((x1, lo1, hi1), (x2, lo2, hi2)):
            ax.plot([x, x], [lo, hi], color=colour, lw=0.9, alpha=0.8, zorder=2, solid_capstyle='butt')
        ax.plot([x1, x2], [y1, y2], color=colour, lw=0.9, alpha=0.5, zorder=1)
        size = 30 if m in SMALL else 42
        ax.scatter(x1, y1, s=size, marker=marker, color=colour, edgecolor='black', lw=0.5, zorder=3)
        ax.scatter(x2, y2, s=size, marker=marker, facecolor='white', edgecolor=colour, lw=1.3, zorder=3)
        dx, dy, ha, anchor = offsets[m]
        ax.annotate(label, (x1, y1) if anchor == 'sn' else (x2, y2), xytext=(dx, dy),
                    textcoords='offset points', ha=ha, va='center', fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel('Mean runtime per dataset (s)')
    ax.set_ylim(-0.06, 1.04)
    ax.grid(True, which='major', ls=':', lw=0.5, color='0.85')
    key = [Line2D([], [], color='0.3', marker='o', ls='', markersize=5, label='Social network'),
           Line2D([], [], marker='o', ls='', markersize=5, markerfacecolor='white', markeredgecolor='0.3',
                  markeredgewidth=1.2, label='Hotel reservation')]
    ax.legend(handles=key, loc=key_loc, frameon=False, fontsize=6.5, handletextpad=0.2)


def report(name, res):
    for m, v in res.items():
        print(f'{name:9s} {m:16s}', '  '.join(f'{dep}: {s:.2f} [{lo:.2f}, {hi:.2f}] ({t:.3g} s)'
                                              for dep, (s, t, lo, hi) in v.items()))


def main():
    use_style()
    det = summarise(read_jsonl('detection.jsonl'), ['fcadl', 'baro', 'nsigma', 'spot', 'birch'], f1)
    rca_rows = read_jsonl('rca.jsonl')
    rca = summarise(rca_rows, ['fcadl', 'circa', 'nsigma', 'baro', 'epsilon', 'pc', 'lingam'], top3)
    report('detection', det)
    report('rca', rca)
    for dep in DEPLOYMENTS:
        print(f'FC-ADL given the true onset, {dep}: '
              f"{top3([r for r in rca_rows if r['method'] == 'fcadl_true_onset' and r['deployment'] == dep]):.2f}")

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    draw(ax, det, {'fcadl': (7, 0, 'left', 'sn'), 'baro': (-7, 0, 'right', 'sn'),
                   'nsigma': (0, -10, 'center', 'hotel'), 'spot': (0, 9, 'center', 'sn'),
                   'birch': (6, 0, 'left', 'sn')}, 'lower right')
    ax.set_ylabel('F1-score')
    fig.tight_layout()
    save(fig, 'fig3_detection', dpi=220)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    draw(ax, rca, {'fcadl': (7, -2, 'left', 'sn'), 'circa': (6, 0, 'left', 'sn'), 'nsigma': (6, 0, 'left', 'sn'),
                   'baro': (-7, 0, 'right', 'sn'), 'epsilon': (0, 7, 'center', 'sn'),
                   'pc': (5, 5, 'left', 'sn'), 'lingam': (5, -6, 'left', 'sn')}, 'upper right')
    ax.set_ylabel('Top-3 accuracy')
    fig.tight_layout()
    save(fig, 'fig4_rca', dpi=220)


if __name__ == '__main__':
    main()
