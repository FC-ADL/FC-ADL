"""Fig. 1: overview of the FC-ADL framework."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from figures.common import save

mpl.rcParams.update({'font.family': 'serif', 'font.size': 8, 'mathtext.fontset': 'dejavuserif'})

PANEL_FC, PANEL_EC = '#E6E6E6', '#3A4A5A'
NODE_EC, EDGE, CHANGED_C = '#1B2A38', '#111111', '#C0151B'
BLUE, PEACH, GREEN, PINK = '#BFE0F2', '#FBD9C3', '#D3EFC8', '#F6CFE6'

# example FC graph at three times
POS = np.array([(0.0, 2.0), (0.9, 2.6), (1.3, 1.7), (0.6, 0.6), (2.6, 2.9), (3.9, 3.3),
                (3.4, 2.0), (4.4, 1.5), (3.3, 0.8), (4.1, 0.2)])
BASE = [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (4, 5), (4, 6), (6, 7), (6, 8), (7, 8), (8, 9), (7, 9)]
T2 = [(0, 1), (1, 2), (1, 3), (3, 6), (2, 6), (4, 5), (4, 6), (6, 7), (6, 8), (7, 8), (8, 9)]
TN = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 6), (3, 6), (4, 6), (5, 6), (6, 7), (6, 8), (7, 8), (8, 9), (7, 9)]
CHANGED = sorted(set(BASE) ^ set(TN))


def panel(ax, x, y, w, h, title, dashed=True, fc=PANEL_FC, size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2.2', fc=fc, ec=PANEL_EC,
                                lw=1.0, ls=(0, (3, 2)) if dashed else '-', zorder=0))
    ax.text(x + w / 2, y + h - 2.0, title, ha='center', va='center', fontsize=size)


def graph(ax, origin, scale, edges, colour, sizes=None, edge_colours=None, edge_widths=None):
    p = np.asarray(origin) + POS * scale
    for k, (i, j) in enumerate(edges):
        ax.plot(*zip(p[i], p[j]), color=(edge_colours or {}).get((i, j), EDGE),
                lw=(edge_widths or {}).get((i, j), 0.9), zorder=2, solid_capstyle='round')
    for n, xy in enumerate(p):
        r = 0.38 * scale * (1 if sizes is None else sizes[n])
        ax.add_patch(Circle(xy, r, fc=colour, ec=NODE_EC, lw=0.8, zorder=3))
    return p


def arrow(ax, a, b, style='arc3,rad=0', colour='black', lw=2.2, head=10):
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle=style, arrowstyle='-|>', mutation_scale=head,
                                 lw=lw, color=colour, zorder=1, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(7.16, 3.1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 43.3)
ax.set_aspect('equal')
ax.axis('off')

# time-varying FC inference
panel(ax, 4, 25.5, 50, 17, 'Time-varying FC inference')
for k, (edges, colour, lab) in enumerate([(BASE, BLUE, '$T_1$'), (T2, PEACH, '$T_2$'), (TN, GREEN, '$T_N$')]):
    o = (8.5 + 16 * k, 28.6)
    graph(ax, o, 2.1, edges, colour)
    ax.text(o[0] + 4.6, 36.8, lab, ha='center', fontsize=8.5)
arrow(ax, (8, 27.2), (50, 27.2), lw=0.9, head=7)

# structural change detection
panel(ax, 62, 25.5, 34, 17, 'Structural change detection')
graph(ax, (64.5, 31.0), 1.8, BASE, PINK)
arrow(ax, (74.5, 37.6), (81.0, 34.0), style='arc3,rad=-0.4', colour=CHANGED_C, lw=2.2, head=9)
graph(ax, (82.5, 27.0), 1.8, sorted(set(BASE) | set(TN)), BLUE,
      edge_colours={e: CHANGED_C for e in CHANGED}, edge_widths={e: 1.2 for e in CHANGED})

# root cause localisation: node size is the total edge change
panel(ax, 62, 1.5, 34, 17, 'Root cause localisation')
strength = np.zeros(len(POS))
for i, j in CHANGED:
    strength[i] += 1
    strength[j] += 1
sizes = 0.6 + 0.6 * strength / strength.max()
p = graph(ax, (71.0, 3.0), 2.75, CHANGED, BLUE, sizes=sizes,
          edge_colours={e: CHANGED_C for e in CHANGED}, edge_widths={e: 1.5 for e in CHANGED})
for rank, n in enumerate(np.argsort(strength)[::-1][:3], 1):
    ax.text(*p[n], str(rank), ha='center', va='center_baseline', fontsize=7.5, fontweight='bold', zorder=4)

# ranked output
rows = [('Top 1', 'MS A'), ('Top 2', 'MS D'), ('Top 3', 'MS G'), ('...', '...')]
for k, (a, b) in enumerate(rows):
    y = 14.2 - 3.4 * k
    for x, txt in ((42.0, a), (50.0, b)):
        ax.add_patch(FancyBboxPatch((x, y), 7.2, 2.7, boxstyle='round,pad=0,rounding_size=0.6',
                                    fc=PANEL_FC, ec=PANEL_EC, lw=0.9, zorder=1))
        ax.text(x + 3.6, y + 1.35, txt, ha='center', va='center', fontsize=8)

# monitoring system with the MSA
panel(ax, 4, 1.0, 35, 19.5, 'Monitoring system', dashed=False, fc='white')
ax.add_patch(FancyBboxPatch((9.5, 2.4), 24, 14.2, boxstyle='round,pad=0,rounding_size=1.6', fc='white',
                            ec=PANEL_EC, lw=1.0, zorder=1))
ax.text(12.5, 15.0, 'MSA', ha='center', va='center', fontsize=8)
boxes = {'r': (19.0, 12.6, '#EFCDEB'), 'a': (13.0, 8.2, BLUE), 'b': (17.6, 8.2, '#C6EBC9'),
         'c': (22.2, 8.2, BLUE), 'd': (26.6, 8.2, PEACH)}
leaves = [(11.6, 3.5), (15.2, 3.5), (18.8, 3.5), (22.4, 3.5), (26.0, 3.5)]
w, h = 2.8, 2.6


def centre(x, y):
    return x + w / 2, y + h / 2


for src, dst in [('r', 'a'), ('r', 'b'), ('r', 'c'), ('c', 'd')]:
    s, d = centre(*boxes[src][:2]), centre(*boxes[dst][:2])
    arrow(ax, (s[0], s[1] - h / 2) if src == 'r' else (s[0] + w / 2, s[1]),
          (d[0], d[1] + h / 2) if src == 'r' else (d[0] - w / 2, d[1]), lw=0.6, head=4)
for parent, kids in (('a', [0, 1]), ('c', [2, 3]), ('d', [4])):
    s = centre(*boxes[parent][:2])
    for k in kids:
        d = centre(*leaves[k])
        arrow(ax, (s[0], s[1] - h / 2), (d[0], d[1] + h / 2), lw=0.6, head=4)
for x, y, c in boxes.values():
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.4', fc=c, ec=PANEL_EC,
                                lw=0.8, zorder=2))
for x, y in leaves:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.4', fc='#D3D3D3',
                                ec=PANEL_EC, lw=0.8, zorder=2))

# flow between stages
arrow(ax, (4.0, 10.5), (4.0, 34.0), style='bar,fraction=-0.15')
arrow(ax, (54.0, 34.0), (62.0, 34.0))
arrow(ax, (96.0, 34.0), (96.0, 10.0), style='bar,fraction=-0.15')
arrow(ax, (62.0, 10.0), (58.2, 10.0))

save(fig, 'fig1_framework', dpi=250)
