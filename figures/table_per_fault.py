"""Appendix table: detection F1-score and top-3 accuracy per fault -> output/table_per_fault.tex."""
from figures.common import DEPLOYMENTS, FAULTS, OUTPUT, f1, read_jsonl, top3

DETECTION = [('FC-ADL', 'fcadl'), ('BARO', 'baro'), ('N-Sigma', 'nsigma'), ('SPOT', 'spot'), ('BIRCH', 'birch')]
RCA = [('FC-ADL', 'fcadl'), ('CIRCA$^*$', 'circa'), ('N-Sigma', 'nsigma'), ('BARO', 'baro'),
       (r'$\epsilon$-Diagnosis$^*$', 'epsilon'), ('PC', 'pc'), ('LiNGAM', 'lingam')]


def block(title, methods, rows, score):
    out = [rf'\multicolumn{{5}}{{l}}{{\textit{{{title}}}}} \\']
    for label, m in methods:
        cells = [score([r for r in rows if r['method'] == m and r['fault'] == f]) for f in FAULTS]
        out.append(label + ' & ' + ' & '.join(f'{c:.2f}' for c in cells) + r' \\')
    return out


def main():
    det, rca = read_jsonl('detection.jsonl'), read_jsonl('rca.jsonl')
    lines = [r'\begin{table}[!h]', r'\centering',
             r'\caption{Detection F1-score and top-3 localisation accuracy for each fault condition. '
             r'Approaches marked $^*$ are given the ground-truth fault onset time.}',
             r'\label{tab:per_fault}', r'\footnotesize', r'\begin{tabular}{l cccc}', r'\toprule',
             ' & ' + ' & '.join(FAULTS.values()) + r' \\']
    for dep, name in DEPLOYMENTS.items():
        lines += [r'\midrule', rf'\multicolumn{{5}}{{l}}{{\textbf{{{name}}}}} \\']
        lines += block('F1-score', DETECTION, [r for r in det if r['deployment'] == dep], f1)
        lines += [r'\addlinespace'] + block('Top-3 accuracy', RCA, [r for r in rca if r['deployment'] == dep], top3)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / 'table_per_fault.tex').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
