"""T14: Table Type Detection classifier table — accuracy & macro-F1 per classifier."""

import pandas as pd
from pathlib import Path
import config_helpers as h


def create_table(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()

    classifiers = ['XGBoost', 'MLP', 'KNeighbors']
    metrics = [('accuracy', 'accuracy'), ('f1_macro', 'macro-F1')]

    # Collect all (classifier, metric) pairs
    pairs = []
    for cls in classifiers:
        for metric_key, metric_label in metrics:
            col = f'{cls}_{metric_key} (↑)_mean'
            if col in filtered.columns:
                pairs.append((metric_label, cls, col))

    if not pairs:
        print("WARNING: No TTD classifier metrics found")
        return

    # Build a multi-column pivot: rows=chart_name, cols=(classifier, metric)
    records = []
    for _, row in filtered.iterrows():
        name = row['chart_name']
        for metric_label, cls, col in pairs:
            records.append({
                'approach': name,
                'classifier': cls,
                'metric': metric_label,
                'value': row[col],
            })

    recs_df = pd.DataFrame(records)
    pivoted = recs_df.pivot_table(
        index='approach',
        columns=['classifier', 'metric'],
        values='value',
        aggfunc='mean',
    )

    # Write custom LaTeX with 2-level header, spanning both columns
    with open(plots_folder / 'ttd_classifier_table.tex', 'w') as f:
        f.write('\\begin{table*}[t]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabular*}{\\textwidth}{l' + 'c' * len(pairs) + '}\n')
        f.write('\\hline\n')

        # 2-level header
        # Level 1: classifier names
        f.write('Approach')
        for metric_label, cls, _ in pairs:
            f.write(f' & \\multicolumn{{1}}{{c}}{{{cls}}}')
        f.write(' \\\\\n')

        # Level 2: metric names
        f.write('')
        for metric_label, _, _ in pairs:
            f.write(f' & {metric_label}')
        f.write(' \\\\\n')
        f.write('\\hline\n')

        for approach, row in pivoted.iterrows():
            if approach == 'Mean':
                f.write('\\hline\n')
            formatted = []
            for metric_label, cls, _ in pairs:
                v = row.get((cls, metric_label), None)
                if pd.isna(v):
                    formatted.append('---')
                else:
                    formatted.append(f'{v:.4f}')
            f.write(f'{approach} & ' + ' & '.join(formatted) + ' \\\\\n')

        f.write('\\hline\n')
        f.write('\\end{tabular*}\n')
        f.write('\\caption{Table Type Detection: accuracy and macro-F1 per classifier '
                'on WDC Schema.org (header-stripped, frozen embeddings).}\n')
        f.write('\\label{tab:ttd_classifier}\n')
        f.write('\\end{table*}\n')
