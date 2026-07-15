"""T14: Table Type Detection classifier table — accuracy & macro-F1 per classifier."""

import pandas as pd
import numpy as np
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

    # Compute bold/underline per column (higher is better)
    # Each column is a (classifier, metric) pair
    value_cols = [(cls, ml) for ml, cls, _ in pairs]
    cell_formats = {}  # (approach, (cls, metric)) -> (is_best, is_second)
    for col in value_cols:
        col_vals = {}
        for idx in pivoted.index:
            v = pivoted.loc[idx, col]
            if pd.notna(v):
                col_vals[idx] = v
        if len(col_vals) < 2:
            if len(col_vals) == 1:
                idx = list(col_vals.keys())[0]
                cell_formats[(idx, col)] = (True, False)
            continue
        sorted_items = sorted(col_vals.items(), key=lambda x: x[1], reverse=True)
        best_idx = sorted_items[0][0]
        second_idx = sorted_items[1][0] if len(sorted_items) > 1 else None
        cell_formats[(best_idx, col)] = (True, False)
        if second_idx is not None:
            cell_formats[(second_idx, col)] = (False, True)

    # Write custom LaTeX with 2-level header
    with open(plots_folder / 'ttd_classifier_table.tex', 'w') as f:
        f.write('\\begin{table*}[t]\n')
        f.write('\\centering\n')
        n_cols = len(pairs)
        col_spec = 'l' + 'c' * n_cols
        f.write(f'\\begin{{tabular*}}{{\\textwidth}}{{{col_spec}}}\n')
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
            formatted = []
            for metric_label, cls, _ in pairs:
                col_key = (cls, metric_label)
                v = row[col_key]
                if pd.isna(v):
                    formatted.append('---')
                else:
                    s = f'{v:.4f}'
                    fmt = cell_formats.get((approach, col_key))
                    if fmt:
                        is_best, is_second = fmt
                        if is_best:
                            s = f'\\textbf{{{s}}}'
                        elif is_second:
                            s = f'\\underline{{{s}}}'
                    formatted.append(s)
            f.write(f'{h.escape_latex(approach)} & ' + ' & '.join(formatted) + ' \\\\\n')

        f.write('\\hline\n')
        f.write('\\end{tabular*}\n')
        f.write('\\caption{Table Type Detection: accuracy and macro-F1 per classifier '
                'on WDC Schema.org (header-stripped, frozen embeddings). '
                'Best per column in bold, second-best underlined.}\n')
        f.write('\\label{tab:ttd_classifier}\n')
        f.write('\\end{table*}\n')
