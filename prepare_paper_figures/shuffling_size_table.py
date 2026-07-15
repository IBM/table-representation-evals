"""T11: Size ablation table — accuracy for BIG/SMALL × perturbation type, avg over 4 datasets."""

import pandas as pd
from pathlib import Path
import config_helpers as h


SIZE_LABELS = {
    'v9':  ('Both', 'BIG'),
    'v10': ('Row reorder', 'BIG'),
    'v11': ('Col reorder', 'BIG'),
    'v12': ('Both', 'SMALL'),
    'v13': ('Row reorder', 'SMALL'),
    'v14': ('Col reorder', 'SMALL'),
}


def create_table(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()

    # Exclude transformer CSV variants
    md_mask = filtered['chart_name'].str.endswith('(md)')
    no_serial_mask = ~filtered['chart_name'].str.contains(r'\(', regex=True, na=False)
    filtered = filtered[md_mask | no_serial_mask]

    filtered['base_ds'], filtered['variation'] = zip(
        *filtered['dataset'].apply(h.parse_variation)
    )

    # Exclude CKAN & ECB (no SMALL tables exist for them)
    filtered = filtered[~filtered['base_ds'].isin(['ckan_subset', 'ecb'])]

    # Size ablation variations
    target_vars = ['v9', 'v10', 'v11', 'v12', 'v13', 'v14']
    filtered = filtered[filtered['variation'].isin(target_vars)]

    metric_col = 'TripletAccuracy_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found in shuffling data")
        return

    agg = filtered.groupby(['chart_name', 'variation'])[metric_col].mean().reset_index()
    agg['pert_type'] = agg['variation'].map(lambda v: SIZE_LABELS[v][0])
    agg['size'] = agg['variation'].map(lambda v: SIZE_LABELS[v][1])

    pivoted = agg.pivot_table(
        index='chart_name',
        columns=['pert_type', 'size'],
        values=metric_col,
        aggfunc='mean',
    )

    # Reorder columns
    col_order = [
        ('Both', 'BIG'), ('Both', 'SMALL'),
        ('Row reorder', 'BIG'), ('Row reorder', 'SMALL'),
        ('Col reorder', 'BIG'), ('Col reorder', 'SMALL'),
    ]
    pivoted = pivoted.reindex(columns=col_order)
    # Flatten MultiIndex columns to tuples for consistent access
    pivoted.columns = pivoted.columns.tolist()  # list of tuples
    # Add Mean column
    pivoted['Mean'] = pivoted.mean(numeric_only=True, axis=1)
    all_cols = col_order + ['Mean']

    # Write custom LaTeX with 2-level header
    pert_types = ['Both', 'Row reorder', 'Col reorder']
    sizes = ['BIG', 'SMALL']

    with open(plots_folder / 'shuffling_size_table.tex', 'w') as f:
        f.write('\\begin{table*}[t]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabular*}{\\textwidth}{l' + 'c' * 6 + '|c}\n')
        f.write('\\hline\n')

        # Level 1: perturbation type (spanning 2 cols each)
        f.write('Approach')
        for pt in pert_types:
            f.write(f' & \\multicolumn{{2}}{{c}}{{{pt}}}')
        f.write(' & \\multicolumn{1}{c}{}')
        f.write(' \\\\\n')

        # Level 2: size
        f.write('')
        for pt in pert_types:
            for sz in sizes:
                f.write(f' & {sz}')
        f.write(' & Mean')
        f.write(' \\\\\n')
        f.write('\\hline\n')

        # Compute per-column best and second-best for bold/underline
        col_best = {}
        col_second = {}
        for col in all_cols:
            col_vals = pivoted[col].dropna()
            if len(col_vals) >= 2:
                sorted_vals = sorted(col_vals, reverse=True)
                col_best[col] = sorted_vals[0]
                col_second[col] = sorted_vals[1]

        for approach, row in pivoted.iterrows():
            formatted = []
            for col in all_cols:
                v = row[col]
                if pd.isna(v):
                    formatted.append('---')
                else:
                    s = f'{v:.4f}'
                    if v == col_best.get(col):
                        s = f'\\textbf{{{s}}}'
                    elif v == col_second.get(col):
                        s = f'\\underline{{{s}}}'
                    formatted.append(s)
            f.write(f'{approach} & ' + ' & '.join(formatted) + ' \\\\\n')

        f.write('\\hline\n')
        f.write('\\end{tabular*}\n')
        f.write('\\caption{Table Shuffling: perturbation-type breakdown \\textbf{and} size ablation '
                '(hi-pos/hi-neg). '
                'Averaged over fetaqa, tabfact, ottqa, spider-train. '
                'CKAN and ECB excluded (no tables pass SMALL window filter).}\n')
        f.write('\\label{tab:shuffling_size}\n')
        f.write('\\end{table*}\n')
