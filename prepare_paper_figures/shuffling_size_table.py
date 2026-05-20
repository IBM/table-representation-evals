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

    # Write custom LaTeX with 2-level header
    pert_types = ['Both', 'Row reorder', 'Col reorder']
    sizes = ['BIG', 'SMALL']

    with open(plots_folder / 'shuffling_size_table.tex', 'w') as f:
        f.write('\\begin{table*}[t]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabular*}{\\textwidth}{l' + 'c' * 6 + '}\n')
        f.write('\\hline\n')

        # Level 1: perturbation type (spanning 2 cols each)
        f.write('Approach')
        for pt in pert_types:
            f.write(f' & \\multicolumn{{2}}{{c}}{{{pt}}}')
        f.write(' \\\\\n')

        # Level 2: size
        f.write('')
        for pt in pert_types:
            for sz in sizes:
                f.write(f' & {sz}')
        f.write(' \\\\\n')
        f.write('\\hline\n')

        for approach, row in pivoted.iterrows():
            formatted = []
            for col in col_order:
                v = row.get(col, None)
                if pd.isna(v):
                    formatted.append('---')
                else:
                    formatted.append(f'{v:.4f}')
            f.write(f'{approach} & ' + ' & '.join(formatted) + ' \\\\\n')

        f.write('\\hline\n')
        f.write('\\end{tabular*}\n')
        f.write('\\caption{Table Shuffling: size ablation (hi-pos/hi-neg). '
                'Averaged over fetaqa, tabfact, ottqa, spider-train. '
                'CKAN and ECB excluded (no tables pass SMALL window filter).}\n')
        f.write('\\label{tab:shuffling_size}\n')
        f.write('\\end{table*}\n')
