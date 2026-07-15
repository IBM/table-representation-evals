"""T7: Markdown vs CSV retrieval table — MRR@10, rl=100, transformers only."""

import pandas as pd
from pathlib import Path
import config_helpers as h


def create_table(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()

    # Slice: rl=100, transformers only (those with serialization variants)
    filtered = h.filter_by_row_limit(filtered, 100)
    # Keep only chart_names that have a serialization suffix
    filtered = filtered[
        filtered['chart_name'].str.endswith('(md)') |
        filtered['chart_name'].str.endswith('(csv)')
    ]

    # Keep only spider-train among spider splits
    filtered = filtered[~filtered['dataset'].isin(['spider-test', 'spider-validation'])]

    metric_col = 'MRR@10_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found in retrieval data")
        return

    # Pivot: rows=approach, columns=dataset
    pivoted = filtered.pivot_table(
        index='chart_name',
        columns='dataset',
        values=metric_col,
        aggfunc='mean',
    ).sort_index()

    # Build CI map for MRR@10
    ci_map = None
    ci_path = Path(__file__).parent / 'bootstrap_cis.csv'
    if ci_path.exists():
        ci_df = pd.read_csv(ci_path)
        ci_mrr = ci_df[(ci_df['Metric'] == 'MRR@10') & (ci_df['Dataset'] != '__aggregate__')]
        if len(ci_mrr) > 0:
            # Build mapping: (chart_name, dataset) -> (ci_lower, ci_upper)
            filtered_for_map = filtered[['chart_name', 'Approach', 'Configuration', 'dataset']].drop_duplicates()
            name_to_keys = {}
            for _, row in filtered_for_map.iterrows():
                name_to_keys[(row['Approach'], row['Configuration'], row['dataset'])] = (row['chart_name'], row['dataset'])

            ci_map = {}
            for _, row in ci_mrr.iterrows():
                key = (row['Approach'], row['Configuration'], row['Dataset'])
                mapped = name_to_keys.get(key)
                if mapped:
                    ci_map[mapped] = (row['CI_lower'], row['CI_upper'])

    h.write_latex_table(
        pivoted,
        plots_folder,
        filename='retrieval_md_vs_csv_table.tex',
        caption='Table Retrieval MRR@10: markdown vs CSV serialization (rl=100, transformers only, spider-train only among spider splits). '
                'Best per column in bold, second-best underlined. Bracketed values show bootstrapped 95\\% CIs.',
        label='tab:retrieval_md_csv',
        index_name='Approach',
        float_fmt='.4f',
        axis='columns',
        add_mean_column=True,
        ci_map=ci_map,
        tabcolsep=7.13,
    )
