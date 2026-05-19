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

    metric_col = 'MRR@10_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found in retrieval data")
        return

    pivoted = filtered.pivot_table(
        index='dataset',
        columns='chart_name',
        values=metric_col,
        aggfunc='mean',
    ).sort_index()

    h.write_latex_table(
        pivoted,
        plots_folder,
        filename='retrieval_md_vs_csv_table.tex',
        caption='Table Retrieval MRR@10: markdown vs CSV serialization (rl=100, transformers only). '
                'Best per dataset in bold, second-best underlined.',
        label='tab:retrieval_md_csv',
        float_fmt='.4f',
    )
