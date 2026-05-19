"""T4: Table Retrieval main table — MRR@10 per dataset, rl=100, markdown."""

import pandas as pd
from pathlib import Path
import config_helpers as h


def create_table(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()

    # Slice: rl=100, markdown for transformers, all for BoW/HyTrel
    filtered = h.filter_by_row_limit(filtered, 100)
    md_mask = filtered['chart_name'].str.endswith('(md)')
    no_serial_mask = ~filtered['chart_name'].str.contains(r'\(', regex=True, na=False)
    filtered = filtered[md_mask | no_serial_mask]

    metric_col = 'MRR@10_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found in retrieval data")
        return

    # Pivot: rows=dataset, columns=chart_name
    pivoted = filtered.pivot_table(
        index='dataset',
        columns='chart_name',
        values=metric_col,
        aggfunc='mean',
    )

    # Sort datasets for consistent output
    pivoted = pivoted.sort_index()

    h.write_latex_table(
        pivoted,
        plots_folder,
        filename='retrieval_main_table.tex',
        caption='Table Retrieval MRR@10 per dataset (rl=100, markdown serialization). '
                'Best per dataset in bold, second-best underlined.',
        label='tab:retrieval_main',
        float_fmt='.4f',
    )
