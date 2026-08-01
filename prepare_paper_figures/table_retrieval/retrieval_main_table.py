"""T4: Table Retrieval main table — MRR@10/MAP@10 per dataset, markdown (each approach at its default row limit)."""

import pandas as pd
from pathlib import Path
import config_helpers as h


def create_table(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()

    # Slice: markdown for transformers, all for BoW/HyTrel (no rl=100-labeled variant
    # currently exists for table_retrieval, unlike table_similarity_search's ablation runs —
    # each approach here just ran at its own default row limit)
    md_mask = filtered['chart_name'].str.endswith('(md)')
    no_serial_mask = ~filtered['chart_name'].str.contains(r'\(', regex=True, na=False)
    filtered = filtered[md_mask | no_serial_mask]

    for metric_col in ('MRR@10', 'MAP@10'):
        if metric_col not in filtered.columns:
            print(f"WARNING: {metric_col} not found in retrieval data")
            continue

        # Pivot: rows=dataset, columns=chart_name
        pivoted = filtered.pivot_table(
            index='dataset',
            columns='chart_name',
            values=metric_col,
            aggfunc='mean',
        )

        # Sort datasets for consistent output, columns alphabetically by chart_name
        pivoted = pivoted.sort_index()
        pivoted = h.sort_columns_by_chart_name(pivoted)

        metric_slug = metric_col.replace('@', '').lower()
        h.write_latex_table(
            pivoted,
            plots_folder,
            filename=f'retrieval_main_table_{metric_slug}.tex',
            caption=f'Table Retrieval {metric_col} per dataset (markdown serialization). '
                    'Best per dataset in bold, second-best underlined.',
            label=f'tab:retrieval_main_{metric_slug}',
            float_fmt='.2f',
            star=True,
        )
