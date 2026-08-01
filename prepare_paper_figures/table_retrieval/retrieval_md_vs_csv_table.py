"""T7: Markdown vs CSV retrieval table — MRR@10/MAP@10, transformers with a serialization ablation only."""

import pandas as pd
from pathlib import Path
import config_helpers as h


def create_table(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()

    # Keep only chart_names that have a serialization suffix
    filtered = filtered[
        filtered['chart_name'].str.endswith('(md)') |
        filtered['chart_name'].str.endswith('(csv)')
    ]

    if filtered.empty:
        print("WARNING: No markdown/CSV serialization variants found in retrieval data, skipping md-vs-csv table")
        return

    for metric_col in ('MRR@10', 'MAP@10'):
        if metric_col not in filtered.columns:
            print(f"WARNING: {metric_col} not found in retrieval data")
            continue

        pivoted = filtered.pivot_table(
            index='dataset',
            columns='chart_name',
            values=metric_col,
            aggfunc='mean',
        ).sort_index()
        pivoted = h.sort_columns_by_chart_name(pivoted)

        metric_slug = metric_col.replace('@', '').lower()
        h.write_latex_table(
            pivoted,
            plots_folder,
            filename=f'retrieval_md_vs_csv_table_{metric_slug}.tex',
            caption=f'Table Retrieval {metric_col}: markdown vs CSV serialization (transformers with a serialization ablation only). '
                    'Best per dataset in bold, second-best underlined.',
            label=f'tab:retrieval_md_csv_{metric_slug}',
            float_fmt='.2f',
            star=True,
        )
