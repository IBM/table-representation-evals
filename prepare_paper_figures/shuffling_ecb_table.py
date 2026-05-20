"""T12: ECB vs others — accuracy at v0, ECB vs mean of 5 semantic datasets."""

import pandas as pd
from pathlib import Path
import config_helpers as h


def create_table(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()

    filtered['base_ds'], filtered['variation'] = zip(
        *filtered['dataset'].apply(h.parse_variation)
    )

    # v0 only
    filtered = filtered[filtered['variation'] == 'v0']

    metric_col = 'TripletAccuracy_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found in shuffling data")
        return

    # ECB value for each approach — aggregate over duplicate chart_names first
    ecb = filtered[filtered['base_ds'] == 'ecb']
    ecb = ecb.groupby('chart_name')[metric_col].mean()

    # Mean of the 5 semantic datasets
    semantic = filtered[filtered['base_ds'] != 'ecb']
    semantic_mean = semantic.groupby('chart_name')[metric_col].mean()

    # Build summary
    summary = pd.DataFrame({
        'ECB': ecb,
        'Mean of 5 other datasets': semantic_mean,
    })

    h.write_latex_table(
        summary,
        plots_folder,
        filename='shuffling_ecb_table.tex',
        caption='Table Shuffling: ECB vs other datasets (v0, hi-pos/hi-neg, both perturbation). '
                'ECB = low-lexical numerical regime; Mean(5) = fetaqa, tabfact, ottqa, spider-train, ckan. '
                'Best (highest) per column in bold.',
        label='tab:shuffling_ecb',
        index_name='Approach',
        float_fmt='.4f',
        axis='columns',
        add_mean_column=True,
    )
