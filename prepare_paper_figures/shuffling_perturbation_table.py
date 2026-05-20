"""T9: Perturbation-type breakdown — accuracy for both/row/col, averaged over datasets."""

import pandas as pd
from pathlib import Path
import config_helpers as h


# Map variation → perturbation type label
PERT_LABELS = {
    'v0': 'Both',
    'v3': 'Row reorder',
    'v6': 'Col reorder',
}


def create_table(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()

    # Exclude transformer CSV variants
    md_mask = filtered['chart_name'].str.endswith('(md)')
    no_serial_mask = ~filtered['chart_name'].str.contains(r'\(', regex=True, na=False)
    filtered = filtered[md_mask | no_serial_mask]

    # Parse variation from dataset name
    filtered['base_ds'], filtered['variation'] = zip(
        *filtered['dataset'].apply(h.parse_variation)
    )

    # Keep only the main-grid hi-pos/hi-neg perturbation variants
    target_vars = ['v0', 'v3', 'v6']
    filtered = filtered[filtered['variation'].isin(target_vars)]

    metric_col = 'TripletAccuracy_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found in shuffling data")
        return

    # Average over all 6 datasets
    agg = filtered.groupby(['chart_name', 'variation'])[metric_col].mean().reset_index()
    agg['perturbation'] = agg['variation'].map(PERT_LABELS)

    pivoted = agg.pivot_table(
        index='chart_name',
        columns='perturbation',
        values=metric_col,
        aggfunc='mean',
    )[['Both', 'Row reorder', 'Col reorder']]

    h.write_latex_table(
        pivoted,
        plots_folder,
        filename='shuffling_perturbation_table.tex',
        caption='Table Shuffling: perturbation-type breakdown. '
                'Accuracy averaged over all 6 datasets (hi-pos/hi-neg, default window). '
                'Best per perturbation type in bold.',
        label='tab:shuffling_perturbation',
        index_name='Approach',
        float_fmt='.4f',
        axis='columns',
        add_mean_column=True,
    )
