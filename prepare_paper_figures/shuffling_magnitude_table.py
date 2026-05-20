"""T10: Magnitude grid — accuracy for hi/hi, lo/hi, hi/lo, perturbation=both, avg datasets."""

import pandas as pd
from pathlib import Path
import config_helpers as h


MAG_LABELS = {
    'v0': 'hi-pos / hi-neg',
    'v1': 'lo-pos / hi-neg',
    'v2': 'hi-pos / lo-neg',
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

    # Perturbation=both, magnitude vary (v0, v1, v2)
    target_vars = ['v0', 'v1', 'v2']
    filtered = filtered[filtered['variation'].isin(target_vars)]

    metric_col = 'TripletAccuracy_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found in shuffling data")
        return

    agg = filtered.groupby(['chart_name', 'variation'])[metric_col].mean().reset_index()
    agg['magnitude'] = agg['variation'].map(MAG_LABELS)

    pivoted = agg.pivot_table(
        index='chart_name',
        columns='magnitude',
        values=metric_col,
        aggfunc='mean',
    )[['hi-pos / hi-neg', 'lo-pos / hi-neg', 'hi-pos / lo-neg']]

    h.write_latex_table(
        pivoted,
        plots_folder,
        filename='shuffling_magnitude_table.tex',
        caption='Table Shuffling: magnitude grid. '
                'Accuracy averaged over all 6 datasets (perturbation=both, default window). '
                'Rows = positive-magnitude / negative-magnitude. Best in bold.',
        label='tab:shuffling_magnitude',
        index_name='Approach',
        float_fmt='.4f',
        axis='columns',
    )
