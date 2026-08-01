"""P6: Variation heatmap — TripletAccuracy across variations × approaches, avg over datasets."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


# Same perturbation-type / magnitude vocabulary as shuffling_perturbation_table.py
# and shuffling_magnitude_table.py, combined per variation (see configs/dataset/table_shuffling.yaml).
VARIATION_LABELS = {
    'v0': 'Both, hi/hi',
    'v1': 'Both, lo/hi',
    'v2': 'Both, hi/lo',
    'v3': 'Row, hi/hi',
    'v4': 'Row, lo/hi',
    'v5': 'Row, hi/lo',
    'v6': 'Col, hi/hi',
    'v7': 'Col, lo/hi',
    'v8': 'Col, hi/lo',
}


def create_heatmap(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()
    filtered['base_ds'], filtered['variation'] = zip(
        *filtered['dataset'].apply(h.parse_variation)
    )

    metric_col = 'TripletAccuracy'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found")
        return

    # Average over all datasets per (approach, variation)
    agg = filtered.groupby(['chart_name', 'variation'])[metric_col].mean().reset_index()

    # Pivot: rows=variation, cols=chart_name
    pivoted = agg.pivot(index='variation', columns='chart_name', values=metric_col)
    pivoted = h.sort_columns_by_chart_name(pivoted)

    # Grouped by magnitude band (easiest -> hardest), reorder type as sub-rows within
    # each band: it's the magnitude, not the reorder type, that drives difficulty.
    row_order = ['v4', 'v7', 'v1', 'v3', 'v6', 'v0', 'v5', 'v8', 'v2']
    pivoted = pivoted.reindex([v for v in row_order if v in pivoted.index])

    fig, ax = plt.subplots(figsize=(max(10, len(pivoted.columns) * 1.2),
                                   max(6, len(pivoted) * 0.5)))
    im = ax.imshow(pivoted.values, aspect='auto', cmap='RdYlGn', vmin=0.4, vmax=1.0)

    # Separator lines between magnitude bands
    for boundary in (2.5, 5.5):
        ax.axhline(boundary, color='black', linewidth=1.5)

    ax.set_xticks(range(len(pivoted.columns)))
    ax.set_xticklabels(pivoted.columns, rotation=25, ha='right', fontsize=14)
    ax.set_yticks(range(len(pivoted.index)))
    ax.set_yticklabels([VARIATION_LABELS.get(v, v) for v in pivoted.index], fontsize=14)

    for i in range(len(pivoted.index)):
        for j in range(len(pivoted.columns)):
            v = pivoted.iloc[i, j]
            if not pd.isna(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=13, color='black' if 0.4 < v < 0.8 else 'white')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Triplet Accuracy', fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    ax.set_xlabel('Approach', fontsize=15)
    ax.set_ylabel('Variation', fontsize=15)

    fig.tight_layout()
    fig.savefig(plots_folder / 'shuffling_variation_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)
