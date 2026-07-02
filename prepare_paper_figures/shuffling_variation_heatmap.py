"""P6: Variation heatmap — TripletAccuracy across variations × approaches, avg over datasets."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


def create_heatmap(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()
    filtered['base_ds'], filtered['variation'] = zip(
        *filtered['dataset'].apply(h.parse_variation)
    )

    metric_col = 'TripletAccuracy_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found")
        return

    # Average over all datasets per (approach, variation)
    agg = filtered.groupby(['chart_name', 'variation'])[metric_col].mean().reset_index()

    # Pivot: rows=variation, cols=chart_name
    pivoted = agg.pivot(index='variation', columns='chart_name', values=metric_col)

    # Sort variations naturally
    def sort_key(v):
        try:
            return int(v[1:])
        except ValueError:
            return 999
    pivoted = pivoted.reindex(sorted(pivoted.index, key=sort_key))

    fig, ax = plt.subplots(figsize=(max(10, len(pivoted.columns) * 1.2),
                                   max(6, len(pivoted) * 0.5)))
    im = ax.imshow(pivoted.values, aspect='auto', cmap='RdYlGn', vmin=0.4, vmax=1.0)

    ax.set_xticks(range(len(pivoted.columns)))
    ax.set_xticklabels(pivoted.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivoted.index)))
    ax.set_yticklabels(pivoted.index, fontsize=8)

    for i in range(len(pivoted.index)):
        for j in range(len(pivoted.columns)):
            v = pivoted.iloc[i, j]
            if not pd.isna(v):
                ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                        fontsize=6, color='black' if 0.4 < v < 0.8 else 'white')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Triplet Accuracy', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel('Approach', fontsize=9)
    ax.set_ylabel('Variation', fontsize=9)

    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(plots_folder / 'shuffling_variation_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)
