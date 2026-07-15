"""P9: Row vs column perturbation scatter — accuracy on col_reorder vs row_reorder, labeled points."""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


def create_scatter(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()
    filtered['base_ds'], filtered['variation'] = zip(
        *filtered['dataset'].apply(h.parse_variation)
    )

    # hi-pos/hi-neg, row_reorder (v3) and col_reorder (v6)
    v3 = filtered[filtered['variation'] == 'v3']
    v6 = filtered[filtered['variation'] == 'v6']

    metric_col = 'TripletAccuracy_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found")
        return

    # Average over datasets
    row_scores = v3.groupby('chart_name').agg(
        row_acc=(metric_col, 'mean'),
        color=('color', 'first'),
    ).reset_index()

    col_scores = v6.groupby('chart_name').agg(
        col_acc=(metric_col, 'mean'),
    ).reset_index()

    merged = row_scores.merge(col_scores, on='chart_name', how='inner')
    if merged.empty:
        print("WARNING: No data for row-col scatter")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    for _, r in merged.iterrows():
        ax.scatter(r['row_acc'], r['col_acc'], color=r['color'],
                   s=180, edgecolors='white', linewidth=1, zorder=5)
        ax.annotate(r['chart_name'],
                    (r['row_acc'], r['col_acc']),
                    textcoords='offset points', xytext=(8, 6),
                    fontsize=13, color=r['color'],
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Diagonal line (symmetric behaviour)
    min_val = min(merged['row_acc'].min(), merged['col_acc'].min()) - 0.05
    max_val = max(merged['row_acc'].max(), merged['col_acc'].max()) + 0.05
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Row = Col')

    ax.set_xlabel('Accuracy on Row Reorder (v3)', fontsize=14)
    ax.set_ylabel('Accuracy on Col Reorder (v6)', fontsize=16)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=13)

    fig.tight_layout()
    fig.savefig(plots_folder / 'shuffling_row_col_scatter.pdf', bbox_inches='tight')
    plt.close(fig)
