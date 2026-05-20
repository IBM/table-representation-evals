"""P4: Recall@k line chart — rl=100, markdown, averaged over datasets."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


def create_lineplot(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()
    filtered = h.filter_by_row_limit(filtered, 100)
    md_mask = filtered['chart_name'].str.endswith('(md)')
    no_serial_mask = ~filtered['chart_name'].str.contains(r'\(', regex=True, na=False)
    filtered = filtered[md_mask | no_serial_mask]

    k_values = [1, 3, 5, 10]
    recall_cols = [f'Recall@{k}_mean' for k in k_values]

    # Check columns exist
    missing = [c for c in recall_cols if c not in filtered.columns]
    if missing:
        print(f"WARNING: Missing recall columns: {missing}")
        return

    approaches = sorted(filtered['chart_name'].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for approach in approaches:
        app_df = filtered[filtered['chart_name'] == approach]
        color = app_df['color'].iloc[0] if len(app_df) > 0 else '#333333'

        mean_vals = app_df[recall_cols].mean()
        ax.plot(k_values, mean_vals.values, 'o-', label=approach, color=color,
                linewidth=1.8, markersize=6)

    ax.set_xlabel('k', fontsize=14)
    ax.set_ylabel('Recall@k', fontsize=16)
    ax.set_title('Table Retrieval: Recall@k (rl=100, markdown, avg over 7 datasets)')
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_folder / 'retrieval_recall_line.pdf')
    plt.close(fig)
