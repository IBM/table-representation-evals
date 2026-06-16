"""P7: ECB spotlight bars — accuracy per variation for ECB only."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


def create_barplot(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()
    filtered['base_ds'], filtered['variation'] = zip(
        *filtered['dataset'].apply(h.parse_variation)
    )

    # ECB only
    filtered = filtered[filtered['base_ds'] == 'ecb']

    metric_col = 'TripletAccuracy_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found")
        return

    agg = filtered.groupby(['chart_name', 'variation'])[metric_col].mean().reset_index()

    variations = sorted(agg['variation'].unique(),
                        key=lambda v: int(v[1:]) if v[1:].isdigit() else 999)
    approaches = sorted(agg['chart_name'].unique())

    x = np.arange(len(variations))
    bar_width = 0.8 / len(approaches)

    fig, ax = plt.subplots(figsize=(max(12, len(variations) * 1.5), 5))

    for i, approach in enumerate(approaches):
        ad = agg[agg['chart_name'] == approach].set_index('variation')
        color_row = filtered[filtered['chart_name'] == approach]
        color = color_row['color'].iloc[0] if len(color_row) > 0 else '#333'

        values = [ad.loc[v, metric_col] if v in ad.index else 0 for v in variations]
        ax.bar(x + i * bar_width, values, bar_width, label=approach, color=color)

    ax.set_xticks(x + bar_width * (len(approaches) - 1) / 2)
    ax.set_xticklabels(variations, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Triplet Accuracy')
    ax.set_title('ECB Spotlight: Accuracy Across All Perturbation Variations')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_folder / 'shuffling_ecb_bars.pdf', bbox_inches='tight')
    plt.close(fig)
