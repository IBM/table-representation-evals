"""P3: Per-dataset MRR@10 grouped bar chart for table retrieval (rl=100, markdown)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


def create_barplot(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()
    filtered = h.filter_by_row_limit(filtered, 100)
    md_mask = filtered['chart_name'].str.endswith('(md)')
    no_serial_mask = ~filtered['chart_name'].str.contains(r'\(', regex=True, na=False)
    filtered = filtered[md_mask | no_serial_mask]

    metric_col = 'MRR@10_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found")
        return

    approaches = sorted(filtered['chart_name'].unique())
    datasets = sorted(filtered['dataset'].unique())

    # Aggregate to mean per (approach, dataset)
    agg = filtered.groupby(['chart_name', 'dataset'])[metric_col].mean().reset_index()

    x = np.arange(len(datasets))
    bar_width = 0.8 / len(approaches)

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.8), 5))

    for i, approach in enumerate(approaches):
        approach_data = agg[agg['chart_name'] == approach].set_index('dataset')
        color_row = filtered[filtered['chart_name'] == approach]
        color = color_row['color'].iloc[0] if len(color_row) > 0 else '#333333'

        values = [approach_data.loc[d, metric_col] if d in approach_data.index else 0
                  for d in datasets]
        bars = ax.bar(x + i * bar_width, values, bar_width, label=approach, color=color)

        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=11, rotation=90)

    ax.set_ylabel('MRR@10', fontsize=16)
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_title('Table Retrieval MRR@10 per Dataset (rl=100, markdown)')
    ax.set_xticks(x + bar_width * (len(approaches) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=13)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=4, fontsize=11)

    fig.tight_layout()
    fig.savefig(plots_folder / 'retrieval_per_dataset_bars.pdf', bbox_inches='tight')
    plt.close(fig)
