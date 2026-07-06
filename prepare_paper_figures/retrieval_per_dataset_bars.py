"""P3: Per-dataset MRR@10 grouped bar chart for table retrieval (rl=100, markdown)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


SPIDER_DATASETS = {'spider-train', 'spider-test', 'spider-validation'}


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

    # Merge spider splits into one "spider" aggregate (weighted by query count)
    filtered['dataset'] = filtered['dataset'].apply(
        lambda d: 'spider' if d in SPIDER_DATASETS else d
    )

    def weighted_mean(group):
        w = group['total_queries_mean'].fillna(0)
        if w.sum() == 0:
            return group[metric_col].mean()
        return (group[metric_col] * w).sum() / w.sum()

    grouped = filtered.groupby(['chart_name', 'dataset'])
    records = []
    for (approach, ds), grp in grouped:
        records.append({
            'chart_name': approach,
            'dataset': ds,
            metric_col: weighted_mean(grp),
        })
    agg = pd.DataFrame(records)

    approaches = sorted(filtered['chart_name'].unique())
    datasets = sorted(filtered['dataset'].unique())
    # Put spider before tabfact
    if 'spider' in datasets:
        datasets.remove('spider')
        tabfact_idx = datasets.index('tabfact')
        datasets.insert(tabfact_idx, 'spider')

    # Wider inter-group gaps to visually separate dataset clusters
    n_approaches = len(approaches)
    bar_width = 0.70 / n_approaches
    gap_factor = 1.3
    x = np.arange(len(datasets)) * gap_factor

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 2.0), 5.2 * 0.7))

    for i, approach in enumerate(approaches):
        approach_data = agg[agg['chart_name'] == approach].set_index('dataset')
        color_row = filtered[filtered['chart_name'] == approach]
        color = color_row['color'].iloc[0] if len(color_row) > 0 else '#333333'

        values = [approach_data.loc[d, metric_col] if d in approach_data.index else 0
                  for d in datasets]
        ax.bar(x + i * bar_width, values, bar_width, label=approach, color=color)

    ax.set_ylabel('MRR@10', fontsize=18)

    ax.set_xticks(x + bar_width * (n_approaches - 1) / 2)
    ax.set_xticklabels(datasets, rotation=0, ha='center', fontsize=15)
    ax.set_ylim(0, 0.98)
    ax.tick_params(labelsize=15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=13, ncol=1)

    fig.tight_layout()
    fig.savefig(plots_folder / 'retrieval_per_dataset_bars.pdf', bbox_inches='tight')
    plt.close(fig)
