"""P3: Per-dataset MRR@10/MAP@10 grouped bar chart for table retrieval (markdown, each approach at its default row limit)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import config_helpers as h


def create_barplot(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()
    md_mask = filtered['chart_name'].str.endswith('(md)')
    no_serial_mask = ~filtered['chart_name'].str.contains(r'\(', regex=True, na=False)
    filtered = filtered[md_mask | no_serial_mask]

    for metric_col in ('MRR@10', 'MAP@10'):
        if metric_col not in filtered.columns:
            print(f"WARNING: {metric_col} not found")
            continue

        approaches = sorted(filtered['chart_name'].unique(), key=h.chart_name_sort_key)
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
                            f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

        ax.set_ylabel(metric_col)
        ax.set_xlabel('Dataset')
        ax.set_title(f'Table Retrieval {metric_col} per Dataset (markdown)')
        ax.set_xticks(x + bar_width * (len(approaches) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=4, fontsize=8)

        fig.tight_layout()
        metric_slug = metric_col.replace('@', '').lower()
        fig.savefig(plots_folder / f'retrieval_per_dataset_bars_{metric_slug}.pdf', bbox_inches='tight')
        plt.close(fig)
