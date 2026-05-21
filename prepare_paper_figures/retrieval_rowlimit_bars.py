"""P5: rl=0 vs rl=100 paired bars — MRR@10, markdown, averaged over datasets."""

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

    metric_col = 'MRR@10_mean'
    if metric_col not in filtered.columns:
        print(f"WARNING: {metric_col} not found")
        return

    # Group by approach and row_limit
    filtered['row_limit'] = filtered['Configuration'].apply(h.extract_row_limit)
    filtered = filtered[filtered['row_limit'].isin([0, 100])]

    agg = filtered.groupby(['chart_name', 'row_limit'])[metric_col].mean().reset_index()
    approaches = sorted(agg['chart_name'].unique())

    x = np.arange(len(approaches))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(approaches) * 1.2), 5))

    for rl, label, hatch in [(0, 'Schema only (rl=0)', '//'), (100, 'Schema + 100 rows', '')]:
        rl_data = agg[agg['row_limit'] == rl].set_index('chart_name')
        values = [rl_data.loc[a, metric_col] if a in rl_data.index else 0 for a in approaches]
        offset = 0 if rl == 0 else bar_width
        color = '#5a9' if rl == 100 else '#d95f02'
        bars = ax.bar(x + offset, values, bar_width, label=label, color=color, hatch=hatch)

        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=11, rotation=90)

    ax.set_ylabel('MRR@10', fontsize=16)

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(approaches, rotation=20, ha='right', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=13)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=11)

    fig.tight_layout()
    fig.savefig(plots_folder / 'retrieval_rowlimit_bars.pdf', bbox_inches='tight')
    plt.close(fig)
