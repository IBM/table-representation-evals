"""P5: rl=0 vs rl=100 paired bars — MRR@10/MAP@10, markdown, averaged over datasets."""

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

    # Group by approach and row_limit
    filtered['row_limit'] = filtered['Configuration'].apply(h.extract_row_limit)
    filtered = filtered[filtered['row_limit'].isin([0, 100])]

    if filtered.empty:
        print("WARNING: No table_row_limit=0/100-labeled retrieval runs found (rl ablation not run yet), skipping rowlimit bars")
        return

    for metric_col in ('MRR@10', 'MAP@10'):
        if metric_col not in filtered.columns:
            print(f"WARNING: {metric_col} not found")
            continue

        agg = filtered.groupby(['chart_name', 'row_limit'])[metric_col].mean().reset_index()
        approaches = sorted(agg['chart_name'].unique(), key=h.chart_name_sort_key)

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
                            f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

        ax.set_ylabel(metric_col)
        ax.set_title('Table Retrieval: Schema-only vs Schema+Content (markdown, avg over datasets)')
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(approaches, rotation=20, ha='right')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)

        fig.tight_layout()
        metric_slug = metric_col.replace('@', '').lower()
        fig.savefig(plots_folder / f'retrieval_rowlimit_bars_{metric_slug}.pdf', bbox_inches='tight')
        plt.close(fig)
