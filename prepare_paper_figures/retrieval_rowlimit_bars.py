"""P5: rl=0 vs rl=100 paired bars — MRR@10, markdown, averaged over datasets."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
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
    bar_width = 0.40

    # Load bootstrap CIs
    ci_path = Path(__file__).parent / 'bootstrap_cis.csv'
    ci_df = pd.read_csv(ci_path) if ci_path.exists() else None

    # Build (Approach, Configuration) -> CI for MRR@10 aggregate
    ci_agg = {}
    if ci_df is not None:
        ci_mrr = ci_df[(ci_df['Metric'] == 'MRR@10') & (ci_df['Dataset'] == '__aggregate__')]
        for _, row in ci_mrr.iterrows():
            ci_agg[(row['Approach'], row['Configuration'])] = (row['CI_lower'], row['CI_upper'], row['Mean'])

    fig, ax = plt.subplots(figsize=(max(8, len(approaches) * 1.2), 3.8 * 0.9))

    # Each approach gets its own color; hatch differentiates schema-only vs schema+100
    all_values = []
    for i, approach in enumerate(approaches):
        approach_df = filtered[filtered['chart_name'] == approach]
        color = approach_df['color'].iloc[0] if len(approach_df) > 0 else '#333333'

        for rl, offset, hatch in [(0, 0, '//'), (100, bar_width, '')]:
            rl_data = agg[(agg['chart_name'] == approach) & (agg['row_limit'] == rl)]
            if len(rl_data) == 0:
                continue
            value = rl_data[metric_col].iloc[0]
            all_values.append(value)

            # Look up CI for this approach+config
            yerr = None
            sub = approach_df[approach_df['Configuration'].apply(h.extract_row_limit) == rl]
            if len(sub) > 0:
                config = sub['Configuration'].iloc[0]
                ci_tuple = ci_agg.get((sub['Approach'].iloc[0], config))
                if ci_tuple is not None:
                    lo, hi, _ = ci_tuple
                    yerr = [[max(0.0, value - lo)], [max(0.0, hi - value)]]

            bar = ax.bar(x[i] + offset, value, bar_width,
                         color=color, hatch=hatch,
                         edgecolor='white', linewidth=0.5,
                         yerr=yerr, capsize=8,
                         error_kw={'linewidth': 2.0, 'ecolor': '#111111'})
            if value > 0:
                ax.bar_label(bar, fmt='%.2f', fontsize=15, padding=3)

    ax.set_ylabel('MRR@10', fontsize=18)

    ax.set_xticks(x + bar_width / 2)
    labels = [a.replace(' (md)', '\n(md)') for a in approaches]
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # Legend inside plot (top-right): gray exemplars for the two conditions
    legend_elements = [
        Patch(facecolor='lightgray', hatch='//', edgecolor='white', linewidth=0.5,
              label='Schema only (rl=0)'),
        Patch(facecolor='lightgray', edgecolor='white', linewidth=0.5,
              label='Schema + 100 rows'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)

    fig.tight_layout()
    fig.savefig(plots_folder / 'retrieval_rowlimit_bars.pdf', bbox_inches='tight')
    plt.close(fig)
