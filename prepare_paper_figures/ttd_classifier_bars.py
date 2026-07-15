"""P14: TTD classifier bars — grouped by classifier, one bar per approach per group, macro-F1."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


def create_barplot(df: pd.DataFrame, plots_folder: Path):
    # Exclude CSV variants — keep markdown + approaches without serialization
    md_mask = df['chart_name'].str.endswith('(md)')
    no_serial_mask = ~df['chart_name'].str.contains(r'\(', regex=True, na=False)
    filtered = df[md_mask | no_serial_mask].copy()

    classifiers = ['KNeighbors', 'MLP', 'XGBoost']
    f1_label = 'f1_macro (↑)'

    data = {}
    colors = {}
    for _, row in filtered.iterrows():
        name = row['chart_name']
        colors[name] = row.get('color', '#333333')
        data[name] = {}
        for cls in classifiers:
            col = f'{cls}_{f1_label}_mean'
            data[name][cls] = row.get(col, np.nan)

    if not data:
        print("WARNING: No TTD classifier data found")
        return

    approaches = sorted(data.keys(),
                        key=lambda a: max(data[a].values()),
                        reverse=True)
    n_approaches = len(approaches)
    n_classifiers = len(classifiers)

    fig, ax = plt.subplots(figsize=(10.34, 5.2 * 0.776))

    group_width = 0.75
    bar_width = group_width / n_approaches
    x_centers = np.arange(n_classifiers)

    for i, approach in enumerate(approaches):
        vals = [data[approach].get(cls, np.nan) for cls in classifiers]
        offset = (i - (n_approaches - 1) / 2) * bar_width
        xs = x_centers + offset
        color = colors.get(approach, '#333333')

        bars = ax.bar(xs, vals, bar_width, color=color, edgecolor='white',
                      linewidth=0.5, label=approach)
        # Vertical labels above bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=14, rotation=90)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(classifiers, fontsize=17, rotation=0, ha='center')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('macro-F1', fontsize=21)
    ax.tick_params(labelsize=17)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # Legend below the figure, outside
    ax.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=len(approaches), frameon=True, framealpha=0.9, edgecolor='#cccccc')

    fig.tight_layout()
    fig.savefig(plots_folder / 'ttd_classifier_bars.pdf', bbox_inches='tight')
    plt.close(fig)
