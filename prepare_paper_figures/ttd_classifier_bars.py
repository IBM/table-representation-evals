"""P14: TTD classifier bars — grouped by classifier, one bar per approach per group, macro-F1."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


def create_barplot(df: pd.DataFrame, plots_folder: Path):
    classifiers = ['XGBoost', 'MLP', 'KNeighbors']
    f1_label = 'f1_macro (↑)'

    data = {}
    colors = {}
    for _, row in df.iterrows():
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

    fig, ax = plt.subplots(figsize=(10, 5))

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

        for bar, v in zip(bars, vals):
            if not np.isnan(v) and v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=6,
                        rotation=90)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(classifiers, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('macro-F1', fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc='lower right')

    fig.suptitle('Table Type Detection: macro-F1 per Classifier\n'
                 '(WDC Schema.org, frozen embeddings)', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(plots_folder / 'ttd_classifier_bars.pdf', bbox_inches='tight')
    plt.close(fig)
