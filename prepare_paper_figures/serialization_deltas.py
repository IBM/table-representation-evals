"""P13: Serialization delta bars — Δ = csv − markdown for 3 transformers × 3 tasks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


def _get_delta(df: pd.DataFrame, task_name: str, metric: str,
               row_limit: int = 100) -> pd.DataFrame:
    """Compute csv − markdown delta per approach for a task."""
    d = df[df['task'] == task_name].copy()
    d = h.filter_by_row_limit(d, row_limit)
    if metric not in d.columns:
        return pd.DataFrame()

    agg = d.groupby('chart_name')[metric].mean().reset_index()

    deltas = []
    for base in ['GritLM', 'MiniLM', 'Granite-R2']:
        md_name = f'{base} (md)'
        csv_name = f'{base} (csv)'
        if md_name in agg['chart_name'].values and csv_name in agg['chart_name'].values:
            md_val = agg[agg['chart_name'] == md_name][metric].iloc[0]
            csv_val = agg[agg['chart_name'] == csv_name][metric].iloc[0]
            deltas.append({'approach': base, 'delta': csv_val - md_val})

    return pd.DataFrame(deltas) if deltas else pd.DataFrame()


def create_plot(df: pd.DataFrame, plots_folder: Path):
    tasks = [
        ('Table Retrieval', 'table_retrieval', 'MRR@10_mean'),
        ('Table Shuffling', 'table_shuffling', 'TripletAccuracy_mean'),
        ('Table Type Detection', 'table_type_detection', 'XGBoost_f1_macro (↑)_mean'),
    ]

    colors = {'GritLM': '#1f77b4', 'MiniLM': '#ff7f0e', 'Granite-R2': '#9467bd'}

    all_deltas = []
    for task_label, task_key, metric in tasks:
        td = _get_delta(df, task_key, metric)
        if not td.empty:
            td['task'] = task_label
            all_deltas.append(td)

    if not all_deltas:
        print("WARNING: No serialization delta data")
        return

    combined = pd.concat(all_deltas, ignore_index=True)
    approaches = ['GritLM', 'MiniLM', 'Granite-R2']
    task_labels = [t[0] for t in tasks]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(task_labels))
    bar_width = 0.25

    for i, approach in enumerate(approaches):
        vals = []
        for tl in task_labels:
            row = combined[(combined['approach'] == approach) & (combined['task'] == tl)]
            vals.append(row['delta'].iloc[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * bar_width, vals, bar_width,
                      label=approach, color=colors.get(approach, '#333'))

        for bar, v in zip(bars, vals):
            y_pos = bar.get_height() + 0.002 if v >= 0 else bar.get_height() - 0.015
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{v:+.4f}', ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=11, rotation=45)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(task_labels, fontsize=13)
    ax.set_ylabel(r'$\Delta$ (csv $-$ markdown)', fontsize=16)

    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=13)

    fig.tight_layout()
    fig.savefig(plots_folder / 'serialization_deltas.pdf', bbox_inches='tight')
    plt.close(fig)
