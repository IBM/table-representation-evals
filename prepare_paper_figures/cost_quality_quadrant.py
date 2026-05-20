"""P12: Cost vs quality quadrant chart — pooled across tasks, one point per approach."""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from pathlib import Path


def _get_quality(df: pd.DataFrame, task: str, metric: str, normalize: bool = False) -> pd.DataFrame:
    """Extract mean quality score per chart_name for a task."""
    d = df[df['task'] == task].copy()
    if metric not in d.columns:
        return pd.DataFrame()
    agg = d.groupby('chart_name').agg(
        quality=(metric, 'mean'),
        color=('color', 'first'),
    ).reset_index()
    if normalize and len(agg) > 0:
        max_val = agg['quality'].max()
        if max_val > 0:
            agg['quality'] = agg['quality'] / max_val
    return agg


def create_plot(df: pd.DataFrame, plots_folder: Path):
    import config_helpers as h

    # Retrieve quality data per task
    # Retrieval: MRR@10, rl=100, md for transformers, all for BoW/HyTrel
    r_raw = df[df['task'] == 'table_retrieval'].copy()
    r_raw = h.filter_by_row_limit(r_raw, 100)
    md_mask = r_raw['chart_name'].str.endswith('(md)')
    no_serial_mask = ~r_raw['chart_name'].str.contains(r'\(', regex=True, na=False)
    r_raw = r_raw[md_mask | no_serial_mask]
    r_df = _get_quality(r_raw, 'table_retrieval', 'MRR@10_mean', normalize=True)
    s = df[df['task'] == 'table_shuffling'].copy()
    s['_var'] = s['dataset'].apply(lambda x: x.rsplit('@@', 1)[-1] if '@@' in x else '')
    s = s[s['_var'] == 'v0']
    s_df = _get_quality(s, 'table_shuffling', 'TripletAccuracy_mean', normalize=True)

    # TTD: XGBoost accuracy
    t_df = _get_quality(df[df['task'] == 'table_type_detection'],
                        'table_type_detection', 'XGBoost_f1_macro (↑)_mean', normalize=True)

    # Compute mean quality across tasks
    all_quality = pd.concat([r_df, s_df, t_df], ignore_index=True)
    if all_quality.empty:
        print("WARNING: No quality data for quadrant chart")
        return

    quality = all_quality.groupby('chart_name').agg(
        avg_quality=('quality', 'mean'),
        color=('color', 'first'),
    ).reset_index()

    # Time: mean execution_time (s) per approach, from retrieval (rl=100, all canonical)
    time_col = 'execution_time (s)_mean'
    r_time = h.filter_by_row_limit(
        df[df['task'] == 'table_retrieval'], 100)
    md_mask = r_time['chart_name'].str.endswith('(md)')
    no_serial_mask = ~r_time['chart_name'].str.contains(r'\(', regex=True, na=False)
    r_time = r_time[md_mask | no_serial_mask]
    if time_col not in r_time.columns:
        print(f"WARNING: {time_col} not found")
        return

    time_df = r_time.groupby('chart_name')[time_col].mean().reset_index()
    time_df.columns = ['chart_name', 'time']

    merged = quality.merge(time_df, on='chart_name', how='inner')
    if merged.empty:
        print("WARNING: No merged data for quadrant chart")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    mid_x = merged['time'].median()
    mid_y = merged['avg_quality'].median()

    ax.axhline(mid_y, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(mid_x, color='gray', linestyle='--', alpha=0.4)

    for _, row in merged.iterrows():
        ax.scatter(row['time'], row['avg_quality'], color=row['color'],
                   s=120, edgecolors='white', linewidth=0.8, zorder=5)
        ax.annotate(row['chart_name'],
                    (row['time'], row['avg_quality']),
                    textcoords='offset points', xytext=(8, 6),
                    fontsize=8, color=row['color'],
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_xlabel('Embedding Time per Table (s)')
    ax.set_ylabel('Average Quality (pooled across tasks, normalized)')
    ax.set_title('Cost vs Quality: Embedding Time Against Pooled Task Performance')

    fig.tight_layout()
    fig.savefig(plots_folder / 'cost_quality_quadrant.pdf', bbox_inches='tight')
    plt.close(fig)
