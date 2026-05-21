"""P1: Three-panel headline grouped bar chart — one panel per task, shared approach ordering."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


def _compute_retrieval_scores(df: pd.DataFrame) -> pd.DataFrame:
    """MRR@10, rl=100, markdown for transformers, all for BoW/HyTrel, avg over datasets."""
    d = df[df['task'] == 'table_retrieval'].copy()
    d = h.filter_by_row_limit(d, 100)
    md_mask = d['chart_name'].str.endswith('(md)')
    no_serial_mask = ~d['chart_name'].str.contains(r'\(', regex=True, na=False)
    d = d[md_mask | no_serial_mask]
    metric = 'MRR@10_mean'
    if metric not in d.columns:
        return pd.DataFrame()
    return d.groupby('chart_name').agg(
        score=(metric, 'mean'),
        color=('color', 'first'),
    ).reset_index()


def _compute_shuffling_scores(df: pd.DataFrame) -> pd.DataFrame:
    """TripletAccuracy, v0, avg over datasets."""
    d = df[df['task'] == 'table_shuffling'].copy()
    md_mask = d['chart_name'].str.endswith('(md)')
    no_serial_mask = ~d['chart_name'].str.contains(r'\(', regex=True, na=False)
    d = d[md_mask | no_serial_mask]
    d['base_ds'], d['variation'] = zip(*d['dataset'].apply(h.parse_variation))
    d = d[d['variation'] == 'v0']
    metric = 'TripletAccuracy_mean'
    if metric not in d.columns:
        return pd.DataFrame()
    return d.groupby('chart_name').agg(
        score=(metric, 'mean'),
        color=('color', 'first'),
    ).reset_index()


def _compute_ttd_scores(df: pd.DataFrame) -> pd.DataFrame:
    """XGBoost macro-F1, frozen embeddings."""
    d = df[df['task'] == 'table_type_detection'].copy()
    md_mask = d['chart_name'].str.endswith('(md)')
    no_serial_mask = ~d['chart_name'].str.contains(r'\(', regex=True, na=False)
    d = d[md_mask | no_serial_mask]
    metric = 'XGBoost_f1_macro (↑)_mean'
    if metric not in d.columns:
        return pd.DataFrame()
    return d.groupby('chart_name').agg(
        score=(metric, 'mean'),
        color=('color', 'first'),
    ).reset_index()


def create_plot(df: pd.DataFrame, plots_folder: Path):
    r_df = _compute_retrieval_scores(df)
    s_df = _compute_shuffling_scores(df)
    t_df = _compute_ttd_scores(df)

    if r_df.empty or s_df.empty or t_df.empty:
        print("WARNING: Missing data for headline bars")
        return

    # Unified approach list (union of all three, ordered by retrieval score descending)
    r_order = r_df.sort_values('score', ascending=False)['chart_name'].tolist()
    all_approaches = r_order + [a for a in s_df['chart_name'] if a not in r_order] + \
                     [a for a in t_df['chart_name'] if a not in r_order and a not in s_df['chart_name']]

    panels = [
        ('Table Retrieval\n(MRR@10)', r_df, 'Retrieval'),
        ('Table Shuffling\n(Triplet Accuracy)', s_df, 'Shuffling'),
        ('Table Type Detection\n(XGBoost macro-F1)', t_df, 'TTD'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes = axes.flatten()

    for ax, (title, panel_df, _) in zip(axes, panels):
        panel_df = panel_df.set_index('chart_name').reindex(all_approaches)
        colors = panel_df['color'].fillna('#999999').values
        has_data = panel_df['score'].notna().values
        scores = panel_df['score'].fillna(0).values

        bars = ax.bar(range(len(all_approaches)), scores, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(all_approaches)))
        ax.set_xticklabels(all_approaches, rotation=45, ha='right', fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='y', labelsize=13)

        for bar, v, present in zip(bars, scores, has_data):
            if present:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=11, rotation=90)

    fig.suptitle('Table-Level Embedding Quality Across Three Diagnostic Tasks', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(plots_folder / 'headline_bars.pdf', bbox_inches='tight')
    plt.close(fig)
