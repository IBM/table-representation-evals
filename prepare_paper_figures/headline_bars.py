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


def _load_ci_map(ci_df: pd.DataFrame, metric: str) -> dict:
    """Build a dict mapping chart_name -> (ci_lower, ci_upper) from aggregate CI rows."""
    agg = ci_df[(ci_df['Dataset'] == '__aggregate__') & (ci_df['Metric'] == metric)]
    ci_map = {}
    for _, row in agg.iterrows():
        ci_map[(row['Approach'], row['Configuration'])] = (row['CI_lower'], row['CI_upper'])
    return ci_map


def _build_key_mapping(df: pd.DataFrame) -> dict:
    """Build chart_name -> list of (Approach, Configuration) tuples."""
    mapping = {}
    for _, row in df[['chart_name', 'Approach', 'Configuration']].drop_duplicates().iterrows():
        mapping.setdefault(row['chart_name'], []).append((row['Approach'], row['Configuration']))
    return mapping


def create_plot(df: pd.DataFrame, plots_folder: Path):
    # Load bootstrap CIs
    ci_path = Path(__file__).parent / 'bootstrap_cis.csv'
    ci_df = pd.read_csv(ci_path) if ci_path.exists() else None

    # Build chart_name -> (Approach, Configuration) for CI lookup
    key_map = _build_key_mapping(df)

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
        ('Table Shuffling\n(Triplet Accuracy, v0 variation)', s_df, 'Shuffling'),
        ('Table Type Detection\n(XGBoost macro-F1)', t_df, 'TTD'),
    ]

    # 3 bar panels + 1 narrow legend panel
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2 * 0.8935), layout='constrained',
                             gridspec_kw={'width_ratios': [3, 3, 3, 0.7], 'wspace': 0.08})

    panel_axes = axes[:3]
    legend_ax = axes[3]

    panel_labels = [
        'Table Retrieval\n(MRR@10)',
        'Table Shuffling\n(Triplet Accuracy, v0 variation)',
        'Table Type Detection\n(XGBoost macro-F1)',
    ]

    for ax, (title, panel_df, task_label), label in zip(panel_axes, panels, panel_labels):
        panel_df = panel_df.set_index('chart_name').reindex(all_approaches)
        colors = panel_df['color'].fillna('#999999').values
        has_data = panel_df['score'].notna().values
        scores = panel_df['score'].fillna(0).values

        # Build yerr for this panel
        yerr_lower = np.zeros(len(all_approaches))
        yerr_upper = np.zeros(len(all_approaches))
        if ci_df is not None:
            if task_label == 'Retrieval':
                ci_metric = 'MRR@10'
            elif task_label == 'Shuffling':
                ci_metric = 'Accuracy'
            else:
                ci_metric = None

            if ci_metric:
                for j, name in enumerate(all_approaches):
                    if not has_data[j]:
                        continue
                    for approach, config in key_map.get(name, []):
                        ci_row = ci_df[(ci_df['Approach'] == approach) &
                                       (ci_df['Configuration'] == config) &
                                       (ci_df['Metric'] == ci_metric) &
                                       (ci_df['Dataset'] == '__aggregate__')]
                        if len(ci_row) > 0:
                            mean_val = ci_row['Mean'].iloc[0]
                            lo = ci_row['CI_lower'].iloc[0]
                            hi = ci_row['CI_upper'].iloc[0]
                            yerr_lower[j] = max(0.0, mean_val - lo)
                            yerr_upper[j] = max(0.0, hi - mean_val)
                            break  # Use first matching config

        yerr = [yerr_lower, yerr_upper] if yerr_lower.any() or yerr_upper.any() else None

        bars = ax.bar(range(len(all_approaches)), scores, color=colors, edgecolor='white', linewidth=0.5,
                      width=0.90, yerr=yerr, capsize=12, error_kw={'linewidth': 2.0, 'ecolor': '#111111'})
        # Horizontal labels above bars (only for bars with data)
        for bar, has_d, score in zip(bars, has_data, scores):
            if has_d:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=15)
        ax.set_title(label, fontsize=15)
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='y', labelsize=15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    # Build legend from unique (name, color) pairs in approach order
    seen = {}
    for name in all_approaches:
        # find color from any panel that has it
        for _, panel_df, _ in panels:
            row = panel_df[panel_df['chart_name'] == name]
            if not row.empty:
                seen[name] = row['color'].iloc[0]
                break

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=name)
        for name, color in seen.items()
    ]
    legend_ax.legend(handles=legend_handles, loc='center left', fontsize=14,
                     frameon=True, framealpha=0.9, edgecolor='#cccccc',
                     title='Approach', title_fontsize=14)
    legend_ax.axis('off')

    fig.savefig(plots_folder / 'headline_bars.pdf', bbox_inches='tight')
    plt.close(fig)
