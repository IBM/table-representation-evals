"""P4: Recall@k line chart — rl=100, markdown, averaged over datasets."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import config_helpers as h


def create_lineplot(df: pd.DataFrame, plots_folder: Path):
    filtered = df.copy()
    filtered = h.filter_by_row_limit(filtered, 100)
    md_mask = filtered['chart_name'].str.endswith('(md)')
    no_serial_mask = ~filtered['chart_name'].str.contains(r'\(', regex=True, na=False)
    filtered = filtered[md_mask | no_serial_mask]

    k_values = [1, 3, 5, 10]
    recall_cols = [f'Recall@{k}_mean' for k in k_values]

    # Check columns exist
    missing = [c for c in recall_cols if c not in filtered.columns]
    if missing:
        print(f"WARNING: Missing recall columns: {missing}")
        return

    approaches = sorted(filtered['chart_name'].unique())

    # Load bootstrap CIs
    ci_path = Path(__file__).parent / 'bootstrap_cis.csv'
    ci_df = pd.read_csv(ci_path) if ci_path.exists() else None

    # Build (Approach, Configuration) -> per-k CI lookup
    ci_agg = {}
    if ci_df is not None:
        ci_recall = ci_df[(ci_df['Metric'].str.startswith('Recall@')) &
                          (ci_df['Dataset'] == '__aggregate__')]
        for _, row in ci_recall.iterrows():
            ci_agg.setdefault((row['Approach'], row['Configuration']), {})[row['Metric']] = \
                (row['CI_lower'], row['CI_upper'], row['Mean'])

    # Build chart_name -> (Approach, Configuration)
    key_to_name = {}
    for _, row in filtered[['chart_name', 'Approach', 'Configuration']].drop_duplicates().iterrows():
        key_to_name.setdefault(row['chart_name'], []).append((row['Approach'], row['Configuration']))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for approach in approaches:
        app_df = filtered[filtered['chart_name'] == approach]
        color = app_df['color'].iloc[0] if len(app_df) > 0 else '#333333'

        mean_vals = app_df[recall_cols].mean()
        ax.plot(k_values, mean_vals.values, 'o-', label=approach, color=color,
                linewidth=1.8, markersize=6)

        # Add CI shaded band
        if ci_df is not None:
            lower_arr = np.empty(len(k_values))
            upper_arr = np.empty(len(k_values))
            has_ci = False
            for approach_key, config_key in key_to_name.get(approach, []):
                per_k = ci_agg.get((approach_key, config_key), {})
                if per_k:
                    for j, k in enumerate(k_values):
                        metric = f'Recall@{k}'
                        if metric in per_k:
                            lo, hi, _ = per_k[metric]
                            lower_arr[j] = lo
                            upper_arr[j] = hi
                            has_ci = True
                    break  # Use first matching config
            if has_ci:
                ax.fill_between(k_values, lower_arr, upper_arr, alpha=0.15, color=color)

    ax.set_xlabel('k', fontsize=14)
    ax.set_ylabel('Recall@k', fontsize=16)

    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_folder / 'retrieval_recall_line.pdf')
    plt.close(fig)
