"""Silhouette score bar chart — averaged across all datasets (v0, markdown)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def create_plot(df: pd.DataFrame, plots_folder: Path):
    import config_helpers as h

    shuf = df[df['task'] == 'table_shuffling'].copy()
    shuf['base_ds'], shuf['variation'] = zip(*shuf['dataset'].apply(h.parse_variation))

    v0 = shuf[shuf['variation'] == 'v0']
    # Keep only markdown serialization (exclude CSV) and row_limit=100
    v0 = v0[
        ~v0['Configuration'].str.contains('csv', na=False) &
        v0['Configuration'].str.contains('table_row_limit=100', na=False)
    ]
    metric = 'Triplet Silhouette Score_mean'

    # Average over datasets
    avg = v0.groupby('chart_name').agg(
        sil_mean=(metric, 'mean'),
        sil_std=(metric, 'std'),
        sil_count=(metric, 'count'),
        color=('color', 'first'),
    ).reset_index().dropna(subset=['sil_mean'])

    # Load bootstrap CIs for Silhouette (per-dataset, v0 only)
    ci_path = Path(__file__).parent / 'bootstrap_cis.csv'
    ci_lower_map = {}
    ci_upper_map = {}
    if ci_path.exists():
        ci_df = pd.read_csv(ci_path)
        ci_sil = ci_df[(ci_df['Metric'] == 'Silhouette') &
                       (ci_df['Dataset'].str.endswith('@@v0'))]
        # Build approach+config -> chart_name mapping from v0
        key_to_name = {}
        for _, row in v0[['chart_name', 'Approach', 'Configuration']].drop_duplicates().iterrows():
            key_to_name[(row['Approach'], row['Configuration'])] = row['chart_name']
        for _, row in ci_sil.iterrows():
            name = key_to_name.get((row['Approach'], row['Configuration']))
            if name:
                ci_lower_map.setdefault(name, []).append(row['CI_lower'])
                ci_upper_map.setdefault(name, []).append(row['CI_upper'])

    # Sort by silhouette first
    avg = avg.sort_values('sil_mean', ascending=True)

    # Compute pooled CI error bars from per-dataset bootstrap CIs
    yerr_lower = np.zeros(len(avg))
    yerr_upper = np.zeros(len(avg))
    for i, (_, row) in enumerate(avg.iterrows()):
        name = row['chart_name']
        if name in ci_lower_map and len(ci_lower_map[name]) > 0:
            lowers = np.array(ci_lower_map[name])
            uppers = np.array(ci_upper_map[name])
            # Pooled CI: average of per-dataset half-widths via sqrt(sum(hw²))/n
            half_widths = (uppers - lowers) / 2
            pooled_hw = np.sqrt(np.sum(half_widths ** 2)) / len(half_widths)
            yerr_lower[i] = pooled_hw
            yerr_upper[i] = pooled_hw
        else:
            # Fallback to SEM
            sem = row['sil_std'] / np.sqrt(row['sil_count']) if row['sil_count'] > 0 else 0
            yerr_lower[i] = sem
            yerr_upper[i] = sem

    yerr = [yerr_lower, yerr_upper]

    fig, ax = plt.subplots(figsize=(10, 4.0))

    bars = ax.bar(
        avg['chart_name'], avg['sil_mean'],
        yerr=yerr,
        color=avg['color'], edgecolor='white', linewidth=0.5,
        capsize=12, width=0.6,
        error_kw={'linewidth': 2.0, 'ecolor': '#111111'},
    )
    # Horizontal labels above bars
    for bar, val in zip(bars, avg['sil_mean']):
        y_pos = bar.get_height() + 0.02 if bar.get_height() >= 0 else bar.get_height() - 0.06
        va = 'bottom' if bar.get_height() >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'{val:.2f}', ha='center', va=va, fontsize=18)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Silhouette Score', fontsize=22)
    ax.set_xticks(range(len(avg)))
    ax.set_xticklabels(avg['chart_name'], rotation=0, ha='center', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(-1.0, 1.0)

    fig.tight_layout()

    out_path = plots_folder / 'shuffling_silhouette_bars.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    from benchmark_src.results_processing import create_plots as cp

    results_folder = Path("results_complete/results")
    plots_folder = Path("./prepare_paper_figures/main_table_experiments")
    plots_folder.mkdir(exist_ok=True, parents=True)

    df = cp.gather_results_and_metrics(results_folder=results_folder)
    df = df[df["Approach"] != "tfidf"]

    name_mapping = {
        ("GritLM", "embedding_model=GritLM_GritLM-7B,table_row_limit=100,table_serialization_format=markdown"): "GritLM (md)",
        ("sentence_transformer", "embedding_model=all-MiniLM-L6-v2,table_row_limit=100,table_serialization_format=markdown"): "MiniLM (md)",
        ("sentence_transformer", "embedding_model=ibm-granite_granite-embedding-english-r2,table_row_limit=100,table_serialization_format=markdown"): "Granite-R2 (md)",
        ("hytrel", "table_row_limit=100"): "HyTrel",
        ("hashing", "n_features=32768,table_row_limit=100"): "Hashing",
    }
    color_mapping = {
        ("GritLM", "embedding_model=GritLM_GritLM-7B,table_row_limit=100,table_serialization_format=markdown"): "#1f77b4",
        ("sentence_transformer", "embedding_model=all-MiniLM-L6-v2,table_row_limit=100,table_serialization_format=markdown"): "#ff7f0e",
        ("sentence_transformer", "embedding_model=ibm-granite_granite-embedding-english-r2,table_row_limit=100,table_serialization_format=markdown"): "#9467bd",
        ("hytrel", "table_row_limit=100"): "#17becf",
        ("hashing", "n_features=32768,table_row_limit=100"): "#d62728",
    }

    df["chart_name"] = df.apply(lambda row: name_mapping.get((row["Approach"], row["Configuration"]), None), axis=1)
    df["color"] = df.apply(lambda row: color_mapping.get((row["Approach"], row["Configuration"]), "#000000"), axis=1)
    df = df[df["chart_name"].notna()]

    create_plot(df, plots_folder)
