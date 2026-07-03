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
    metric = 'Triplet Silhouette Score_mean'

    # Average over datasets
    avg = v0.groupby('chart_name').agg(
        sil_mean=(metric, 'mean'),
        sil_std=(metric, 'std'),
        color=('color', 'first'),
    ).reset_index().dropna(subset=['sil_mean'])

    # Sort by silhouette
    avg = avg.sort_values('sil_mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.0))

    bars = ax.bar(
        avg['chart_name'], avg['sil_mean'],
        yerr=avg['sil_std'],
        color=avg['color'], edgecolor='white', linewidth=0.5,
        capsize=4, width=0.6,
    )

    # Annotate bars — match headline_bars style: rotated 90°, above bar
    for bar, val in zip(bars, avg['sil_mean']):
        offset = 0.02 if val >= 0 else -0.08
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, rotation=90)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Silhouette Score', fontsize=16)
    ax.set_xticks(range(len(avg)))
    ax.set_xticklabels(avg['chart_name'], rotation=30, ha='right', fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
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
