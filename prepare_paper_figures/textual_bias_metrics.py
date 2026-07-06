"""Aggregate metrics charts for textual bias analysis.

Three panels:
1. TextualBias_pearson per approach across v0/v1/v2 (magnitude sweep)
2. Silhouette score per approach across v0/v1/v2
3. mean_d_pos vs mean_d_neg comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from pathlib import Path


def create_plot(df: pd.DataFrame, plots_folder: Path):
    import config_helpers as h
    shuf = df[df['task'] == 'table_shuffling'].copy()
    shuf['base_ds'], shuf['variation'] = zip(*shuf['dataset'].apply(h.parse_variation))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- Panel 1: TextualBias_pearson across 3 magnitude variations (v0/v1/v2) ----
    ax = axes[0, 0]
    variations = ['v0', 'v1', 'v2']
    var_labels = ['hi-pos / hi-neg\n(v0)', 'lo-pos / hi-neg\n(v1)', 'hi-pos / lo-neg\n(v2)']
    approaches = sorted(shuf['chart_name'].unique())
    colors = [shuf[shuf['chart_name'] == a]['color'].iloc[0] for a in approaches]

    metric = 'TextualBias_pearson_mean'
    x = np.arange(len(approaches))
    width = 0.25

    for i, (var, label) in enumerate(zip(variations, var_labels)):
        var_df = shuf[shuf['variation'] == var]
        scores = []
        for a in approaches:
            a_df = var_df[var_df['chart_name'] == a]
            # Average over datasets
            scores.append(a_df[metric].mean() if not a_df[metric].isna().all() else 0)
        bars = ax.bar(x + i * width, scores, width, label=label, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(approaches, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel("TextualBias Pearson's r", fontsize=12)
    ax.set_title('Textual Bias Across Perturbation Magnitudes\n(both perm., default window, avg over datasets)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    # ---- Panel 2: TextualBias_pearson heatmap (all 15 variations × approach) ----
    ax = axes[0, 1]
    all_vars = [f'v{i}' for i in range(15)]
    var_data = {}
    for a in approaches:
        row = []
        for v in all_vars:
            vdf = shuf[(shuf['chart_name'] == a) & (shuf['variation'] == v)]
            row.append(vdf[metric].mean() if not vdf[metric].isna().all() else np.nan)
        var_data[a] = row

    heatmap_data = np.array([var_data[a] for a in approaches])
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)

    ax.set_xticks(range(len(all_vars)))
    ax.set_xticklabels(all_vars, fontsize=8)
    ax.set_yticks(range(len(approaches)))
    ax.set_yticklabels(approaches, fontsize=10)
    ax.set_title("TextualBias Pearson's r — All 15 Variations\n(avg over datasets)", fontsize=12)

    # Annotate cells
    for i in range(len(approaches)):
        for j in range(len(all_vars)):
            v = heatmap_data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                       color='white' if abs(v) > 0.3 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8)

    # ---- Panel 3: Silhouette score bar chart (v0, per dataset) ----
    ax = axes[1, 0]
    v0_df = shuf[shuf['variation'] == 'v0']
    sil_metric = 'Triplet Silhouette Score_mean'

    pivot = v0_df.pivot_table(values=sil_metric, index='chart_name', columns='base_ds', aggfunc='mean')
    # Reorder: lexical datasets first, ECB last
    desired = ['fetaqa', 'tabfact', 'ottqa', 'spider-train', 'ckan_subset', 'ecb']
    cols = [c for c in desired if c in pivot.columns]
    pivot = pivot[cols]

    x = np.arange(len(pivot.index))
    n_cols = len(cols)
    width = 0.8 / n_cols

    for j, col in enumerate(cols):
        vals = [pivot.loc[a, col] if a in pivot.index and not pd.isna(pivot.loc[a, col]) else 0 for a in approaches]
        bars = ax.bar(x + j * width, vals, width, label=col, alpha=0.85)

    ax.set_xticks(x + width * (n_cols - 1) / 2)
    ax.set_xticklabels(approaches, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Triplet Silhouette Score per Dataset (v0)\n(>0 = pos closer than neg; <0 = inversion)', fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # ---- Panel 4: mean_d_pos vs mean_d_neg scatter (v0, per approach) ----
    ax = axes[1, 1]
    dpos_metric = 'mean_d_pos_mean'
    dneg_metric = 'mean_d_neg_mean'

    for a in approaches:
        a_df = v0_df[v0_df['chart_name'] == a]
        dpos = a_df[dpos_metric].mean()
        dneg = a_df[dneg_metric].mean()
        color = a_df['color'].iloc[0] if len(a_df) > 0 else '#000'
        ax.scatter(dpos, dneg, s=250, c=color, edgecolors='white', linewidth=1.5, zorder=5)
        ax.annotate(a, (dpos, dneg), textcoords='offset points', xytext=(8, 6),
                   fontsize=11, color=color,
                   path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Diagonal (d_pos == d_neg)
    max_val = 0.08
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='d_pos = d_neg')
    ax.set_xlabel('mean d_pos (cosine distance to permuted positive)', fontsize=11)
    ax.set_ylabel('mean d_neg (cosine distance to value-shuffled negative)', fontsize=11)
    ax.set_title('Embedding Distance to Positive vs Negative (v0)\n(above diagonal = inversion; below = structural)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.002, max_val)
    ax.set_ylim(-0.002, max_val)

    fig.tight_layout()
    out_path = plots_folder / 'textual_bias_metrics.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    # Standalone run
    from benchmark_src.results_processing import create_plots
    results_folder = Path("results_complete/results")
    plots_folder = Path("./prepare_paper_figures/main_table_experiments")
    plots_folder.mkdir(exist_ok=True, parents=True)

    df = create_plots.gather_results_and_metrics(results_folder=results_folder)
    df = df[df["Approach"] != "tfidf"]

    # Minimal color/name mapping (same as main.py)
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

    df["chart_name"] = df.apply(
        lambda row: name_mapping.get((row["Approach"], row["Configuration"]), None), axis=1
    )
    df["color"] = df.apply(
        lambda row: color_mapping.get((row["Approach"], row["Configuration"]), "#000000"), axis=1
    )
    df = df[df["chart_name"].notna()]

    # Also fix dataset column — strip variation for base_ds parsing
    # (the existing parse_variation from config_helpers handles @@)

    create_plot(df, plots_folder)
