"""Scatter: textual change (delta) vs embedding distance (d) per triplet, faceted by approach.

Shows the two failure modes:
- BoW (Hashing/TF-IDF): all points at y≈0 — permutation-invariant, can't distinguish anything
- Transformers: positive slope — embedding distance tracks surface text change
- HyTrel: pos near y≈0, neg slightly elevated — structural signal only

Reads raw full_results.json (not aggregated CSV) for per-triplet delta and d values.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ---- config ----
RESULTS_BASE = Path("results_complete/results/table_paper_experiments")
DATASET = "fetaqa@@v0"  # canonical lexical dataset, 500 triplets
SERIALIZATION = "markdown"  # for transformers

# Approach -> (result_dir, config_dir)
APPROACH_CONFIGS = [
    ("GritLM (md)",       "GritLM",               f"embedding_model=GritLM_GritLM-7B,table_row_limit=100,table_serialization_format={SERIALIZATION}"),
    ("MiniLM (md)",       "sentence_transformer", f"embedding_model=all-MiniLM-L6-v2,table_row_limit=100,table_serialization_format={SERIALIZATION}"),
    ("Granite-R2 (md)",   "sentence_transformer", f"embedding_model=ibm-granite_granite-embedding-english-r2,table_row_limit=100,table_serialization_format={SERIALIZATION}"),
    ("HyTrel",            "hytrel",               "table_row_limit=100"),
    ("Hashing",           "hashing",              "n_features=32768,table_row_limit=100"),
    ("TF-IDF",            "tfidf",                "n_features=32768,table_row_limit=100"),
]

COLORS = {
    "GritLM (md)":     "#1f77b4",
    "MiniLM (md)":     "#ff7f0e",
    "Granite-R2 (md)": "#9467bd",
    "HyTrel":          "#17becf",
    "Hashing":         "#d62728",
    "TF-IDF":          "#2ca02c",
}


def load_triplets(result_dir: str, config_dir: str) -> list[dict]:
    path = RESULTS_BASE / result_dir / config_dir / "table_shuffling" / DATASET / "full_results.json"
    if not path.exists():
        print(f"WARNING: {path} not found")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("triplets", [])


def create_plot(plots_folder: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (chart_name, result_dir, config_dir) in zip(axes, APPROACH_CONFIGS):
        triplets = load_triplets(result_dir, config_dir)
        if not triplets:
            ax.set_title(f"{chart_name}\n(no data)")
            continue

        delta_pos = np.array([t["delta_pos"] for t in triplets])
        delta_neg = np.array([t["delta_neg"] for t in triplets])
        d_pos = np.array([t["d_pos"] for t in triplets])
        d_neg = np.array([t["d_neg"] for t in triplets])

        color = COLORS.get(chart_name, "#000000")

        # Positive triplets (permuted table)
        ax.scatter(delta_pos, d_pos, alpha=0.35, s=12, c=color, marker='o', label='Positive (permuted)')

        # Negative triplets (value-shuffled)
        ax.scatter(delta_neg, d_neg, alpha=0.35, s=12, c=color, marker='x', label='Negative (value-shuffled)')

        # Regression lines
        for deltas, ds, ls, marker in [
            (delta_pos, d_pos, '--', 'o'),
            (delta_neg, d_neg, ':', 'x'),
        ]:
            mask = ~np.isnan(deltas) & ~np.isnan(ds)
            if mask.sum() > 2:
                coef = np.polyfit(deltas[mask], ds[mask], 1)
                x_line = np.linspace(deltas[mask].min(), deltas[mask].max(), 50)
                ax.plot(x_line, np.polyval(coef, x_line), ls, color=color, alpha=0.8, linewidth=1.5)

        # Compute TextualBias_pearson
        all_deltas = np.concatenate([delta_pos, delta_neg])
        all_ds = np.concatenate([d_pos, d_neg])
        mask = ~np.isnan(all_deltas) & ~np.isnan(all_ds)
        if mask.sum() > 2:
            pearson = np.corrcoef(all_deltas[mask], all_ds[mask])[0, 1]
        else:
            pearson = 0.0

        acc = (d_pos < d_neg).mean()

        ax.set_title(f"{chart_name}\nacc={acc:.3f}  r={pearson:.3f}", fontsize=12, color=color)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        if chart_name == "GritLM (md)":
            ax.legend(fontsize=8, loc='upper left')

    # Shared axis labels
    fig.text(0.5, 0.02, 'Normalized Levenshtein distance (textual change δ)', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Cosine embedding distance d', va='center', rotation='vertical', fontsize=14)

    fig.suptitle(f'Textual Bias: Embedding Distance vs Surface Text Change\n{DATASET}, v0 (canonical), {SERIALIZATION} serialization',
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.94])

    out_path = plots_folder / 'textual_bias_scatter.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plots_folder = Path("./prepare_paper_figures/main_table_experiments")
    plots_folder.mkdir(exist_ok=True, parents=True)
    create_plot(plots_folder)
