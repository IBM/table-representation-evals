"""Distribution of d_pos - d_neg per triplet, faceted by approach.

Negative margin = inversion (model embeds positive further from anchor than negative).
Zero margin = tie (BoW / permutation-invariant).
Positive margin = correct structural discrimination.

Shows the two failure modes as distinct distribution shapes:
- BoW: delta distribution at exactly 0 (tie artifact)
- Transformers: distribution shifted negative (real inversion — tracking surface change)
- HyTrel: distribution shifted positive (structural signal)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_BASE = Path("results_complete/results/table_paper_experiments")
DATASET = "fetaqa@@v0"
SERIALIZATION = "markdown"

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


def create_plot(plots_folder: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (chart_name, result_dir, config_dir) in zip(axes, APPROACH_CONFIGS):
        path = RESULTS_BASE / result_dir / config_dir / "table_shuffling" / DATASET / "full_results.json"
        if not path.exists():
            ax.set_title(f"{chart_name}\n(no data)")
            continue

        with open(path) as f:
            data = json.load(f)
        triplets = data["triplets"]

        margins = np.array([t["d_pos"] - t["d_neg"] for t in triplets])
        acc = (margins < 0).mean()  # strict inequality
        color = COLORS.get(chart_name, "#000000")

        # Histogram
        bins = np.linspace(-0.15, 0.15, 61)
        ax.hist(margins, bins=bins, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)

        # Mean line
        mean_margin = margins.mean()
        ax.axvline(x=mean_margin, color=color, linewidth=2, linestyle='--', alpha=0.8)
        ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)

        # Inversion fraction
        inv_frac = (margins > 0).mean()

        ax.set_title(f"{chart_name}\nacc={acc:.3f}  mean Δ={mean_margin:.4f}  inv={inv_frac:.2f}",
                    fontsize=11, color=color)
        ax.set_xlabel('d_pos − d_neg', fontsize=9)
        ax.set_ylabel('Triplets', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=9)

    fig.suptitle(f'Margin Distribution: d_pos − d_neg per Triplet\n{DATASET}, v0, {SERIALIZATION}\n'
                 '(Δ<0 = correct; Δ=0 = tie; Δ>0 = inversion)',
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = plots_folder / 'shuffling_margin_dist.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plots_folder = Path("./prepare_paper_figures/main_table_experiments")
    plots_folder.mkdir(exist_ok=True, parents=True)
    create_plot(plots_folder)
