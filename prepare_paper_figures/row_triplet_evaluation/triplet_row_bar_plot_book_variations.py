"""Row triplet evaluation: accuracy across the perturbation-variant datasets
(column-name/genre/column-count ablations on wikidata_books, text-only ablation
on astronomical_objects) — the data behind the paper's Table 6, as two figures
(one per dataset family), grouped by approach so robustness to the variations
is easy to compare across approaches."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

import config_helpers as h

ASTRO_DATASETS = [
    "astronomical_objects",
    "astronomical_objects@only-text",
]

ASTRO_VARIATION_LABELS = {
    "astronomical_objects": "Wikidata Astronomical Objects (Original)",
    "astronomical_objects@only-text": "Text Only",
}

BOOKS_DATASETS = [
    "wikidata_books",
    "wikidata_books@no_col_names",
    "wikidata_books@no_pid_in_col_names",
    "wikidata_books@no_genre",
    "wikidata_books@only_five_cols",
]

BOOKS_VARIATION_LABELS = {
    "wikidata_books": "Wikidata Books (Original)",
    "wikidata_books@no_col_names": "No Column Names",
    "wikidata_books@no_pid_in_col_names": "Column Names Without Property IDs",
    "wikidata_books@no_genre": "No Genre Column",
    "wikidata_books@only_five_cols": "Only Five Columns",
}

# baseline (index 0) is always solid; later variations get distinct hatches
HATCH_CYCLE = [None, "//", "xx", "..", "oo", "\\\\"]


def _plot_family(
    df: pd.DataFrame,
    datasets: list,
    variation_labels: dict,
    plots_folder: Path,
    output_filename: str,
    group_total_width: float = 0.8,
    label_rotation: int = 30,
    legend_ncol: int = None,
    show_bar_labels: bool = False,
):
    metric = "accuracy"

    df_filtered = df[df["dataset"].isin(datasets)].copy()
    present_datasets = [d for d in datasets if d in df_filtered["dataset"].unique()]
    missing = set(datasets) - set(present_datasets)
    if missing:
        print(f"WARNING: no row_triplet_evaluation results for variant datasets {missing}")

    df_filtered["chart_name"] = df_filtered["chart_name"].str.replace("*", "", regex=False)
    plot_df = df_filtered[["Approach", "chart_name", "dataset", metric, "color"]].copy()

    # order approach groups alphabetically by chart_name (display name), matching
    # the convention used elsewhere (e.g. triplet_row_bar_plot_difficulty.py)
    # rather than by score
    methods = sorted(plot_df["chart_name"].unique(), key=h.chart_name_sort_key)

    n_methods = len(methods)
    n_variations = len(present_datasets)
    gap = 0.025
    bar_width = (group_total_width - gap * (n_variations - 1)) / n_variations

    x = np.arange(n_methods)
    total_group_width = n_variations * bar_width + (n_variations - 1) * gap
    group_centers = x + total_group_width / 2

    fig, ax = plt.subplots(figsize=(max(12, 1.4 * n_methods), 6))

    random_line = ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, zorder=0)

    for i, dataset in enumerate(present_datasets):
        dataset_df = (
            plot_df[plot_df["dataset"] == dataset]
            .set_index("chart_name")
            .reindex(methods)
            .reset_index()
        )

        positions = x + i * (bar_width + gap)

        bars = ax.bar(
            positions,
            dataset_df[metric],
            width=bar_width,
            color=dataset_df["color"],
            hatch=HATCH_CYCLE[i % len(HATCH_CYCLE)],
            edgecolor="white",
        )

        if show_bar_labels:
            for bar in bars:
                height = bar.get_height()
                if np.isnan(height):
                    continue
                label = "0" if height == 0 else f"{height:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.015,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    rotation=0,
                )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(methods, rotation=label_rotation, ha="right", fontsize=16)

    ax.set_ylabel("Accuracy", fontsize=19)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=17)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # legend distinguishes variation via hatch, not color (color already encodes approach)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=HATCH_CYCLE[i % len(HATCH_CYCLE)])
        for i in range(n_variations)
    ] + [random_line]
    legend_labels = [variation_labels[d] for d in present_datasets] + ["Random"]

    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=legend_ncol or len(legend_labels),
        frameon=False,
        fontsize=15,
    )

    plt.tight_layout()
    fig.savefig(plots_folder / output_filename, bbox_inches="tight")
    plt.close(fig)


def create_barplot(df: pd.DataFrame, plots_folder: Path):
    df = df.dropna(axis=1, how="all")

    _plot_family(
        df, ASTRO_DATASETS, ASTRO_VARIATION_LABELS, plots_folder,
        "triplet_row_barchart_variations_astronomical_objects.pdf",
        show_bar_labels=True,
    )
    _plot_family(
        df, BOOKS_DATASETS, BOOKS_VARIATION_LABELS, plots_folder,
        "triplet_row_barchart_variations_books.pdf",
        group_total_width=0.9,
        label_rotation=15,
        legend_ncol=3,
    )
