import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

DIFFICULTY_METRICS = ["accuracy_easy", "accuracy_medium"]
DIFFICULTY_LABELS = {"accuracy_easy": "Easy", "accuracy_medium": "Medium"}
DIFFICULTY_HATCHES = {"accuracy_easy": None, "accuracy_medium": "//"}


def create_barplot(df: pd.DataFrame, plots_folder: Path):
    """Bar plot comparing easy vs. medium accuracy per approach on wikidata_books."""
    df_filtered = df[df["dataset"] == "wikidata_books"].copy()
    df_filtered["chart_name"] = df_filtered["chart_name"].str.replace("*", "", regex=False)

    plot_df = df_filtered[["Approach", "chart_name", "color"] + DIFFICULTY_METRICS].copy()
    # order methods by raw Approach name, matching the convention used elsewhere
    # (e.g. create_plots.py, row_sim_plot_aggregated.py) rather than by score
    methods = (
        plot_df.sort_values("Approach")["chart_name"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    n_metrics = len(DIFFICULTY_METRICS)
    gap = 0.025
    bar_width = (0.8 - gap * (n_metrics - 1)) / n_metrics

    x = np.arange(len(methods))
    total_group_width = n_metrics * bar_width + (n_metrics - 1) * gap
    group_centers = x + total_group_width / 2

    for i, metric in enumerate(DIFFICULTY_METRICS):
        method_df = plot_df.set_index("chart_name").loc[methods].reset_index()

        positions = x + i * (bar_width + gap)

        bars = ax.bar(
            positions,
            method_df[metric],
            width=bar_width,
            color=method_df["color"],
            hatch=DIFFICULTY_HATCHES[metric],
            edgecolor="white",
            label=DIFFICULTY_LABELS[metric],
        )

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=0,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(methods, rotation=0, ha="center", fontsize=13)

    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    random_line = ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, zorder=0)

    # legend distinguishes difficulty via hatch, not color (color already encodes approach)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=DIFFICULTY_HATCHES[m])
        for m in DIFFICULTY_METRICS
    ] + [random_line]
    ax.legend(
        legend_handles,
        [DIFFICULTY_LABELS[m] for m in DIFFICULTY_METRICS] + ["Random"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=13,
    )

    plt.tight_layout()

    plt.savefig(plots_folder / "triplet_row_barchart_difficulty_wikidata_books.pdf")
    plt.close()
