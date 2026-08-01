import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import config_helpers as h

METRICS = ("mean_mrr", "mean_map")
METRIC_LABELS = {
    "mean_mrr": "Mean Reciprocal Rank (MRR)",
    "mean_map": "Mean Average Precision (MAP)",
}


def create_barplot(df: pd.DataFrame, plots_folder: Path):
    """One bar per approach, approach names on the x-axis (no legend) — schema linking
    has only a single dataset, so a per-dataset grouping would just repeat that one bar."""
    for metric in METRICS:
        if metric not in df.columns:
            print(f"WARNING: {metric} not found in schema_linking data, skipping")
            continue

        metric_df = df.dropna(subset=[metric]).copy()
        metric_df["chart_name"] = metric_df["chart_name"].str.replace("*", "", regex=False)

        # order bars alphabetically by chart_name (display name), not by score
        plot_df = (
            metric_df[["Approach", "chart_name", "color", metric]]
            .drop_duplicates(subset=["chart_name"])
            .sort_values("chart_name", key=lambda s: s.map(h.chart_name_sort_key))
        )

        fig, ax = plt.subplots(figsize=(12, 4.8))

        bars = ax.bar(
            x=plot_df["chart_name"],
            height=plot_df[metric],
            color=plot_df["color"],
            width=0.8,
        )

        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df["chart_name"], rotation=20, ha="right", fontsize=16)

        ax.set_ylabel(METRIC_LABELS[metric], fontsize=16)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=14)

        ax.grid(axis="y", linestyle="--", alpha=0.3)

        for bar, value in zip(bars, plot_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center", va="bottom", fontsize=14,
            )

        plt.tight_layout()
        metric_slug = metric.replace("mean_", "")
        plt.savefig(plots_folder / f"schema_linking_barchart_{metric_slug}.pdf")
        plt.close()
