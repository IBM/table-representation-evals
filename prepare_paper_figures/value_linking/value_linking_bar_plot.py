import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import config_helpers as h

METRICS = ("mean_mrr", "mean_map")
METRIC_LABELS = {
    "mean_mrr": "Mean Reciprocal Rank (MRR)",
    "mean_map": "Mean Average Precision (MAP)",
}

# baseline dataset (index 0) is always solid; later datasets/"Mean" get distinct hatches
HATCH_CYCLE = [None, "//", "xx", "..", "oo", "\\\\"]


def create_barplot(df: pd.DataFrame, plots_folder: Path):
    """One group of bars per approach, colored by approach, with one bar per dataset
    plus an aggregated 'Mean' bar within each group — dataset identity (or 'Mean') is
    distinguished via hatch pattern rather than color, since color already identifies
    the approach."""
    for metric in METRICS:
        if metric not in df.columns:
            print(f"WARNING: {metric} not found in value_linking data, skipping")
            continue

        metric_df = df.dropna(subset=[metric]).copy()
        metric_df["chart_name"] = metric_df["chart_name"].str.replace("*", "", regex=False)

        # keep only datasets where every approach has a result, so the "Mean" bar
        # is comparable across approaches
        n_approaches = metric_df["chart_name"].nunique()
        full_datasets = metric_df.groupby("dataset")["chart_name"].nunique()
        full_datasets = full_datasets[full_datasets == n_approaches].index
        df_filtered = metric_df[metric_df["dataset"].isin(full_datasets)]

        plot_df = df_filtered[["dataset", metric, "color", "chart_name"]].copy()

        # order approaches alphabetically by chart_name (display name), not by score
        approaches = sorted(df_filtered["chart_name"].unique(), key=h.chart_name_sort_key)

        avg_df = plot_df.groupby(["chart_name", "color"], as_index=False)[metric].mean()
        avg_df["dataset"] = "Mean"
        plot_df = pd.concat([plot_df, avg_df], ignore_index=True)

        datasets = sorted(d for d in plot_df["dataset"].unique() if d != "Mean")
        bar_kinds = datasets + ["Mean"]

        n_methods = len(approaches)
        n_kinds = len(bar_kinds)
        group_total_width = 0.8
        bar_width = group_total_width / n_kinds

        x = np.arange(n_methods)
        group_centers = x + group_total_width / 2

        fig, ax = plt.subplots(figsize=(max(12, 1.4 * n_methods), 5.4))

        for i, bar_kind in enumerate(bar_kinds):
            kind_df = (
                plot_df[plot_df["dataset"] == bar_kind]
                .set_index("chart_name")
                .reindex(approaches)
                .reset_index()
            )

            positions = x + i * bar_width

            bars = ax.bar(
                positions,
                kind_df[metric],
                width=bar_width,
                color=kind_df["color"],
                hatch=HATCH_CYCLE[i % len(HATCH_CYCLE)],
                edgecolor="white",
            )

            for value, bar in zip(kind_df[metric], bars):
                if pd.isna(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.2f}",
                    ha="center", va="bottom", fontsize=11,
                )

        ax.set_xticks(group_centers)
        ax.set_xticklabels(approaches, rotation=15, ha="right", fontsize=15)

        ax.set_ylabel(METRIC_LABELS[metric], fontsize=16)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=14)

        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # legend distinguishes dataset/"Mean" via hatch, not color (color already encodes approach)
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=HATCH_CYCLE[i % len(HATCH_CYCLE)])
            for i in range(n_kinds)
        ]
        legend_labels = [h.VALUE_LINKING_DATASET_NAMES.get(kind, kind) for kind in bar_kinds]
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=n_kinds,
            frameon=False,
            fontsize=15,
        )

        plt.tight_layout()
        metric_slug = metric.replace("mean_", "")
        fig.savefig(plots_folder / f"value_linking_barchart_{metric_slug}_per_dataset.pdf", bbox_inches="tight")
        plt.close(fig)
