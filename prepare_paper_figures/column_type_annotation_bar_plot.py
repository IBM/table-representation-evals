import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

METRIC = "macro_f1 (↑)"


def create_barplot(df: pd.DataFrame, plots_folder: Path):
    df = df.dropna(subset=[METRIC]).copy()
    df["chart_name"] = df["chart_name"].str.replace("*", "", regex=False)

    # keep only datasets where every approach has a result, so the "Mean" group
    # is comparable across approaches
    n_approaches = df["chart_name"].nunique()
    full_datasets = df.groupby("dataset")["chart_name"].nunique()
    full_datasets = full_datasets[full_datasets == n_approaches].index
    df_filtered = df[df["dataset"].isin(full_datasets)]

    plot_df = df_filtered[["dataset", METRIC, "color", "chart_name"]].copy()

    # order methods by raw Approach name, not by score
    methods = (
        df_filtered[["Approach", "chart_name"]]
        .drop_duplicates()
        .sort_values("Approach")["chart_name"]
        .tolist()
    )

    # compute mean across datasets per method, appended as a pseudo-dataset
    avg_df = plot_df.groupby(["chart_name", "color"], as_index=False)[METRIC].mean()
    avg_df["dataset"] = "Mean"
    plot_df = pd.concat([avg_df, plot_df], ignore_index=True)

    datasets = [d for d in plot_df["dataset"].unique() if d != "Mean"] + ["Mean"]

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * len(datasets) * 0.4), 6))

    n_datasets = len(datasets)
    n_methods = len(methods)
    gap = 0.025
    bar_width = (0.8 - gap * (n_methods - 1)) / n_methods

    x = np.arange(n_datasets)
    total_group_width = n_methods * bar_width + (n_methods - 1) * gap
    group_centers = x + total_group_width / 2

    for i, method in enumerate(methods):
        method_df = (
            plot_df[plot_df["chart_name"] == method]
            .set_index("dataset")
            .reindex(datasets)
        )
        positions = x + i * (bar_width + gap)
        color = plot_df.loc[plot_df["chart_name"] == method, "color"].iloc[0]

        bars = ax.bar(positions, method_df[METRIC], width=bar_width, color=color, label=method)

        for dataset_name, bar in zip(datasets, bars):
            height = bar.get_height()
            if pd.isna(height):
                continue
            if dataset_name == "Mean":
                bar.set_alpha(0.8)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.2f}",
                ha="center", va="bottom", fontsize=10,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(datasets, rotation=0, ha="center", fontsize=14)

    ax.set_ylabel("Macro F1", fontsize=16)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=6,
        frameon=False,
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(plots_folder / "cta_barchart_macro_f1_per_dataset.pdf")
    plt.close()
