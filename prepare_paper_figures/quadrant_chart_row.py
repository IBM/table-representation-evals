import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

from benchmark_src.results_processing.metric_info import get_metric_domain


def build_quadrant_chart(df: pd.DataFrame, plots_folder: Path):
    metric = "MRR"
    second_metric = "execution_time (s)"

            # strip * from chart names
    df['chart_name'] = df['chart_name'].str.replace('*', '', regex=False)

    # ----------------------------------------------------------------
    # Keep only datasets where we have results for all approaches/configs
    # ----------------------------------------------------------------
    dataset_counts = (
        df.groupby("dataset")[["Approach", "Configuration"]]
        .nunique()
        .reset_index()
        .rename(columns={"Approach": "num_approaches", "Configuration": "num_configs"})
    )

    n_approaches = df["Approach"].nunique()
    n_configs = df["Configuration"].nunique()

    full_datasets = dataset_counts[
        (dataset_counts["num_approaches"] == n_approaches)
        & (dataset_counts["num_configs"] == n_configs)
    ]["dataset"].tolist()

    df_filtered = df[df["dataset"].isin(full_datasets)]
    num_datasets = df_filtered["dataset"].nunique()
    print(f"Unique datasets used: {num_datasets}")

    # ----------------------------------------------------------------
    # Aggregate over datasets per (Approach, Configuration)
    # ----------------------------------------------------------------
    agg_df = (
        df_filtered.groupby(["Approach", "Configuration"])
        .agg(
            **{metric: (metric, "mean")},
            **{second_metric: (second_metric, "mean")},
            color=("color", "first"),
            chart_name=("chart_name", "first"),
        )
        .reset_index()
    )

    # ----------------------------------------------------------------
    # Quadrant thresholds — midpoint of axis domain
    # ----------------------------------------------------------------
    x_domain_min, x_domain_max = get_metric_domain(agg_df[second_metric], second_metric)
    y_domain_min, y_domain_max = get_metric_domain(agg_df[metric], metric)

    x_domain_max *= 1.1  # padding so rightmost points aren't clipped

    x_mid = (0 + x_domain_max) / 2
    y_mid = (y_domain_min + y_domain_max) / 2

    # ----------------------------------------------------------------
    # Figure
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    # ----------------------------------------------------------------
    # Quadrant dividing lines
    # ----------------------------------------------------------------
    ax.axvline(x=x_mid, color="grey", linewidth=1.8, linestyle="--", zorder=2)
    ax.axhline(y=y_mid, color="grey", linewidth=1.8, linestyle="--", zorder=2)

    # ----------------------------------------------------------------
    # Scatter points — one per row, color and label from df columns
    # ----------------------------------------------------------------
    for _, row in agg_df.iterrows():
        x = row[second_metric]
        y = row[metric]
        color = row["color"]
        label = row["chart_name"]

        ax.scatter(
            x, y,
            color=color,
            s=180,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )

        # Point annotation — slight offset, white halo for readability
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(10, 6),
            textcoords="offset points",
            fontsize=14,
            color=color,
            zorder=5,
            path_effects=[
                pe.withStroke(linewidth=2.5, foreground="white")
            ],
        )

    # ----------------------------------------------------------------
    # Axes styling
    # ----------------------------------------------------------------
    ax.set_xlim(0, x_domain_max)
    ax.set_ylim(y_domain_min, 1)

    print(f"Set x axis limit to {x_domain_max} and y axis limit to {1}")  # MRR max is 1

    ax.set_xlabel("Execution Time (s)", fontsize=16)
    ax.set_ylabel("MRR (mean)", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)


    # ax.set_title(
    #     f"Performance vs. Execution Time\n"
    #     f"(aggregated across {num_datasets} dataset(s))",
    #     fontsize=13,
    #     fontweight="bold",
    #     pad=14,
    # )

    ax.grid(True, linestyle=":", alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------

    # Save PDF
    plt.savefig(plots_folder / f"row_sim_quadrant_chart_{num_datasets}_datasets.pdf")
    plt.close()


def build_quadrant_chart_vram_aggregated(df: pd.DataFrame, plots_folder: Path):
    metric = "MRR"
    second_metric = "peak_gpu_memory (MB)"

    # strip * from chart names
    df['chart_name'] = df['chart_name'].str.replace('*', '', regex=False)

    # ----------------------------------------------------------------
    # Keep only datasets where we have results for all approaches/configs
    # ----------------------------------------------------------------
    dataset_counts = (
        df.groupby("dataset")[["Approach", "Configuration"]]
        .nunique()
        .reset_index()
        .rename(columns={"Approach": "num_approaches", "Configuration": "num_configs"})
    )

    n_approaches = df["Approach"].nunique()
    n_configs = df["Configuration"].nunique()

    full_datasets = dataset_counts[
        (dataset_counts["num_approaches"] == n_approaches)
        & (dataset_counts["num_configs"] == n_configs)
    ]["dataset"].tolist()

    df_filtered = df[df["dataset"].isin(full_datasets)]
    num_datasets = df_filtered["dataset"].nunique()
    print(f"Unique datasets used: {num_datasets}")

    # ----------------------------------------------------------------
    # Aggregate over datasets per (Approach, Configuration)
    # ----------------------------------------------------------------
    agg_df = (
        df_filtered.groupby(["Approach", "Configuration"])
        .agg(
            **{metric: (metric, "mean")},
            **{second_metric: (second_metric, "mean")},
            color=("color", "first"),
            chart_name=("chart_name", "first"),
        )
        .reset_index()
    )

    # Convert MB -> GB for display
    gb_metric = "peak_gpu_memory (GB)"
    agg_df[gb_metric] = agg_df[second_metric] / 1024

    # ----------------------------------------------------------------
    # Quadrant thresholds — midpoint of axis domain
    # ----------------------------------------------------------------
    x_domain_min, x_domain_max = get_metric_domain(agg_df[second_metric], second_metric)
    x_domain_min, x_domain_max = x_domain_min / 1024, x_domain_max / 1024
    y_domain_min, y_domain_max = get_metric_domain(agg_df[metric], metric)

    x_domain_max *= 1.1  # padding so rightmost points aren't clipped

    x_mid = (0 + x_domain_max) / 2
    y_mid = (y_domain_min + y_domain_max) / 2

    # ----------------------------------------------------------------
    # Figure
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    # ----------------------------------------------------------------
    # Quadrant dividing lines
    # ----------------------------------------------------------------
    ax.axvline(x=x_mid, color="grey", linewidth=1.8, linestyle="--", zorder=2)
    ax.axhline(y=y_mid, color="grey", linewidth=1.8, linestyle="--", zorder=2)

    # ----------------------------------------------------------------
    # Scatter points — one per row, color and label from df columns
    # ----------------------------------------------------------------
    for _, row in agg_df.iterrows():
        x = row[gb_metric]
        y = row[metric]
        color = row["color"]
        label = row["chart_name"]

        ax.scatter(
            x, y,
            color=color,
            s=180,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )

        ax.annotate(
            label,
            xy=(x, y),
            xytext=(10, 6),
            textcoords="offset points",
            fontsize=14,
            color=color,
            zorder=5,
            path_effects=[
                pe.withStroke(linewidth=2.5, foreground="white")
            ],
        )

    # ----------------------------------------------------------------
    # Axes styling
    # ----------------------------------------------------------------
    ax.set_xlim(0, x_domain_max)
    ax.set_ylim(y_domain_min, y_domain_max)

    print(f"Set x axis limit to {x_domain_max} GB and y axis limit to {y_domain_max}")

    ax.set_xlabel("Peak GPU Memory (GB)", fontsize=16)
    ax.set_ylabel("MRR (mean)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

    ax.grid(True, linestyle=":", alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    # ----------------------------------------------------------------
    # Save PDF
    # ----------------------------------------------------------------
    plt.savefig(plots_folder / f"row_sim_quadrant_chart_{num_datasets}_datasets_vram.pdf")
    plt.close()


def build_quadrant_chart_vram(df: pd.DataFrame, plots_folder: Path):
    metric = "MRR"
    second_metric = "peak_gpu_memory (MB)"

    # ----------------------------------------------------------------
    # One point per (dataset, Approach, Configuration) — no filtering
    # ----------------------------------------------------------------
    plot_df = (
        df.groupby(["dataset", "Approach", "Configuration"])
        .agg(
            **{metric: (metric, "mean")},
            **{second_metric: (second_metric, "mean")},
            color=("color", "first"),
            chart_name=("chart_name", "first"),
            marker=("marker", "first"),
        )
        .reset_index()
    )

    num_datasets = plot_df["dataset"].nunique()
    print(f"Unique datasets: {num_datasets}")

    # Convert MB -> GB for display
    gb_metric = "peak_gpu_memory (GB)"
    plot_df[gb_metric] = plot_df[second_metric] / 1024

    # ----------------------------------------------------------------
    # Quadrant thresholds — midpoint of axis domain
    # ----------------------------------------------------------------
    x_domain_min, x_domain_max = get_metric_domain(plot_df[second_metric], second_metric)
    x_domain_min, x_domain_max = x_domain_min / 1024, x_domain_max / 1024
    y_domain_min, y_domain_max = get_metric_domain(plot_df[metric], metric)

    x_domain_max *= 1.1  # padding so rightmost points aren't clipped

    x_mid = (0 + x_domain_max) / 2
    y_mid = (y_domain_min + y_domain_max) / 2

    # ----------------------------------------------------------------
    # Figure
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    # ----------------------------------------------------------------
    # Quadrant dividing lines
    # ----------------------------------------------------------------
    ax.axvline(x=x_mid, color="grey", linewidth=1.8, linestyle="--", zorder=2)
    ax.axhline(y=y_mid, color="grey", linewidth=1.8, linestyle="--", zorder=2)

    # ----------------------------------------------------------------
    # Scatter points — color and marker shape by approach (curated marker
    # shapes disambiguate approaches whose colors are visually similar),
    # legend by chart_name
    # ----------------------------------------------------------------
    seen_labels = set()

    for _, row in plot_df.iterrows():
        x = row[gb_metric]
        y = row[metric]
        color = row["color"]
        chart_name = row["chart_name"]
        marker = row["marker"]

        scatter_label = chart_name if chart_name not in seen_labels else None
        seen_labels.add(chart_name)

        ax.scatter(
            x, y,
            color=color,
            marker=marker,
            s=60,
            alpha=0.85,
            zorder=4,
            label=scatter_label,
        )

    # ----------------------------------------------------------------
    # Legend below the plot, 6 columns
    # ----------------------------------------------------------------
    ax.legend(
        fontsize=14,
        framealpha=0.7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=6,
    )

    # ----------------------------------------------------------------
    # Axes styling
    # ----------------------------------------------------------------
    ax.set_xlim(0, x_domain_max)
    ax.set_ylim(y_domain_min, y_domain_max)

    print(f"Set x axis limit to {x_domain_max} GB and y axis limit to {y_domain_max}")

    ax.set_xlabel("Peak GPU Memory (GB)", fontsize=16)
    ax.set_ylabel("MRR (mean)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

    ax.grid(True, linestyle=":", alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    # ----------------------------------------------------------------
    # Save PDF
    # ----------------------------------------------------------------
    plt.savefig(
        plots_folder / f"row_sim_quadrant_chart_{num_datasets}_datasets_vram.pdf",
        bbox_inches="tight",
    )
    plt.close()
