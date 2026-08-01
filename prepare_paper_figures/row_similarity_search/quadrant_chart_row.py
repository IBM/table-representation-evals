import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

from benchmark_src.results_processing.metric_info import get_metric_domain
import config_helpers as h

# Common GPU VRAM capacities (GB), shown as reference lines on VRAM quadrant charts.
GPU_VRAM_GB = {
    "V100 16GB": 16,
    "RTX 4090": 24,
    "A100 40GB": 40,
    "A100 80GB": 80,
}

METRICS = ("MRR", "MAP")


def _add_gpu_vram_reference_lines(ax, x_domain_max: float) -> float:
    # Extend the axis if needed so the largest reference line stays on-chart.
    x_domain_max = max(x_domain_max, max(GPU_VRAM_GB.values()) * 1.05)

    for label, vram_gb in GPU_VRAM_GB.items():
        ax.axvline(x=vram_gb, color="grey", linewidth=1, linestyle=":", zorder=2)
        ax.text(
            vram_gb, 0.97, label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=13,
            color="grey",
            zorder=3,
        )

    return x_domain_max


def build_quadrant_chart(df: pd.DataFrame, plots_folder: Path):
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

    for metric in METRICS:
        if metric not in df_filtered.columns:
            print(f"WARNING: {metric} not found in row_similarity_search data, skipping")
            continue

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

        # legend order (for points pushed into the lower-left-cluster legend below)
        # follows iteration order over agg_df, so sort alphabetically by chart_name
        # (display name) here rather than relying on the default groupby order
        agg_df = agg_df.sort_values("chart_name", key=lambda s: s.map(h.chart_name_sort_key)).reset_index(drop=True)

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
        fig, ax = plt.subplots(figsize=(12, 4))

        # ----------------------------------------------------------------
        # Quadrant dividing lines
        # ----------------------------------------------------------------
        # ax.axvline(x=x_mid, color="grey", linewidth=1.8, linestyle="--", zorder=2)
        # ax.axhline(y=y_mid, color="grey", linewidth=1.8, linestyle="--", zorder=2)

        # ----------------------------------------------------------------
        # Scatter points — one per row, color and label from df columns.
        # Points in the crowded lower-left quadrant get no inline label (to
        # avoid hiding neighboring dots) and are listed in a legend below the
        # plot instead — except TaBERT and TABBIE, which stay labeled inline.
        # SAP-RPT-1 and TabDPT get non-circle markers since their colors are
        # close to TABBIE's in that same cluster.
        # ----------------------------------------------------------------
        inline_label_names = ("TaBERT", "TABBIE")
        marker_overrides = {"SAP-RPT-1": "^", "TabDPT": "s"}

        legend_handles = []
        for _, row in agg_df.iterrows():
            x = row[second_metric]
            y = row[metric]
            color = row["color"]
            label = row["chart_name"]
            marker = next(
                (m for name, m in marker_overrides.items() if label.startswith(name)),
                "o",
            )
            in_lower_left = x < x_mid and y < y_mid
            has_inline_label = not in_lower_left or label in inline_label_names

            ax.scatter(
                x, y,
                color=color,
                marker=marker,
                s=180 if has_inline_label else 120,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )

            if not has_inline_label:
                legend_handles.append(
                    plt.Line2D([0], [0], marker=marker, linestyle="", color=color, markersize=10, label=label)
                )
                continue

            # Point annotation — slight offset, white halo for readability.
            # alpha < 1 so a label overlapping another point doesn't fully hide its dot.
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=14,
                color=color,
                alpha=0.6,
                zorder=5,
                path_effects=[
                    pe.withStroke(linewidth=2.5, foreground="white", alpha=0.6)
                ],
            )

        # ----------------------------------------------------------------
        # Axes styling
        # ----------------------------------------------------------------
        ax.set_xlim(0, x_domain_max)
        ax.set_ylim(y_domain_min, 1)

        print(f"Set x axis limit to {x_domain_max} and y axis limit to {1}")  # metric max is 1

        ax.set_xlabel("Execution Time (s)", fontsize=16)
        ax.set_ylabel(f"{metric} (mean)", fontsize=16)
        ax.tick_params(axis='both', labelsize=14)

        ax.grid(True, linestyle=":", alpha=0.4, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=len(legend_handles),
                fontsize=16,
                handletextpad=0.2,
                columnspacing=0.8,
                frameon=False,
            )

        plt.tight_layout()

        # ----------------------------------------------------------------
        # Save
        # ----------------------------------------------------------------

        # Save PDF
        plt.savefig(plots_folder / f"row_sim_quadrant_chart_{metric.lower()}_{num_datasets}_datasets.pdf")
        plt.close()


def build_quadrant_chart_vram_aggregated(df: pd.DataFrame, plots_folder: Path):
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

    for metric in METRICS:
        if metric not in df_filtered.columns:
            print(f"WARNING: {metric} not found in row_similarity_search data, skipping")
            continue

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

        # order alphabetically by chart_name (display name), not the default
        # (raw Approach name) groupby order
        agg_df = agg_df.sort_values("chart_name", key=lambda s: s.map(h.chart_name_sort_key)).reset_index(drop=True)

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

        # ----------------------------------------------------------------
        # Figure
        # ----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 5))

        # ----------------------------------------------------------------
        # Quadrant dividing lines
        # ----------------------------------------------------------------
        x_domain_max = _add_gpu_vram_reference_lines(ax, x_domain_max)

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
        ax.set_ylabel(f"{metric} (mean)", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)

        ax.grid(True, linestyle=":", alpha=0.4, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()

        # ----------------------------------------------------------------
        # Save PDF
        # ----------------------------------------------------------------
        plt.savefig(plots_folder / f"row_sim_quadrant_chart_{metric.lower()}_{num_datasets}_datasets_vram.pdf")
        plt.close()


def build_quadrant_chart_vram(df: pd.DataFrame, plots_folder: Path):
    second_metric = "peak_gpu_memory (MB)"

    for metric in METRICS:
        if metric not in df.columns:
            print(f"WARNING: {metric} not found in row_similarity_search data, skipping")
            continue

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

        # legend entries are added in row-iteration order (first time a chart_name
        # is seen), so sort by chart_name (display name) here rather than relying
        # on the default (dataset, raw Approach name) groupby order
        plot_df = plot_df.sort_values("chart_name", key=lambda s: s.map(h.chart_name_sort_key)).reset_index(drop=True)

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

        # ----------------------------------------------------------------
        # Figure
        # ----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 5))

        # ----------------------------------------------------------------
        # Quadrant dividing lines
        # ----------------------------------------------------------------
        x_domain_max = _add_gpu_vram_reference_lines(ax, x_domain_max)

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
        ax.set_ylabel(f"{metric} (mean)", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)

        ax.grid(True, linestyle=":", alpha=0.4, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()

        # ----------------------------------------------------------------
        # Save PDF
        # ----------------------------------------------------------------
        plt.savefig(
            plots_folder / f"row_sim_quadrant_chart_{metric.lower()}_{num_datasets}_datasets_vram.pdf",
            bbox_inches="tight",
        )
        plt.close()
