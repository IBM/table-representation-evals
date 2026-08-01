import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

from benchmark_src.results_processing.metric_info import get_metric_domain
import config_helpers as h


def build_quadrant_chart(df: pd.DataFrame, plots_folder: Path):
    second_metric = "execution_time (s)"

    # strip * from chart names
    df['chart_name'] = df['chart_name'].str.replace('*', '', regex=False)

    for metric in ("accuracy", "MAP"):
        if metric not in df.columns:
            print(f"WARNING: {metric} not found in cell_similarity_search data, skipping")
            continue

        # ----------------------------------------------------------------
        # One point per (dataset, Approach, Configuration) — no aggregation
        # over datasets, just keep color and chart_name per group
        # ----------------------------------------------------------------
        plot_df = (
            df.groupby(["dataset", "Approach", "Configuration"])
            .agg(
                **{metric: (metric, "mean")},
                **{second_metric: (second_metric, "mean")},
                color=("color", "first"),
                chart_name=("chart_name", "first"),
            )
            .reset_index()
        )

        # legend entries are added in row-iteration order (first time a chart_name
        # is seen), so sort by chart_name (display name) here rather than relying
        # on the default (dataset, raw Approach name) groupby order
        plot_df = plot_df.sort_values("chart_name", key=lambda s: s.map(h.chart_name_sort_key)).reset_index(drop=True)

        num_datasets = plot_df["dataset"].nunique()
        print(f"Unique datasets: {num_datasets}")

        # ----------------------------------------------------------------
        # Quadrant thresholds — midpoint of axis domain
        # ----------------------------------------------------------------
        x_domain_min, x_domain_max = get_metric_domain(plot_df[second_metric], second_metric)
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
        # ax.axvline(x=x_mid, color="grey", linewidth=1.8, linestyle="--", zorder=2)
        # ax.axhline(y=y_mid, color="grey", linewidth=1.8, linestyle="--", zorder=2)

        # ----------------------------------------------------------------
        # Scatter points — one per (dataset, Approach, Configuration)
        # Legend: one entry per unique chart_name, colored by approach.
        # SAP-RPT-1 and TabDPT get non-circle markers, matching the row
        # similarity search execution-time quadrant chart; HyTrel additionally
        # gets a triangle here since its color clashes with a neighbor in this
        # chart specifically.
        # ----------------------------------------------------------------
        marker_overrides = {"SAP-RPT-1": "^", "TabDPT": "s", "HyTrel": "^"}
        seen_labels = set()  # track chart_names already added to legend

        for _, row in plot_df.iterrows():
            x = row[second_metric]
            y = row[metric]
            color = row["color"]
            chart_name = row["chart_name"]
            marker = next(
                (m for name, m in marker_overrides.items() if chart_name.startswith(name)),
                "o",
            )

            # Only pass label the first time we see this chart_name
            scatter_label = chart_name if chart_name not in seen_labels else None
            seen_labels.add(chart_name)

            ax.scatter(
                x, y,
                color=color,
                marker=marker,
                s=180,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
                label=scatter_label,
            )

            # Label each point with just the dirty/clean variant, not the full dataset name
            dataset_label = row["dataset"].split("@")[-1]

            ax.annotate(
                dataset_label,
                xy=(x, y),
                xytext=(10, 6),
                textcoords="offset points",
                fontsize=14,
                color=color,
                alpha=0.75,
                zorder=5,
                path_effects=[
                    pe.withStroke(linewidth=2.5, foreground="white")
                ],
            )

        # Legend below figure
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=5,
            frameon=False,
            fontsize=13
        )

        # ----------------------------------------------------------------
        # Axes styling
        # ----------------------------------------------------------------
        ax.set_xlim(0, x_domain_max)
        ax.set_ylim(y_domain_min, y_domain_max)

        ax.set_xlabel("Execution Time (s)", fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.tick_params(axis='both', labelsize=14)

        ax.grid(True, linestyle=":", alpha=0.4, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()

        # ----------------------------------------------------------------
        # Save
        # ----------------------------------------------------------------

        # Save PDF
        plt.savefig(plots_folder / f"cell_quadrant_chart_{metric.lower()}.pdf")
        plt.close()
