import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import json
from pathlib import Path


def get_metric_domain(df_filtered: pd.DataFrame, selected_metric: str):
    """
    Determines the domain min and max for a given metric based on metric information or data.
    Handles symbolic infinity '∞'.
    """
    try:
        with open("./prepare_paper_figures/metric_information.json", "r", encoding="utf-8") as f:
            metric_info = json.load(f)

        metric_range = None
        for key, value in metric_info.get("metric_ranges", {}).items():
            if selected_metric.startswith(key):
                metric_range = value
                break

        if metric_range is not None:
            domain_min, domain_max = metric_range

            if domain_max == "∞":
                domain_max = df_filtered[selected_metric].max()
            if domain_min == "∞":
                domain_min = df_filtered[selected_metric].min()

            return domain_min, domain_max

    except FileNotFoundError:
        pass

    # Fallback: data-driven
    return df_filtered[selected_metric].min(), df_filtered[selected_metric].max()


def build_quadrant_chart(df: pd.DataFrame, plots_folder: Path):
    metric = "accuracy_mean"
    second_metric = "execution_time (s)_mean"

    # strip * from chart names
    df['chart_name'] = df['chart_name'].str.replace('*', '', regex=False)

    # ----------------------------------------------------------------
    # One point per (dataset, Approach, Configuration) — no aggregation
    # over datasets, just keep color and chart_name per group
    # ----------------------------------------------------------------
    plot_df = (
        df.groupby(["dataset", "Approach", "Configuration"])
        .agg(
            accuracy_mean=(metric, "mean"),
            **{second_metric: (second_metric, "mean")},
            color=("color", "first"),
            chart_name=("chart_name", "first"),
        )
        .reset_index()
    )

    num_datasets = plot_df["dataset"].nunique()
    print(f"Unique datasets: {num_datasets}")

    # ----------------------------------------------------------------
    # Quadrant thresholds — midpoint of axis domain
    # ----------------------------------------------------------------
    x_domain_min, x_domain_max = get_metric_domain(plot_df, second_metric)
    y_domain_min, y_domain_max = get_metric_domain(plot_df, metric)

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
    # Scatter points — one per (dataset, Approach, Configuration)
    # Legend: one entry per unique chart_name, colored by approach
    # ----------------------------------------------------------------
    seen_labels = set()  # track chart_names already added to legend

    for _, row in plot_df.iterrows():
        x = row[second_metric]
        y = row[metric]
        color = row["color"]
        chart_name = row["chart_name"]

        # Only pass label the first time we see this chart_name
        scatter_label = chart_name if chart_name not in seen_labels else None
        seen_labels.add(chart_name)

        ax.scatter(
            x, y,
            color=color,
            s=180,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
            label=scatter_label,
        )

        # Label each point with just the dataset name
        ax.annotate(
            row["dataset"],
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

    # Legend below figure
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
        fontsize=13
    )

    # ----------------------------------------------------------------
    # Axes styling
    # ----------------------------------------------------------------
    ax.set_xlim(0, x_domain_max)
    ax.set_ylim(y_domain_min, y_domain_max)

    ax.set_xlabel("Execution Time (s)", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)


    ax.grid(True, linestyle=":", alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------

    # Save PDF
    plt.savefig(plots_folder / f"cell_quadrant_chart.pdf")
    plt.close()