"""Predictive ML: performance (ratio to XGBoost baseline) vs. dataset size (row count),
restricted to the in-context tabular foundation models (SAP-RPT-1, TabICL, TabPFN, TabDPT) —
this is where dataset-size sensitivity actually shows up; the generalist embedding
approaches (MiniLM, GritLM, IBM Granite, HyTrel, TabuLa-8B) have no significant
correlation with size and just add noise to the plot.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import config_helpers as h

# Binary/multiclass/regression datasets each populate exactly one of these
# ratio-to-baseline columns; coalescing them gives one comparable metric.
RATIO_METRICS = [
    "XGBoost_roc_auc_score (↑)_ratio_to_baseline",
    "XGBoost_log_loss (↓)_ratio_to_baseline",
    "XGBoost_rmse (↓)_ratio_to_baseline",
]

# base_name (chart_name with trailing '*' stripped) of the approaches to plot —
# both the native in-context mode and its '*' row-embedding counterpart.
TARGET_BASE_NAMES = {"SAP-RPT-1", "TabICL v2", "TabPFN v2.5", "TabDPT"}


def create_scatter(df: pd.DataFrame, plots_folder: Path):
    df = df.copy()

    df["base_name"] = df["chart_name"].str.replace(r"\*$", "", regex=True)
    df = df[df["base_name"].isin(TARGET_BASE_NAMES)]

    present_metrics = [m for m in RATIO_METRICS if m in df.columns]
    df["performance_ratio"] = df[present_metrics].bfill(axis=1).iloc[:, 0]
    df = df.dropna(subset=["performance_ratio"])

    df["n_rows"] = df["dataset"].map(h.PREDICTIVE_ML_DATASET_ROWS)
    missing = df.loc[df["n_rows"].isna(), "dataset"].unique()
    if len(missing) > 0:
        print(f"WARNING: no row count for datasets {list(missing)}, dropping from size scatter")
    df = df.dropna(subset=["n_rows"])

    plot_df = (
        df.groupby(["dataset", "Approach", "Configuration"])
        .agg(
            performance_ratio=("performance_ratio", "mean"),
            n_rows=("n_rows", "first"),
            color=("color", "first"),
            chart_name=("chart_name", "first"),
            marker=("marker", "first"),
        )
        .reset_index()
    )
    plot_df["performance_ratio"] *= 100

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, zorder=2)

    seen_labels = set()
    chart_names = sorted(plot_df["chart_name"].unique(), key=h.chart_name_sort_key)
    for chart_name in chart_names:
        group = plot_df[plot_df["chart_name"] == chart_name]
        color = group["color"].iloc[0]
        marker = group["marker"].iloc[0]
        # Native and '*' (row-embedding) variants share color/marker in the shared
        # approach_plotting.yaml; hollow out the '*' variant's points here so the
        # two are visually distinguishable in this plot specifically.
        is_starred = chart_name.endswith("*")
        scatter_kwargs = (
            dict(facecolors="none", edgecolors=color, linewidths=1.5)
            if is_starred else
            dict(color=color)
        )

        ax.scatter(
            group["n_rows"], group["performance_ratio"],
            marker=marker, s=60, alpha=0.85, zorder=4,
            label=chart_name if chart_name not in seen_labels else None,
            **scatter_kwargs,
        )
        seen_labels.add(chart_name)

        # Trend line: linear fit in log10(rows), needs >=2 distinct sizes
        if group["n_rows"].nunique() >= 2:
            log_x = np.log10(group["n_rows"])
            slope, intercept = np.polyfit(log_x, group["performance_ratio"], 1)
            x_line = np.linspace(log_x.min(), log_x.max(), 100)
            ax.plot(
                10 ** x_line, slope * x_line + intercept,
                color=color, linewidth=1.5, alpha=0.6, zorder=3,
                linestyle="--" if is_starred else "-",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Dataset Size (number of rows, log scale)", fontsize=14)
    ax.set_ylabel("Ratio [%] to XGBoost Baseline", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":", alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        fontsize=11,
        framealpha=0.7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=6,
    )

    plt.tight_layout()
    fig.savefig(plots_folder / "predictive_ml_size_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
