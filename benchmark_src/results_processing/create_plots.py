import hashlib
import logging
from pathlib import Path
from typing import Annotated

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from omegaconf import ListConfig

from benchmark_src.results_processing import results_helper
from benchmark_src.results_processing.metric_info import get_metric_domain
from benchmark_src.utils import cfg_utils

logger = logging.getLogger(__name__)

# Fallback for (Approach, Configuration) pairs missing from configs/color_mapping.yaml,
# cycled by sorted index so colors stay stable across runs given the same input data.
FALLBACK_PALETTE = [plt.get_cmap("tab20")(i) for i in range(20)]


def _primary_metrics_for_task(task: str) -> list[str]:
    """
    Returns the metric(s) to plot for a task, taken from that task's elo_metric
    (configs/task/<task>.yaml). predictive_ml declares a list of subtask metrics
    (binary/multiclass/regression); every other task declares a single metric.
    """
    try:
        task_cfg = cfg_utils.load_task_config(task)
    except FileNotFoundError:
        logger.debug(f"No task config found for {task}, skipping general plots for it")
        return []

    metric = task_cfg.get("elo_metric")
    if metric is None:
        return []
    if isinstance(metric, ListConfig):
        return list(metric)
    return [metric]


def _assign_chart_style(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a chart_name/color per (Approach, Configuration) pair from
    configs/approach_plotting.yaml (shared with prepare_paper_figures, so both
    pipelines render the same approach with the same name/color) wherever a pair
    is listed there with that field set.

    A pair missing a curated name falls back to the plain approach name (or the
    approach name plus a short config hash, if that approach ran with multiple
    configurations); a pair missing a curated color falls back to a palette color
    assigned by sorted index. Both fallbacks are reported in a warning so the
    missing entries can be added.
    """
    df = df.copy()
    configs_per_approach = df.groupby("Approach")["Configuration"].nunique()
    curated = cfg_utils.load_approach_plotting()
    unique_pairs = sorted(df[["Approach", "Configuration"]].drop_duplicates().itertuples(index=False, name=None))

    def fallback_name(approach, configuration):
        if configs_per_approach[approach] == 1:
            return approach
        config_hash = hashlib.md5(str(configuration).encode()).hexdigest()[:6]
        return f"{approach}-{config_hash}"

    name_map, color_map = {}, {}
    missing_names, missing_colors = [], []
    for approach, configuration in unique_pairs:
        key = (approach, configuration)
        entry = curated.get(key, {})

        if entry.get("name"):
            name_map[key] = entry["name"]
        else:
            name_map[key] = fallback_name(approach, configuration)
            missing_names.append(key)

        if entry.get("color"):
            color_map[key] = entry["color"]
        else:
            color_map[key] = FALLBACK_PALETTE[len(missing_colors) % len(FALLBACK_PALETTE)]
            missing_colors.append(key)

    if missing_names:
        logger.warning(
            f"No curated name for {len(missing_names)} (Approach, Configuration) pair(s), "
            f"using the raw approach name instead. Add them to configs/approach_plotting.yaml "
            f"to fix: {missing_names}"
        )
    if missing_colors:
        logger.warning(
            f"No curated color for {len(missing_colors)} (Approach, Configuration) pair(s), "
            f"using a fallback palette color instead. Add them to configs/approach_plotting.yaml "
            f"to fix: {missing_colors}"
        )

    df["chart_name"] = df.apply(lambda row: name_map[(row["Approach"], row["Configuration"])], axis=1)
    df["color"] = df.apply(lambda row: color_map[(row["Approach"], row["Configuration"])], axis=1)
    return df


def _datasets_with_full_coverage(task_df: pd.DataFrame) -> pd.DataFrame:
    """
    Restricts task_df to datasets where every chart_name present anywhere in task_df
    has a (non-null, since task_df is already metric-dropna'd) value — so an aggregated
    mean is comparable across approaches instead of each being averaged over a
    different, approach-specific set of datasets.
    """
    n_charts = task_df["chart_name"].nunique()
    charts_per_dataset = task_df.groupby("dataset")["chart_name"].nunique()
    full_datasets = charts_per_dataset[charts_per_dataset == n_charts].index
    return task_df[task_df["dataset"].isin(full_datasets)]


def _create_bar_chart(task_df: pd.DataFrame, metric: str, task: str, plots_folder: Path) -> None:
    num_datasets = task_df["dataset"].nunique()

    agg_df = (
        task_df.groupby("chart_name")
        .agg(metric_mean=(metric, "mean"), color=("color", "first"), Approach=("Approach", "first"))
        .reset_index()
    )

    # Sort by the raw Approach name (case-sensitive), matching the paper plots'
    # convention (e.g. row_sim_plot_aggregated.py relies on groupby's default sort
    # over ['Approach', 'Configuration']) rather than the curated display name.
    agg_df = agg_df.sort_values("Approach")

    domain_min, domain_max = get_metric_domain(agg_df["metric_mean"], metric)

    fig, ax = plt.subplots(figsize=(max(6, len(agg_df) * 0.9), 5))
    bars = ax.bar(agg_df["chart_name"], agg_df["metric_mean"], color=agg_df["color"])
    ax.set_xticks(range(len(agg_df)))
    ax.set_xticklabels(agg_df["chart_name"], rotation=30, ha="right")
    ax.set_ylim(domain_min, domain_max)
    ax.set_ylabel(metric)
    ax.set_title(f"{task}: {metric} (mean across {num_datasets} datasets)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    label_offset = (domain_max - domain_min) * 0.02
    for bar, value in zip(bars, agg_df["metric_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{value:.2f}",
            ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()

    filename = f"{task}_{results_helper.to_slug(metric)}_bar.png"
    plt.savefig(plots_folder / filename)
    plt.close(fig)


def _create_per_dataset_bar_chart(task_df: pd.DataFrame, metric: str, task: str, plots_folder: Path) -> None:
    """
    Grouped bar chart with one group per dataset (plus a trailing "Mean" group
    averaging across datasets) and one bar per (approach, configuration) within
    each group. task_df must already be restricted to full-coverage datasets
    (see _datasets_with_full_coverage) so every group has the same set of bars.
    """
    agg_df = (
        task_df.groupby("chart_name")
        .agg(metric_mean=(metric, "mean"), color=("color", "first"), Approach=("Approach", "first"))
        .reset_index()
        .sort_values("Approach")
    )
    chart_names = agg_df["chart_name"].tolist()
    color_map = dict(zip(agg_df["chart_name"], agg_df["color"]))

    mean_df = agg_df.rename(columns={"metric_mean": metric})[["chart_name", metric]].copy()
    mean_df["dataset"] = "Mean"
    plot_df = pd.concat([task_df[["dataset", "chart_name", metric]], mean_df], ignore_index=True)

    datasets = sorted(task_df["dataset"].unique()) + ["Mean"]

    domain_min, domain_max = get_metric_domain(agg_df["metric_mean"], metric)

    n_datasets = len(datasets)
    n_methods = len(chart_names)
    gap = 0.025
    bar_width = (0.8 - gap * (n_methods - 1)) / n_methods
    x = np.arange(n_datasets)
    total_group_width = n_methods * bar_width + (n_methods - 1) * gap
    group_centers = x + total_group_width / 2

    fig, ax = plt.subplots(figsize=(max(8, n_datasets * n_methods * 0.35), 6))

    label_offset = (domain_max - domain_min) * 0.02
    for i, chart_name in enumerate(chart_names):
        method_df = (
            plot_df[plot_df["chart_name"] == chart_name]
            .set_index("dataset")
            .reindex(datasets)
        )
        positions = x + i * (bar_width + gap)
        bars = ax.bar(positions, method_df[metric], width=bar_width, color=color_map[chart_name], label=chart_name)

        for dataset_name, bar in zip(datasets, bars):
            height = bar.get_height()
            if pd.isna(height):
                continue
            if dataset_name == "Mean":
                bar.set_alpha(0.8)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + label_offset,
                f"{height:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylim(domain_min, domain_max)
    ax.set_ylabel(metric)
    ax.set_title(f"{task}: {metric} per dataset")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(6, n_methods), frameon=False, fontsize=9)

    plt.tight_layout()

    filename = f"{task}_{results_helper.to_slug(metric)}_bar_per_dataset.png"
    plt.savefig(plots_folder / filename)
    plt.close(fig)


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> pipe-delimited markdown table (avoids a `tabulate` dependency)."""
    headers = [df.index.name or ""] + [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in df.iterrows():
        cells = [str(idx)] + ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _create_results_table(task_df: pd.DataFrame, metric: str, task: str, plots_folder: Path) -> None:
    results_table = task_df.pivot_table(index="dataset", columns="chart_name", values=metric, aggfunc="mean")
    mean_row = results_table.mean().to_frame().T
    mean_row.index = ["Mean"]
    results_table = pd.concat([results_table, mean_row]).round(4)
    results_table.index.name = "dataset"

    basename = f"{task}_{results_helper.to_slug(metric)}_results_table"
    results_table.to_csv(plots_folder / f"{basename}.csv")
    (plots_folder / f"{basename}.md").write_text(_to_markdown_table(results_table))


def run(results_folder_name: str) -> None:
    """
    Generate a generic per-task bar chart + results table for every task with results
    in results_folder_name, using each task's elo_metric as the plotted metric.
    Unlike prepare_paper_figures/main.py, this needs no curated approach/config mapping,
    so it can run against any results folder right after it's produced.
    """
    results_folder = Path(results_folder_name)
    assert results_folder.exists(), f"Could not find results folder at {results_folder}"

    csv_path = results_folder / "all_results.csv"
    if not csv_path.exists():
        logger.debug(f"No all_results.csv found in {results_folder}, skipping general plots")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.debug(f"No results in {csv_path}, skipping general plots")
        return

    plots_folder = results_folder / "plots"
    plots_folder.mkdir(exist_ok=True)

    df = _assign_chart_style(df)

    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task]

        for metric in _primary_metrics_for_task(task):
            if metric not in task_df.columns:
                continue
            metric_df = task_df.dropna(subset=[metric])
            if metric_df.empty:
                continue

            common_df = _datasets_with_full_coverage(metric_df)
            if common_df.empty:
                logger.warning(
                    f"No dataset has {metric} results for every approach in task {task} — "
                    f"skipping the bar chart (each approach was evaluated on a different set "
                    f"of datasets, so no fair aggregate mean can be computed)"
                )
            else:
                logger.info(f"Creating general plots for task {task}, metric {metric}")
                _create_bar_chart(common_df, metric, task, plots_folder)
                if common_df["dataset"].nunique() > 1:
                    _create_per_dataset_bar_chart(common_df, metric, task, plots_folder)

            _create_results_table(metric_df, metric, task, plots_folder)


def main(
    results_folder_name: Annotated[str, typer.Argument(help="Path to the results folder")],
) -> None:
    run(results_folder_name)


if __name__ == "__main__":
    typer.run(main)
