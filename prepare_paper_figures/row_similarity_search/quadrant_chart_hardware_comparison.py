"""
Builds MAP-vs-Execution-Time quadrant charts for row_similarity_search, one per
hardware tier (A100, V100, CPU-only), reusing quadrant_chart_row.build_quadrant_chart
against the three separate hardware-comparison results folders instead of the
single results folder prepare_paper_figures/main.py normally plots from.

Usage: python prepare_paper_figures/row_similarity_search/quadrant_chart_hardware_comparison.py
"""
import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

# quadrant_chart_row does a bare `import config_helpers`, which resolves only if
# prepare_paper_figures/ (not this file's own directory) is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_src.utils import cfg_utils
from row_similarity_search import quadrant_chart_row

TASK = "row_similarity_search"


def _load_task_df(results_folder: Path, color_mapping: dict, name_mapping: dict, marker_mapping: dict) -> pd.DataFrame:
    df = pd.read_csv(results_folder / "all_results.csv")
    df = df[df["task"] == TASK].copy()

    df["color"] = df.apply(
        lambda row: color_mapping.get((row["Approach"], row["Configuration"]), "#000000"), axis=1
    )
    df["chart_name"] = df.apply(
        lambda row: name_mapping.get((row["Approach"], row["Configuration"]), "TODO"), axis=1
    )
    df["marker"] = df.apply(
        lambda row: marker_mapping.get((row["Approach"], row["Configuration"]), "o"), axis=1
    )

    # Drop MRR so build_quadrant_chart's per-metric loop only produces the MAP chart.
    return df.drop(columns=["MRR"], errors="ignore")


def build_charts(
    a100_folder: Annotated[str, typer.Option(help="Original/A100 results folder")] = "results/revision_experiments",
    v100_folder: Annotated[str, typer.Option(help="V100 GPU results folder")] = "results/hardware_comparison_v100_gpu",
    cpu_folder: Annotated[str, typer.Option(help="CPU-only results folder")] = "results/hardware_comparison_v100_cpu",
    output_folder: Annotated[str, typer.Option(help="Where to write the chart PDFs")] = "prepare_paper_figures/generated_figures/hardware_comparison",
):
    approach_plotting = cfg_utils.load_approach_plotting()
    color_mapping = {key: entry["color"] for key, entry in approach_plotting.items() if "color" in entry}
    name_mapping = {key: entry["name"] for key, entry in approach_plotting.items() if "name" in entry}
    marker_mapping = {key: entry["marker"] for key, entry in approach_plotting.items() if "marker" in entry}

    hardware_folders = {
        "a100": Path(a100_folder),
        "v100": Path(v100_folder),
        "cpu": Path(cpu_folder),
    }
    output_folder = Path(output_folder)

    for hardware, folder in hardware_folders.items():
        df = _load_task_df(folder, color_mapping, name_mapping, marker_mapping)
        if df.empty:
            print(f"No {TASK} results in {folder}, skipping {hardware}.")
            continue

        missing_chart_names = df[df["chart_name"] == "TODO"]
        if not missing_chart_names.empty:
            missing = missing_chart_names[["Approach", "Configuration"]].drop_duplicates().to_dict("records")
            raise ValueError(f"Missing chart names for {hardware} run: {missing}")

        hardware_output_folder = output_folder / hardware
        hardware_output_folder.mkdir(parents=True, exist_ok=True)
        quadrant_chart_row.build_quadrant_chart(df, hardware_output_folder)
        print(f"Wrote {hardware} MAP-vs-time quadrant chart to {hardware_output_folder}")


if __name__ == "__main__":
    typer.run(build_charts)
