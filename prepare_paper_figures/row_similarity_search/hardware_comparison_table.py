"""
Builds the row_similarity_search hardware-comparison table (A100 vs. V100 vs. CPU-only), showing
how approaches' performance and resource usage vary across hardware tiers. Reads three separate
results.json/all_results.csv trees rather than the single-results-folder flow the rest of
prepare_paper_figures/ uses, since this table joins across three separate hardware runs rather
than plotting one run's data.

Usage: python prepare_paper_figures/row_similarity_search/hardware_comparison_table.py
"""
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from benchmark_src.utils import cfg_utils

TASK = "row_similarity_search"

# Approaches deliberately excluded from the full V100/CPU runs on confirmed-OOM /
# confirmed-impractical grounds, with the note to print under the table.
KNOWN_EXCLUSIONS = {
    ("tabula_8b", "tabula_8b"): {
        "v100": "OOM",
        "cpu": "partial",
    },
    ("GritLM", "embedding_model=GritLM_GritLM-7B"): {
        "v100": "OOM",
        "cpu": "impractical",
    },
}


def _load(results_folder: Path) -> pd.DataFrame:
    df = pd.read_csv(results_folder / "all_results.csv")
    return df[df["task"] == TASK]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Approach", "Configuration"])
        .agg(
            n=("dataset", "nunique"),
            MAP=("MAP", "mean"),
            time_s=("execution_time (s)", "sum"),
            vram_mb=("peak_gpu_memory (MB)", "max"),
            ram_mb=("peak_memory (MB)", "max"),
        )
        .reset_index()
    )


def build_table(
    a100_folder: Annotated[str, typer.Option(help="Original/A100 results folder")] = "results/revision_experiments",
    v100_folder: Annotated[str, typer.Option(help="V100 GPU results folder")] = "results/hardware_comparison_v100_gpu",
    cpu_folder: Annotated[str, typer.Option(help="CPU-only results folder")] = "results/hardware_comparison_v100_cpu",
    output_folder: Annotated[str, typer.Option(help="Where to write the .tex file")] = "prepare_paper_figures/generated_figures/hardware_comparison",
):
    a100 = _aggregate(_load(Path(a100_folder)))
    v100 = _aggregate(_load(Path(v100_folder)))
    cpu = _aggregate(_load(Path(cpu_folder)))

    a100 = a100.set_index(["Approach", "Configuration"])
    v100 = v100.set_index(["Approach", "Configuration"])
    cpu = cpu.set_index(["Approach", "Configuration"])

    name_mapping = {k: v["name"] for k, v in cfg_utils.load_approach_plotting().items() if "name" in v}

    # Sort rows alphabetically (case-insensitive) by A100's own chart_name, matching
    # the ordering convention used across prepare_paper_figures/.
    def _chart_name_sort_key(key: tuple[str, str]) -> str:
        approach, configuration = key
        return name_mapping.get(key, f"{approach} ({configuration})").rstrip("*").lower()

    rows = []
    for key in sorted(a100.index, key=_chart_name_sort_key):
        a100_row = a100.loc[key]
        approach, configuration = key
        chart_name = name_mapping.get(key, f"{approach} ({configuration})")

        v100_row = v100.loc[key] if key in v100.index else None
        cpu_row = cpu.loc[key] if key in cpu.index else None
        exclusion = KNOWN_EXCLUSIONS.get(key)

        v100_partial = v100_row is not None and v100_row["n"] < a100_row["n"]
        cpu_partial = cpu_row is not None and cpu_row["n"] < a100_row["n"]

        vram_a100_gb = f"{a100_row['vram_mb'] / 1024:.1f}" if pd.notna(a100_row["vram_mb"]) else "---"
        if v100_row is None:
            vram_v100_gb = "OOM" if exclusion else "---"
        elif v100_partial:
            vram_v100_gb = "OOM"
        else:
            vram_v100_gb = f"{v100_row['vram_mb'] / 1024:.1f}" if pd.notna(v100_row["vram_mb"]) else "---"

        if v100_row is not None:
            speedup_v100 = "---" if v100_partial else f"{v100_row['time_s'] / a100_row['time_s']:.2f}$\\times$"
        else:
            speedup_v100 = "---"

        if cpu_row is not None:
            speedup_cpu = "---" if cpu_partial else f"{cpu_row['time_s'] / a100_row['time_s']:.2f}$\\times$"
        elif exclusion and exclusion.get("cpu") == "impractical":
            speedup_cpu = "impractical"
        else:
            speedup_cpu = "---"

        if cpu_row is None or cpu_partial:
            ram_cpu_gb = "---"
        else:
            ram_cpu_gb = f"{cpu_row['ram_mb'] / 1024:.1f}" if pd.notna(cpu_row["ram_mb"]) else "---"

        rows.append({
            "Approach": chart_name,
            "MAP": f"{a100_row['MAP']:.3f}",
            "VRAM A100 (GB)": vram_a100_gb,
            "VRAM V100 (GB)": vram_v100_gb,
            "Speedup (A100 vs V100)": speedup_v100,
            "Speedup (A100 vs CPU)": speedup_cpu,
            "Peak RAM CPU (GB)": ram_cpu_gb,
        })

    table_df = pd.DataFrame(rows)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / "row_sim_hardware_comparison.tex"

    with open(out_path, "w") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} lcccccc @{}}\n"
            "\\toprule\n"
            "Approach & MAP & VRAM A100 (GB) & VRAM V100 (GB) & Speedup (A100 vs V100) & Speedup (A100 vs CPU) & Peak RAM CPU (GB) \\\\\n"
            "\\midrule\n"
        )
        for _, row in table_df.iterrows():
            f.write(
                f"{row['Approach']} & {row['MAP']} & "
                f"{row['VRAM A100 (GB)']} & {row['VRAM V100 (GB)']} & "
                f"{row['Speedup (A100 vs V100)']} & {row['Speedup (A100 vs CPU)']} & {row['Peak RAM CPU (GB)']} \\\\\n"
            )
        f.write(
            "\\bottomrule\n"
            "\\end{tabular*}\n"
            "\\caption{Row Similarity Search: hardware comparison across an A100 80GB, a V100 16GB, "
            "and CPU-only (2$\\times$ Intel Xeon Gold 5120, no GPU, 504GB RAM available). OOM in the "
            "VRAM column indicates the approach ran out of VRAM on that hardware, either before "
            "completing any dataset or partway through the largest datasets. --- in the Speedup or "
            "Peak RAM CPU column indicates the approach did not complete the same datasets as on the "
            "A100 on that hardware, so no like-for-like value is available (see technical report text "
            "for the per-dataset breakdown of partial completions). \\emph{impractical} indicates the "
            "approach was still running after several hours with no completed batch and was stopped "
            "rather than run to completion (see technical report text). Where reported, Peak RAM CPU "
            "is the highest system-memory usage observed across all 9 datasets on the CPU-only run; "
            "the host's 504GB RAM was never close to a limiting factor for any approach.}\n"
            "\\label{tab:row_sim_hardware_comparison}\n"
            "\\end{table*}\n"
        )

    print(f"Wrote {out_path}")
    print(table_df.to_string(index=False))


if __name__ == "__main__":
    typer.run(build_table)
