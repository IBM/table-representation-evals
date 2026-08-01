import json
import pandas as pd
from pathlib import Path

import config_helpers as h

DATASET_NAME_MAP = {
    "astronomical_objects": "Wikidata Astronomical Objects",
    "wikidata_books": "Wikidata Books",
}

VARIANT_NAME_MAP = {
    "only-text": "TEXT ONLY",
    "no_col_names": "NO COLUMN NAMES",
    "no_pid_in_col_names": "SIMPLER COLUMN NAMES (NO PID)",
    "no_genre": "NO GENRE COLUMN",
    "only_five_cols": "ONLY 5 COLUMNS",
}

# Base dataset -> ordered list of variants to show below it as hooked sub-rows
# (None = the base/original dataset itself). Matches configs/global_datasets.yaml's
# row_triplet_evaluation_datasets entries; order here is curated, not alphabetical.
DATASET_GROUPS = [
    ("astronomical_objects", [None, "only-text"]),
    ("wikidata_books", [None, "no_col_names", "no_pid_in_col_names", "no_genre", "only_five_cols"]),
]

# cache/datasets/more_similar_than holds one dataset_information.json per dataset/variant,
# with the ground-truth column count for that variant (task predates its rename to
# row_triplet_evaluation, so the cache directory name is still the old one).
MORE_SIMILAR_THAN_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "datasets" / "more_similar_than"


def get_num_cols(base: str, variant: str | None) -> int:
    subdir = "original" if variant is None else f"@{variant}"
    info_path = MORE_SIMILAR_THAN_CACHE_DIR / base / subdir / "dataset_information.json"
    with open(info_path) as f:
        info = json.load(f)
    return info["input_table_num_cols"]


def format_row_label(base: str, variant: str | None) -> str:
    if variant is None:
        return DATASET_NAME_MAP.get(base, base.replace("_", " ").title())

    variant_pretty = VARIANT_NAME_MAP.get(variant, variant.replace("_", "-").upper())
    return f"\\ensuremath{{\\hookrightarrow}} \\textsubscript{{{variant_pretty}}}"


def create_results_table(df: pd.DataFrame, results_folder: Path):
    metric = "accuracy"
    # only keep columns needed for the results table
    filtered_task_df = df[['Approach', 'Configuration', 'task', 'dataset', metric, 'chart_name']]

    # pivot the dataframe to have datasets as rows and approaches as columns
    results_table = filtered_task_df.pivot_table(
        index='dataset',
        columns='chart_name',
        values=metric,
        aggfunc='mean'
    )
    results_table = h.sort_columns_by_chart_name(results_table)

    # Compute mean over datasets and add as a new row
    mean_row = results_table.mean().to_frame().T  # convert Series to single-row DataFrame
    mean_row.index = ['Mean']  # set row label
    results_table = pd.concat([results_table, mean_row])

    # Round values before formatting
    results_table = results_table.round(2)

    n_approach_cols = len(results_table.columns)

    def format_value_row(row: pd.Series) -> str:
        numeric_values = [v for v in row if pd.notna(v)]
        if numeric_values:
            row_max = max(numeric_values)
            remaining = [v for v in numeric_values if v < row_max]
            row_second = max(remaining) if remaining else None
        else:
            row_max = row_second = None

        values = []
        for v in row:
            if pd.isna(v):
                values.append("-")
            else:
                val_str = f"{v:.2f}"
                if v == row_max:
                    val_str = f"\\textbf{{{val_str}}}"
                elif row_second is not None and v == row_second:
                    val_str = f"\\underline{{{val_str}}}"
                values.append(val_str)
        return " & ".join(values)

    with open(results_folder / "triplet_table.tex", "w") as f:
        f.write(
            "\\begin{table*}[h]\n"
            "\\centering\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            f"\\begin{{tabular}}{{lr{'c' * n_approach_cols}}}\n"
            "\\toprule\n"
            "Dataset & \\#Cols & " + " & ".join(results_table.columns) + " \\\\\n"
            "\\midrule\n"
        )

        for group_idx, (base, variants) in enumerate(DATASET_GROUPS):
            if group_idx > 0:
                f.write("\\midrule\n")
            for variant in variants:
                dataset_key = base if variant is None else f"{base}@{variant}"
                if dataset_key not in results_table.index:
                    continue
                row_label = format_row_label(base, variant)
                num_cols = get_num_cols(base, variant)
                values_str = format_value_row(results_table.loc[dataset_key])
                f.write(f"{row_label} & {num_cols} & {values_str} \\\\\n")

        f.write("\\midrule\n")
        f.write(f"Mean & - & {format_value_row(results_table.loc['Mean'])} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}%\n}\n")
        f.write("\\caption{Triplet-Based Row Embedding Evaluations: Accuracy per dataset for all approaches. - indicates that the approach could not be run on the dataset, mostly due to memory constraints.}\n")
        f.write("\\label{tab:triplet_per_dataset}\n")
        f.write("\\vspace{-1.5em}\n")
        f.write("\\end{table*}\n")
