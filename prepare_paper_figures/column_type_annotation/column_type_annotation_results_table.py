import pandas as pd
from pathlib import Path

import config_helpers as h

METRIC = "macro_f1 (↑)"


def create_results_table(df: pd.DataFrame, results_folder: Path):
    filtered_task_df = df[["Approach", "Configuration", "task", "dataset", METRIC, "chart_name"]].copy()

    # use pretty dataset names (matches column_type_annotation_bar_plot.py) so raw
    # underscores (e.g. gittables_cta) never reach the LaTeX table
    filtered_task_df["dataset"] = filtered_task_df["dataset"].replace(h.COLUMN_TYPE_ANNOTATION_DATASET_NAMES)

    results_table = filtered_task_df.pivot_table(
        index="dataset",
        columns="chart_name",
        values=METRIC,
        aggfunc="mean",
    )
    results_table = h.sort_columns_by_chart_name(results_table)

    mean_row = results_table.mean().to_frame().T
    mean_row.index = ["Mean"]
    results_table = pd.concat([results_table, mean_row])

    print(results_table)

    with open(results_folder / "cta_table.tex", "w") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l " + "c " * len(results_table.columns) + "@{}}\n"
            "\\toprule\n"
            + "Dataset & " + " & ".join(results_table.columns) + " \\\\\n"
            "\\midrule\n"
        )

        results_table = results_table.round(2)

        for idx, row in results_table.iterrows():
            if idx == 'Mean':
                f.write("\\midrule\n")

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
                    values.append("*")
                else:
                    val_str = f"{v:.2f}"
                    if v == row_max:
                        val_str = f"\\textbf{{{val_str}}}"
                    elif row_second is not None and v == row_second:
                        if v != 0.00:
                            val_str = f"\\underline{{{val_str}}}"
                    values.append(val_str)

            values_str = " & ".join(values)
            f.write(f"{idx} & {values_str} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular*}\n")
        f.write("\\caption{Column Type Annotation: Macro F1 results per dataset for all approaches.}\n")
        f.write("\\label{tab:cta}\n")
        f.write("\\end{table*}\n")
