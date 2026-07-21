import pandas as pd
from pathlib import Path

METRIC = "mean_mrr"

TASK_LABEL_TITLE = {
    "column_mapping": "Schema Linking",
    "cell2column_mapping": "Value Linking",
}


def create_results_table(df: pd.DataFrame, results_folder: Path, task_label: str):
    filtered_task_df = df[["Approach", "Configuration", "task", "dataset", METRIC, "chart_name"]]

    results_table = filtered_task_df.pivot_table(
        index="dataset",
        columns="chart_name",
        values=METRIC,
        aggfunc="mean",
    )

    mean_row = results_table.mean().to_frame().T
    mean_row.index = ["Mean"]
    results_table = pd.concat([results_table, mean_row])

    print(results_table)

    title = TASK_LABEL_TITLE.get(task_label, task_label)

    with open(results_folder / f"nl2_{task_label}_table.tex", "w") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            f"\\begin{{tabular*}}{{\\textwidth}}{{l" + "c" * len(results_table.columns) + "}\n"
            "\\hline\n"
            + "Dataset & " + " & ".join(results_table.columns) + " \\\\\n"
            "\\hline\n"
        )

        results_table = results_table.round(2)

        for idx, row in results_table.iterrows():
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

        f.write("\\hline\n\\end{tabular*}\n")
        f.write(f"\\caption{{{title}: MRR results per dataset for all approaches.}}\n")
        f.write(f"\\label{{tab:nl2_{task_label}}}\n")
        f.write("\\end{table*}\n")
