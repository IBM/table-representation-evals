import pandas as pd
from pathlib import Path

DATASET_NAME_MAP = {
    "astronomical_objects": "Wikidata Astronomical Objects",
    "wikidata_books": "Wikidata Books",
}

VARIANT_NAME_MAP = {
    "only-text": "TEXT",
    "no_col_names": "NO-COL-NAMES",
    "no_genre": "NO-GENRE-COL",
    "no_pid_in_col_names": "NO-PID-IN-COL",
    "only_five_cols": "ONLY-5-COLS",
}


def format_dataset_name(name: str) -> str:
    """
    Format dataset name using explicit mappings and LaTeX subscripts.
    """

    if "@" in name:
        base, variant = name.split("@", 1)
    else:
        base, variant = name, None

    # Map base dataset name
    base_pretty = DATASET_NAME_MAP.get(
        base,
        base.replace("_", " ").title()  # fallback
    )

    # Map variant
    if variant:
        variant_pretty = VARIANT_NAME_MAP.get(
            variant,
            variant.replace("_", "-").upper()  # fallback
        )

        return f"{base_pretty}\\textsubscript{{{variant_pretty}}}"

    return base_pretty



def create_results_table(df: pd.DataFrame, results_folder: Path):
    metric = "accuracy_mean"
    print(df)
    print(df.columns)
    # only keep columns 'Approach', 'Configuration', 'task', 'dataset', metric, 'MRR_std_mean', 'MAP_mean', 'MAP_std_mean', 'Precision_mean', 'Precision_std_mean', 'Recall_mean', 'Recall_std_mean', '# Runs'
    filtered_task_df = df[['Approach', 'Configuration', 'task', 'dataset',  metric, 'chart_name']]   
    
    # create results table, dataset names as rows, approaches as columns (order is hytrel, sap_rpt_oss, sentence_transformer, tabicl)

    # pivot the dataframe to have datasets as rows and approaches as columns
    results_table = filtered_task_df.pivot_table(
        index='dataset',
        columns='chart_name',
        values=metric,
        aggfunc='mean'
    )

    # Compute mean over datasets and add as a new row
    mean_row = results_table.mean().to_frame().T  # convert Series to single-row DataFrame
    mean_row.index = ['Mean']  # set row label
    results_table = pd.concat([results_table, mean_row])

    print(results_table)

    # Generate LaTeX using tabular* for double-column layout
    with open(results_folder / "triplet_table.tex", "w") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            f"\\begin{{tabular*}}{{\\textwidth}}{{l" + "c" * len(results_table.columns) + "}\n"
            "\\hline\n"
            + "Dataset & " + " & ".join(results_table.columns) + " \\\\\n"
            "\\hline\n"
        )

        # Round MRR values in the pivot table before the row loop
        results_table = results_table.round(2)

        # Add rows
        for idx, row in results_table.iterrows():
            # Collect numeric values, ignoring NaNs
            numeric_values = [v for v in row if pd.notna(v)]
            if numeric_values:
                row_max = max(numeric_values)
                # second-highest is max of remaining values
                remaining = [v for v in numeric_values if v < row_max]
                row_second = max(remaining) if remaining else None
            else:
                row_max = row_second = None

            values = []
            for v in row:
                if pd.isna(v):
                    values.append("-")  # replace NaN
                else:
                    val_str = f"{v:.2f}"  # already rounded, this just formats as string
                    if v == row_max:
                        val_str = f"\\textbf{{{val_str}}}"       # bold all max values
                    elif row_second is not None and v == row_second:
                        val_str = f"\\underline{{{val_str}}}"    # underline all second-highest
                    values.append(val_str)

            values_str = " & ".join(values)
            pretty_idx = format_dataset_name(idx)
            f.write(f"{pretty_idx} & {values_str} \\\\\n")



        f.write("\\hline\n\\end{tabular*}\n")
        f.write("\\caption{Triplet-Based Row Embedding Evaluations: Accuracy per dataset for all approaches. - indicates that the approach could not be run on the dataset, mostly due to memory constraints.}\n")
        f.write("\\label{tab:triplet_per_dataset}\n")
        f.write("\\end{table*}\n")