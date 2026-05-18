import pandas as pd
from pathlib import Path

def create_results_table_binary_classification(df: pd.DataFrame, results_folder: Path):
    # only keep relevant columns
    filtered_task_df = df[['Approach', 'Configuration', 'task', 'dataset',  "XGBoost_roc_auc_score (↑)_mean", 'chart_name']]   

    # create results table, dataset names as rows, approaches as columns (order is hytrel, sap_rpt_oss, sentence_transformer, tabicl)

    # pivot the dataframe to have datasets as rows and approaches as columns
    results_table = filtered_task_df.pivot_table(
        index='dataset',
        columns='chart_name',
        values='XGBoost_roc_auc_score (↑)_mean',
        aggfunc='mean'
    )

    # rename column "Baseline" to "XGBoost"
    results_table = results_table.rename(columns={"Baseline": "XGBoost"})
    results_table = results_table.rename(columns={"GritLM": "GritLM*"})
    results_table = results_table.rename(columns={"HyTrel": "HyTrel*"})
    results_table = results_table.rename(columns={"IBM Granite R2": "Granite*"})
    results_table = results_table.rename(columns={"MiniLM": "MiniLM*"})
    results_table = results_table.rename(columns={"SAP-RPT-1": "SAP-RPT"})
    results_table = results_table.rename(columns={"SAP-RPT-1_row": "SAP-RPT*"})

    # tmp: add column for tabula non-row: (has bug, TODO)
    #results_table['TabuLa-8B'] = "TODO"

    # Compute mean over datasets and add as a new row
    mean_row = results_table.mean().to_frame().T  # convert Series to single-row DataFrame
    mean_row.index = ['Mean']  # set row label
    results_table = pd.concat([results_table, mean_row])

    # Replace '_row' with '*' in all column names that end with '_row'
    results_table.rename(
        columns={col: col.replace("_row", "*") for col in results_table.columns if col.endswith("_row")},
        inplace=True
    )

    # Escape underscores in the first column (dataset names)
    results_table.index = results_table.index.str.replace("_", r"\_", regex=False)
    results_table.index = results_table.index.map(
        lambda s: (
            (s[:19].rstrip("-").rstrip("_") + ".")   # cut to 19, remove trailing '-', add dot
            if len(s) > 19
            else s         
        )
    )

    print(results_table)

    # Generate LaTeX using tabular* for double-column layout
    with open(results_folder / "tabular_prediction_table_binary.tex", "w") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\footnotesize\n"
            "\\centering\n"
            # tabular* with @{\extracolsep{\fill}} and dynamic number of columns
            f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l " + " ".join(["c"] * len(results_table.columns)) + " @{{}}}\n"
            "\\hline\n"
            + "Dataset & " + " & ".join(results_table.columns) + " \\\\\n"
            "\\hline\n"
        )

        # Round MRR values in the pivot table before the row loop
        results_table = results_table.round(2)

        # Add rows
        for idx, row in results_table.iterrows():
            # Collect numeric values, ignoring NaNs
            numeric_values = [v for v in row if pd.notna(pd.to_numeric(v, errors='coerce'))]
            if numeric_values:
                row_max = max(numeric_values)
                # second-highest is max of remaining values
                remaining = [v for v in numeric_values if v < row_max]
                row_second = max(remaining) if remaining else None
            else:
                row_max = row_second = None

            values = []
            for v in row:
                if isinstance(v, str):
                    values.append(v)
                elif pd.isna(v):
                    values.append("-")  # replace NaN
                else:
                    val_str = f"{v:.2f}"  # already rounded, this just formats as string
                    if v == row_max:
                        val_str = f"\\textbf{{{val_str}}}"       # bold all max values
                    elif row_second is not None and v == row_second:
                        val_str = f"\\underline{{{val_str}}}"    # underline all second-highest
                    values.append(val_str)

            values_str = " & ".join(values)
            f.write(f"{idx} & {values_str} \\\\\n")


        f.write("\\hline\n\\end{tabular*}\n")
        f.write("\\caption{Tabular Prediction: Binary Classification Results per Dataset measured by ROC AUC Score (higher is better). * indicates that the results were obtained by training XGBoost on top of row embeddings.}\n")
        f.write("\\label{tab:tabular_prediction_binary}\n")
        f.write("\\end{table*}\n")


def create_results_table_multiclass_classification(df: pd.DataFrame, results_folder: Path):
    # only keep relevant columns
    filtered_task_df = df[['Approach', 'Configuration', 'task', 'dataset',  "XGBoost_log_loss (↓)_mean", 'chart_name']]   

    # create results table, dataset names as rows, approaches as columns (order is hytrel, sap_rpt_oss, sentence_transformer, tabicl)

    # pivot the dataframe to have datasets as rows and approaches as columns
    results_table = filtered_task_df.pivot_table(
        index='dataset',
        columns='chart_name',
        values='XGBoost_log_loss (↓)_mean',
        aggfunc='mean'
    )

    # rename column "Baseline" to "XGBoost"
    results_table = results_table.rename(columns={"Baseline": "XGBoost"})
    results_table = results_table.rename(columns={"GritLM": "GritLM*"})
    results_table = results_table.rename(columns={"HyTrel": "HyTrel*"})
    results_table = results_table.rename(columns={"IBM Granite R2": "Granite*"})
    results_table = results_table.rename(columns={"MiniLM": "MiniLM*"})
    results_table = results_table.rename(columns={"SAP-RPT-1": "SAP-RPT"})
    results_table = results_table.rename(columns={"SAP-RPT-1_row": "SAP-RPT*"})

    # tmp: add column for tabula non-row: (has bug, TODO)
    #results_table['TabuLa-8B'] = "TODO"

    # Compute mean over datasets and add as a new row
    mean_row = results_table.mean().to_frame().T  # convert Series to single-row DataFrame
    mean_row.index = ['Mean']  # set row label
    results_table = pd.concat([results_table, mean_row])

    # Replace '_row' with '*' in all column names that end with '_row'
    results_table.rename(
        columns={col: col.replace("_row", "*") for col in results_table.columns if col.endswith("_row")},
        inplace=True
    )

    # Escape underscores in the first column (dataset names)
    results_table.index = results_table.index.str.replace("_", r"\_", regex=False)
    results_table.index = results_table.index.map(
        lambda s: (
            (s[:19].rstrip("-").rstrip("_") + ".")   # cut to 19, remove trailing '-', add dot
            if len(s) > 19
            else s         
        )
    )
    print(results_table)

    # Generate LaTeX using tabular* for double-column layout
    with open(results_folder / "tabular_prediction_table_multiclass.tex", "w") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\footnotesize\n"
            "\\centering\n"
            # tabular* with @{\extracolsep{\fill}} and dynamic number of columns
            f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l " + " ".join(["c"] * len(results_table.columns)) + " @{{}}}\n"
            "\\hline\n"
            + "Dataset & " + " & ".join(results_table.columns) + " \\\\\n"
            "\\hline\n"
        )

        # Round MRR values in the pivot table before the row loop
        results_table = results_table.round(2)

        # Add rows
        for idx, row in results_table.iterrows():
            # Collect numeric values, ignoring NaNs
            numeric_values = [v for v in row if pd.notna(pd.to_numeric(v, errors='coerce'))]
            if numeric_values:
                row_max = min(numeric_values)
                # second-highest is max of remaining values
                remaining = [v for v in numeric_values if v > row_max]
                row_second = min(remaining) if remaining else None
            else:
                row_max = row_second = None

            values = []
            for v in row:
                if isinstance(v, str):
                    values.append(v)
                elif pd.isna(v):
                    values.append("-")  # replace NaN
                else:
                    val_str = f"{v:.2f}"  # already rounded, this just formats as string
                    if v == row_max:
                        val_str = f"\\textbf{{{val_str}}}"       # bold all max values
                    elif row_second is not None and v == row_second:
                        val_str = f"\\underline{{{val_str}}}"    # underline all second-highest
                    values.append(val_str)

            values_str = " & ".join(values)
            f.write(f"{idx} & {values_str} \\\\\n")


        f.write("\\hline\n\\end{tabular*}\n")
        f.write("\\caption{Tabular Prediction: Multiclass Classification Results per Dataset measured by Log Loss Score (lower is better!). * indicates that the results were obtained by training XGBoost on top of row embeddings.}\n")
        f.write("\\label{tab:tabular_prediction_multiclass}\n")
        f.write("\\end{table*}\n")


def create_results_table_regression(df: pd.DataFrame, results_folder: Path):
    # only keep relevant columns
    filtered_task_df = df[['Approach', 'Configuration', 'task', 'dataset',  "XGBoost_rmse (↓)_mean", 'chart_name']]   

    # create results table, dataset names as rows, approaches as columns (order is hytrel, sap_rpt_oss, sentence_transformer, tabicl)

    # pivot the dataframe to have datasets as rows and approaches as columns
    results_table = filtered_task_df.pivot_table(
        index='dataset',
        columns='chart_name',
        values='XGBoost_rmse (↓)_mean',
        aggfunc='mean'
    )

    # rename column "Baseline" to "XGBoost"
    results_table = results_table.rename(columns={"Baseline": "XGBoost"})
    results_table = results_table.rename(columns={"GritLM": "GritLM*"})
    results_table = results_table.rename(columns={"HyTrel": "HyTrel*"})
    results_table = results_table.rename(columns={"IBM Granite R2": "Granite*"})
    results_table = results_table.rename(columns={"MiniLM": "MiniLM*"})
    results_table = results_table.rename(columns={"SAP-RPT-1": "SAP-RPT"})
    results_table = results_table.rename(columns={"SAP-RPT-1_row": "SAP-RPT*"})

    # tmp: add column for tabula non-row: (has bug, TODO)
    #results_table['TabuLa-8B'] = "TODO"

    # Compute mean over datasets and add as a new row
    mean_row = results_table.mean().to_frame().T  # convert Series to single-row DataFrame
    mean_row.index = ['Mean']  # set row label
    results_table = pd.concat([results_table, mean_row])

    # Replace '_row' with '*' in all column names that end with '_row'
    results_table.rename(
        columns={col: col.replace("_row", "*") for col in results_table.columns if col.endswith("_row")},
        inplace=True
    )

    # Escape underscores in the first column (dataset names)
    results_table.index = results_table.index.str.replace("_", r"\_", regex=False)
    results_table.index = results_table.index.map(
        lambda s: (
            (s[:19].rstrip("-").rstrip("_") + ".")   # cut to 19, remove trailing '-', add dot
            if len(s) > 19
            else s         
        )
    )
    print(results_table)

    # Generate LaTeX using tabular* for double-column layout
    with open(results_folder / "tabular_prediction_regression.tex", "w") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\footnotesize\n"
            "\\centering\n"
            # tabular* with @{\extracolsep{\fill}} and dynamic number of columns
            f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l " + " ".join(["r"] * len(results_table.columns)) + " @{{}}}\n"
            "\\hline\n"
            + "Dataset & " + " & ".join(results_table.columns) + " \\\\\n"
            "\\hline\n"
        )

        # Round MRR values in the pivot table before the row loop
        results_table = results_table.round(2)

        # Add rows
        for idx, row in results_table.iterrows():
            # Collect numeric values, ignoring NaNs
            numeric_values = [v for v in row if pd.notna(pd.to_numeric(v, errors='coerce'))]
            if numeric_values:
                row_max = min(numeric_values)
                # second-highest is max of remaining values
                remaining = [v for v in numeric_values if v > row_max]
                row_second = min(remaining) if remaining else None
            else:
                row_max = row_second = None

            values = []
            for v in row:
                if isinstance(v, str):
                    values.append(v)
                elif pd.isna(v):
                    values.append("-")  # replace NaN
                else:
                    val_str = f"{v:.2f}"  # already rounded, this just formats as string
                    if v == row_max:
                        val_str = f"\\textbf{{{val_str}}}"       # bold all max values
                    elif row_second is not None and v == row_second:
                        val_str = f"\\underline{{{val_str}}}"    # underline all second-highest
                    values.append(val_str)

            values_str = " & ".join(values)
            f.write(f"{idx} & {values_str} \\\\\n")


        f.write("\\hline\n\\end{tabular*}\n")
        f.write("\\caption{Tabular Prediction: Regression results per Dataset measured by RMSE (lower is better!). * indicates that the results were obtained by training XGBoost on top of row embeddings.}\n")
        f.write("\\label{tab:tabular_prediction_regression}\n")
        f.write("\\end{table*}\n")