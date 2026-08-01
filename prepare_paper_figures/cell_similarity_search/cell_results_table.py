import pandas as pd
from pathlib import Path

import config_helpers as h

def create_results_table(df: pd.DataFrame, results_folder: Path):
    for metric in ("accuracy", "MAP"):
        if metric not in df.columns:
            print(f"WARNING: {metric} not found in cell_similarity_search data, skipping")
            continue

        # only keep columns needed for the results table
        filtered_task_df = df[['Approach', 'Configuration', 'task', 'dataset', metric, 'chart_name']]

        # create results table, dataset names as rows, approaches as columns
        # strip * from chart names
        filtered_task_df['chart_name'] = filtered_task_df['chart_name'].str.replace('*', '', regex=False)

        # pivot the dataframe to have datasets as rows and approaches as columns
        results_table = filtered_task_df.pivot_table(
            index='dataset',
            columns='chart_name',
            values=metric,
            aggfunc='mean'
        )
        results_table = h.sort_columns_by_chart_name(results_table)

        # Reorder the rows: dirty first, clean second
        dataset_order = ['s2abel@dirty', 's2abel@clean']
        results_table = results_table.reindex(dataset_order)

        # Compute mean over datasets and add as a new row
        mean_row = results_table.mean().to_frame().T  # convert Series to single-row DataFrame
        mean_row.index = ['Mean']  # set row label
        results_table = pd.concat([results_table, mean_row])

        print(results_table)

        # Generate LaTeX using tabular* for double-column layout
        with open(results_folder / f"cell_table_{metric.lower()}.tex", "w") as f:
            f.write(
                "\\begin{table*}[t]\n"
                "\\centering\n"
                f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l " + "c " * len(results_table.columns) + "@{}}\n"
                "\\toprule\n"
                + "Dataset & " + " & ".join(results_table.columns) + " \\\\\n"
                "\\midrule\n"
            )

            # Round metric values in the pivot table before the row loop
            results_table = results_table.round(2)

            # Add rows
            for idx, row in results_table.iterrows():
                if idx == 'Mean':
                    f.write("\\midrule\n")

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
                        values.append("*")  # replace NaN
                    else:
                        val_str = f"{v:.2f}"  # already rounded, this just formats as string
                        if v == row_max:
                            val_str = f"\\textbf{{{val_str}}}"       # bold all max values
                        elif row_second is not None and v == row_second:
                            if v != 0.00:
                                val_str = f"\\underline{{{val_str}}}"    # underline all second-highest
                        values.append(val_str)

                values_str = " & ".join(values)
                f.write(f"{idx} & {values_str} \\\\\n")

            f.write("\\bottomrule\n\\end{tabular*}\n")
            metric_label = "Retrieval Accuracy" if metric == "accuracy" else metric
            f.write(f"\\caption{{Cell Level Semantic Retrieval: {metric_label} per dataset for all approaches.}}\n")
            f.write(f"\\label{{tab:cell_results_{metric.lower()}}}\n")
            f.write("\\end{table*}\n")
