import re
import hashlib
import pandas as pd
from pathlib import Path

from benchmark_src.results_processing import results_helper

def sanitize_sheet_name(name: str) -> tuple[str, str | None]:
    """
    Returns a valid Excel sheet name and optionally a hash (if name was truncated).
    Truncates intelligently at @@ if present.
    """
    # Replace invalid characters
    name_clean = re.sub(r'[\[\]\:\*\?\/\\]+', ' ', str(name))
    name_clean = re.sub(r'\s+', ' ', name_clean).strip()

    max_length = 31
    hash_suffix = None

    if len(name_clean) > max_length:
        # Name too long -> compute short hash
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:6]

        # Try to truncate after @@
        if '@@' in name_clean:
            prefix = name_clean.split('@@')[0]
        else:
            prefix = name_clean

        # Ensure prefix + '_' + hash fits Excel limit
        max_prefix_length = max_length - len(hash_suffix) - 1
        if len(prefix) > max_prefix_length:
            prefix = prefix[:max_prefix_length]

        sheet_name = f"{prefix}_{hash_suffix}"
    else:
        sheet_name = name_clean

    return sheet_name, hash_suffix


def create_excel_files_per_dataset(averaged_data_df: pd.DataFrame, results_folder, mean_decimals=4):
    """
    Writes the results to Excel files, creating one file per task and one
    sheet per dataset within each file.

        Args:
            averaged_data_df (pd.DataFrame): DataFrame containing the results (one row per approach/configuration/dataset).
            results_folder (str): Directory to save the Excel files
            mean_decimals (int): Number of decimal places for metric values
    """
    unique_tasks = averaged_data_df['task'].unique()

    for task in unique_tasks:
        # Filter data for the current task
        task_df = averaged_data_df[averaged_data_df['task'] == task].copy()

        # Drop _std columns (each approach/configuration/dataset has exactly one result,
        # so std is always undefined) and present the _mean columns as plain metric values.
        std_cols = [col for col in task_df.columns if col.endswith('_std')]
        task_df = task_df.drop(columns=std_cols)
        task_df = task_df.rename(columns={
            col: col[:-len('_mean')] for col in task_df.columns if col.endswith('_mean')
        })
        for col in task_df.columns:
            if pd.api.types.is_float_dtype(task_df[col]):
                task_df[col] = task_df[col].round(mean_decimals)

        # create folder per task
        task_folder = results_folder / results_helper.to_slug(task)
        task_folder.mkdir(exist_ok=True)

        # save task df to disk
        task_df.to_csv(task_folder / f"{task}_results.csv", index=False)

        excel_filename = task_folder / f"{task}_results.xlsx"

        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            unique_datasets = task_df['dataset'].unique()

            for dataset in unique_datasets:
                dataset_df = task_df[task_df['dataset'] == dataset].copy()
                dataset_df = dataset_df.dropna(axis=1, how="all")
                dataset_df = dataset_df.drop(columns=["task", "dataset"])

                # Sanitize sheet name
                sheet_name, _ = sanitize_sheet_name(dataset)

                # Write the full dataset name in the first row, leave second row empty, then df
                dataset_df.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)
                worksheet = writer.sheets[sheet_name]
                worksheet.write(0, 0, dataset)  # write full dataset name in first row

                # Adjust column widths
                for column in dataset_df:
                    column_length = max(dataset_df[column].astype(str).map(len).max(), len(column)) + 2
                    col_idx = dataset_df.columns.get_loc(column)
                    worksheet.set_column(col_idx, col_idx, column_length)

        print(f"Excel file created for task '{task}': {excel_filename}")