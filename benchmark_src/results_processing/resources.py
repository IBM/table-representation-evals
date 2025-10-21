import logging
import pandas as pd
from pathlib import Path
import ast

from benchmark_src.results_processing.excel_writer import sanitize_sheet_name
from benchmark_src.results_processing import results_helper

logger = logging.getLogger(__name__)

def gather_resources(results_folder: Path, detailed_results_folder: Path):
    all_resource_dfs = []
    for results_file in results_folder.rglob("*/resource_metrics_formatted.csv"):
        dataset_name, task_name, configuration_str = results_helper.get_setup_infos(results_file)
        resource_df = pd.read_csv(results_file, header=None)

        # Need tuple column names back as real tuples
        header_row = resource_df.iloc[0].tolist()
        tuple_column_names = [ast.literal_eval(col) if col.startswith('(') and col.endswith(')') else col for col in header_row]
        resource_df.columns = tuple_column_names
        resource_df = resource_df[1:].reset_index(drop=True)

        # Temporary solution to not re-run all experiments right now
        # Define a dictionary for replacements
        replacement_map = {
            'run_model_setup': 'model_setup',
            'run_similarity_search_based_on_row_embeddings': 'task_inference',
            'run_training_based_on_row_embeddings': 'model_setup',
            'run_inference_based_on_row_embeddings': 'task_inference'
        }

        # Apply the replacement to the column headers
        new_columns = []
        for col in resource_df.columns:
            if isinstance(col, tuple) and col[0] in replacement_map:
                new_columns.append((replacement_map[col[0]], col[1]))
            else:
                new_columns.append(col)
        resource_df.columns = new_columns

        resource_df["run"] = "TODO"
        resource_df["dataset"] = dataset_name
        resource_df["task"] = task_name
        resource_df["configuration"] = configuration_str

        all_resource_dfs.append(resource_df)

    # end if there are no results to gather
    if len(all_resource_dfs) == 0:
        print(f'No resource results found to be gathered, please check if resource_metrics_formatted.csv files were created in the folder: *{results_folder}*, if this is intentional just ignore this.')
        return

    combined_df = pd.concat(all_resource_dfs)
    combined_df = combined_df.reset_index(drop=True)

    # convert numeric columns back to numeric types
    for col in combined_df.columns:
        try:
            combined_df[col] = pd.to_numeric(combined_df[col]) 
        except ValueError:
            pass # keep string columns as strings

    # TODO group and aggregate different runs?
    #grouping_columns = ["approach", "configuration", "task", "dataset"]
    #grouped_data = combined_df.groupby(grouping_columns)
    unique_tasks = combined_df['task'].unique()

    for task in unique_tasks:
        # Filter data for the current task
        task_df = combined_df[combined_df['task'] == task].copy()
        
        # create folder per task
        task_folder = detailed_results_folder / results_helper.to_slug(task)
        task_folder.mkdir(exist_ok=True)

        excel_filename = task_folder / f"{task}_resources.xlsx"
        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            unique_datasets = task_df['dataset'].unique()

            for dataset in unique_datasets:
                # Filter data for the current dataset
                dataset_df = task_df[task_df['dataset'] == dataset].copy()
                dataset_df = dataset_df.drop(columns=["task", "dataset"])
                dataset_df = dataset_df.round(2)

                multi_index_cols = [x if isinstance(x, tuple) else ("Info", x) for x in dataset_df.columns]
                dataset_df.columns = pd.MultiIndex.from_tuples(multi_index_cols, names=["Stage", "Metric"])
                
                # excel worksheet names must be <= 31 characters and cannot include certain characters
                sheet_name, _ = sanitize_sheet_name(dataset)

                dataset_df.to_excel(writer, sheet_name=sheet_name)

                # adjust column width
                for col_idx, (stage, metric) in enumerate(dataset_df.columns):
                    column_length = len(metric) + 2 
                    
                    writer.sheets[sheet_name].set_column(col_idx, col_idx, column_length)

        print(f"Excel file created for task '{task}': {excel_filename}")