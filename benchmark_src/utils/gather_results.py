import hydra
from omegaconf import DictConfig
import pandas as pd
import json
from pathlib import Path
import re
import os
import numpy as np
import ast
import logging
import attrs
from hydra.core.config_store import ConfigStore

from benchmark_src.utils import benchmark_metrics

logger = logging.getLogger(__name__)

@attrs.define
class ResultsConfig:
    results_folder_name: str = "results"

ConfigStore.instance().store(name="results_config", node=ResultsConfig)

def to_slug(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("/", "-")
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w-]", "", s)
    return s

performance_cols = {
    'XGBoost_rmse (↓)': 'lower_is_better',
    'KNeighbors_rmse (↓)': 'lower_is_better',
    'LinearRegression_rmse (↓)': 'lower_is_better',
    'XGBoost_roc_auc_score (↑)': 'higher_is_better',
    'KNeighbors_roc_auc_score (↑)': 'higher_is_better',
    'XGBoost_log_loss (↓)': 'lower_is_better',
    'KNeighbors_log_loss (↓)': 'lower_is_better'
}


def save_results(cfg, metrics: dict):
    """
    Saves the results to disk.

        Args:
            - metrics: dictionary of task specific metrics
    """
    print(f"Dataset: {cfg.dataset_name}")

    results = {
    "task": cfg.task.task_name,
    "dataset": cfg.dataset_name,
    "approach": cfg.approach.approach_name,
    }

    results.update(metrics)

    try:
        with open("results.json", "w") as file:
            json.dump(results, file, indent=2)
    except (TypeError, OverflowError):
        logger.error(f"Received result dict that is not json serializable:")
        logger.error(f"Dict: {metrics}")

    print("Saved metrics to disk")

def create_excel_files_per_dataset(averaged_data_df: pd.DataFrame, results_folder, mean_decimals=4, std_decimals=4, tolerance=1e-9):
    """
    Writes the averaged results to Excel files, creating one file per task
    and one sheet per dataset within each file. Includes standard deviation
    in brackets behind each mean value.

        Args:
            averaged_data_df (pd.DataFrame): DataFrame containing the averaged results.
            results_folder (str): Directory to save the Excel files
            mean_decimals (int): Number of decimal places for mean values
            std_decimals (int): Number of decimal places for standard deviation values
            tolerance (float): The threshold below which a standard deviation is considered "nearly zero"
    """
    unique_tasks = averaged_data_df['task'].unique()

    for task in unique_tasks:
        # Filter data for the current task
        task_df = averaged_data_df[averaged_data_df['task'] == task].copy()

        # create folder per task
        task_folder = results_folder / to_slug(task)
        task_folder.mkdir(exist_ok=True)

        # save task df to disk
        task_df.to_csv(task_folder / f"{task}_results.csv", index=False)

        excel_filename = task_folder / f"{task}_results.xlsx"
        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            unique_datasets = task_df['dataset'].unique()

            for dataset in unique_datasets:
                # Filter data for the current dataset
                dataset_df = task_df[task_df['dataset'] == dataset].copy()
                # drop columns with all nans (result metrics from other tasks will be nan)
                dataset_df = dataset_df.dropna(axis=1, how="all")

                # Select columns that end with '_std'
                std_cols = [col for col in dataset_df.columns if col.endswith('_std')]
                dataset_df['Deterministic runs?'] = None
                condition = dataset_df['# Runs'] > 1
                # Check if all values in these columns are nearly zero for each row, but only if there was more than one run
                dataset_df.loc[condition, 'Deterministic runs?'] = (dataset_df.loc[condition, std_cols].abs() < tolerance).all(axis=1)

                # Combine mean and std columns into a single column with format "mean (std)"
                for col in dataset_df.columns:
                    if col.endswith('_mean'):
                        std_col = col.replace('_mean', '_std')
                        if std_col in dataset_df.columns:
                            new_col_name = col.replace('_mean', ' mean (std)')
                            dataset_df[new_col_name] = dataset_df[col].round(mean_decimals).astype(str) + ' (' + dataset_df[std_col].round(std_decimals).astype(str) + ')'
                            # Drop the original std and mean columns
                            dataset_df = dataset_df.drop(columns=[std_col, col])

                # drop task and dataset columns
                dataset_df = dataset_df.drop(columns=["task", "dataset"])

                if len(dataset) > 31: # excel worksheet names must be <= 31 characters
                    sheet_name = dataset[:31]
                else:
                    sheet_name = dataset

                dataset_df.to_excel(writer, sheet_name=sheet_name, index=False)
                for column in dataset_df:
                    column_length = max(dataset_df[column].astype(str).map(len).max(), len(column)) + 2
                    col_idx = dataset_df.columns.get_loc(column)
                    writer.sheets[sheet_name].set_column(col_idx, col_idx, column_length)

        print(f"Excel file created for task '{task}': {excel_filename}")

def aggregate_results(df: pd.DataFrame, grouping_columns: list, rename: bool=False) -> pd.DataFrame:
    """
    Aggregates results and formats the DataFrame according to the desired output.
    """
    grouped_data = df.groupby(grouping_columns)

    aggregations = {}
    for col in df.columns:
        if col not in grouping_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                aggregations[col] = ['mean', 'std'] 
            elif col == 'run':
                aggregations['run'] = 'count'
            else:
                logger.debug(f"Not aggregating: ", col, df[col].dtype)

    # Apply the aggregations
    aggregated_df = grouped_data.agg(aggregations)
    # Flatten the multi-level column index
    aggregated_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in aggregated_df.columns]
    final_output_df = aggregated_df.reset_index()

    if rename:
        # TODO: only rename before writing to disk, not here
        # Rename columns 
        rename_dict = {
            'approach': 'Approach',
            'configuration': 'Configuration',
            'run_count': '# Runs'
        }

        final_output_df = final_output_df.rename(columns=rename_dict)

    return final_output_df

def get_setup_infos(results_file: Path):
    """
    Extract setup information from result file path

        Args:
            results_file: pathlib.Path  the path to extract from

        Returns:
            str: dataset name
            str: task name
            str: configuration 
    """
    dataset_folder = results_file.parent
    task_folder = dataset_folder.parent
    configuration_folder = task_folder.parent

    return dataset_folder.name, task_folder.name, configuration_folder.name

def gather_results(results_folder: Path, detailed_results_folder: Path):
    """
    Gathers all results.json files from the nested result directories into a single Pandas DataFrame.
    Assumes the structure: approach/configuration/task/dataset/results.json
    """
    data = [] 

    # gather the results from all results.json files
    for results_file in results_folder.rglob("*/results.json"):
        # get configuration name based on folder names
        _, _, configuration_name = get_setup_infos(results_file)
        
        with open(results_file, "r") as f:
            result_dict = json.load(f)

        result_dict["configuration"] = configuration_name
        result_dict["run"] = "TODO"
        data.append(result_dict)

    # create a single dataframe
    gathered_results_df = pd.DataFrame(data)

    # end if there are no results to gather
    if len(gathered_results_df) == 0:
        print(f'No results found to be gathered, please check if results.json files were created in the folder: *{results_folder}*')
        return

    ###########################################################################
    # prepare baseline values as added column in the results dataframe
    ###########################################################################
    # keep only baseline rows
    baseline_only = gathered_results_df.loc[gathered_results_df['approach'] == 'baseline']

    # restrict to columns that contain performance scores
    baseline_cols = [col for col in performance_cols if col in gathered_results_df.columns]

    # compute first non-null baseline value per (task, dataset)
    baseline_df = (
        baseline_only
        .groupby(["task", "dataset"], as_index=False)[baseline_cols]
        .first()
        .rename(columns={col: f"baseline_{col}" for col in baseline_cols})
    )

    # merge baseline values as column into results dataframe
    len_df_before_merge = len(gathered_results_df)
    gathered_results_df = gathered_results_df.merge(baseline_df, on=["task", "dataset"], how="left")
    len_df_after_merge = len(gathered_results_df)

    if len_df_before_merge != len_df_after_merge:
        print(f"Length of df changed after merge, please check the code")
        print(f"{len_df_before_merge} before - {len_df_after_merge} after")
        return

    # compute ratios for each col
    for col, col_type in performance_cols.items():
        if col in gathered_results_df.columns:
            baseline_col = f"baseline_{col}"

            if col_type == "higher_is_better":
                gathered_results_df[f"{col}_ratio_to_baseline"] = np.where(
                    gathered_results_df[col].notna()
                    & gathered_results_df[baseline_col].notna()
                    & (gathered_results_df[baseline_col] != 0),
                    gathered_results_df[col] / gathered_results_df[baseline_col],
                    np.nan,
                )

            elif col_type == "lower_is_better":
                gathered_results_df[f"{col}_ratio_to_baseline"] = np.where(
                    gathered_results_df[col].notna()
                    & gathered_results_df[baseline_col].notna()
                    & (gathered_results_df[baseline_col] != 0),
                    gathered_results_df[baseline_col] / gathered_results_df[col],
                    np.nan,
                )

    # drop all baseline_* columns, only keep the ratios in the final df
    baseline_cols_prefixed = [f"baseline_{col}" for col in performance_cols if col in gathered_results_df.columns]
    gathered_results_df = gathered_results_df.drop(columns=baseline_cols_prefixed)

    gathered_results_df.to_csv(results_folder/"all_results.csv", index=False)

    # aggregate the results if there were multiple runs for the same configuration
    grouping_columns = ["approach", "configuration", "task", "dataset"]
    aggregated_results_df = aggregate_results(gathered_results_df, grouping_columns, rename=True)
    aggregated_results_df.to_csv(results_folder/"all_results_aggregated.csv", index=False)

    # create an excel sheet for each individual task with one sheet per dataset
    create_excel_files_per_dataset(aggregated_results_df, results_folder=detailed_results_folder)

def gather_resources(results_folder: Path, detailed_results_folder: Path):
    all_resource_dfs = []
    for results_file in results_folder.rglob("*/resource_metrics_formatted.csv"):
        dataset_name, task_name, configuration_str = get_setup_infos(results_file)
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
        task_folder = detailed_results_folder / to_slug(task)
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

                if len(dataset) > 31: # excel worksheet names must be <= 31 characters
                    sheet_name = dataset[:31]
                else:
                    sheet_name = dataset

                dataset_df.to_excel(writer, sheet_name=sheet_name)

                # adjust column width
                for col_idx, (stage, metric) in enumerate(dataset_df.columns):
                    column_length = len(metric) + 2 
                    
                    writer.sheets[sheet_name].set_column(col_idx, col_idx, column_length)

        print(f"Excel file created for task '{task}': {excel_filename}")

@hydra.main(version_base=None, config_name="results_config")
def main(cfg: DictConfig) -> None:
    assert cfg.results_folder_name != "", f"Error in gathering results: Please enter the foldername of the results folder"
    results_folder = Path(cfg.results_folder_name)
    print(f"Gathering based on results folder: *{results_folder}*")
    assert results_folder.exists(), f"Could not find results folder at {results_folder}"

    detailed_results_folder = results_folder / "results_per_task"
    detailed_results_folder = Path(detailed_results_folder)
    detailed_results_folder.mkdir(parents=True, exist_ok=True)    

    print(f"Calling gather_results")
    gather_results(results_folder, detailed_results_folder)

    print(f"Calling gather_resources")
    gather_resources(results_folder, detailed_results_folder)


if __name__ == "__main__":
    main()