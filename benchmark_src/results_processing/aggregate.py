import logging
import pandas as pd
from pathlib import Path
import numpy as np
import json

from benchmark_src.results_processing.excel_writer import create_excel_files_per_dataset
from benchmark_src.results_processing import results_helper

logger = logging.getLogger(__name__)

performance_cols = {
    'XGBoost_rmse (↓)': 'lower_is_better',
    'KNeighbors_rmse (↓)': 'lower_is_better',
    'LinearRegression_rmse (↓)': 'lower_is_better',
    'XGBoost_roc_auc_score (↑)': 'higher_is_better',
    'KNeighbors_roc_auc_score (↑)': 'higher_is_better',
    'XGBoost_log_loss (↓)': 'lower_is_better',
    'KNeighbors_log_loss (↓)': 'lower_is_better'
}


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


def gather_results(results_folder: Path, detailed_results_folder: Path):
    """
    Gathers all results.json files from the nested result directories into a single Pandas DataFrame.
    Assumes the structure: approach/configuration/task/dataset/results.json
    """
    data = [] 

    # gather the results from all results.json files
    for results_file in results_folder.rglob("*/results.json"):
        # get configuration name based on folder names
        _, _, configuration_name = results_helper.get_setup_infos(results_file)
        
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