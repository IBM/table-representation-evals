import logging
import pandas as pd
from pathlib import Path
import numpy as np
import json

from benchmark_src.results_processing.excel_writer import create_excel_files_per_dataset
from benchmark_src.results_processing import results_helper, ranking

logger = logging.getLogger(__name__)

def prepare_results_for_reporting(df: pd.DataFrame, grouping_columns: list) -> pd.DataFrame:
    """
    Each (approach, configuration, task, dataset) is expected to have exactly one
    results.json (one job in the output directory structure). Warn and keep only
    the first if that's ever violated, and rename columns for display.
    """
    duplicate_mask = df.duplicated(subset=grouping_columns, keep=False)
    if duplicate_mask.any():
        logger.warning(
            f"Found multiple results.json files for the same approach/configuration/task/dataset, "
            f"keeping only the first of each:\n{df.loc[duplicate_mask, grouping_columns]}"
        )
        df = df.drop_duplicates(subset=grouping_columns, keep='first')

    return df.rename(columns={'approach': 'Approach', 'configuration': 'Configuration'})


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
        current_result_subfolder = results_file.parent

        with open(results_file, "r") as f:
            result_dict = json.load(f)

        result_dict["configuration"] = configuration_name

        # need resource file from the folder of the results file
        new_metrics_file = current_result_subfolder / "resource_metrics_task.json"
        if new_metrics_file.exists():
            with open(new_metrics_file) as file:
                metrics = json.load(file)

            # take only the inference metrics!
            result_dict["execution_time (s)"] = metrics["execution_time (s)"]
            result_dict["peak_cpu (%)"] = metrics["peak_cpu (%)"]
            result_dict["peak_memory (MB)"] = metrics["peak_memory (MB)"]
            try:
                result_dict["peak_gpu_memory (MB)"] = metrics["peak_gpu_memory (MB)"]
            except KeyError:
                result_dict["peak_gpu_memory (MB)"] = None
        else:
            old_metrics_file = current_result_subfolder / "resource_metrics.csv"
            if old_metrics_file.exists():
                metrics = pd.read_csv(old_metrics_file)

                # Select only the inference row
                inference_row = metrics[metrics["function"] == "task_inference"]

                if inference_row.empty:
                    raise ValueError("No task_inference row found in old metrics file")

                # Take the first (and usually only) matching row
                inference_row = inference_row.iloc[0]

                # Fill result_dict with inference metrics
                result_dict["execution_time (s)"] = inference_row["execution_time (s)"]
                result_dict["peak_cpu (%)"] = inference_row["peak_cpu (%)"]
                result_dict["peak_memory (MB)"] = inference_row["peak_memory (MB)"]

                # Old format may not have GPU memory
                if "peak_gpu_memory (MB)" in inference_row:
                    result_dict["peak_gpu_memory (MB)"] = inference_row["peak_gpu_memory (MB)"]
                else:
                    result_dict["peak_gpu_memory (MB)"] = None
            else:
                print(f"Don't have resource metrics for {current_result_subfolder}")
                result_dict["execution_time (s)"] = None
                result_dict["peak_cpu (%)"] = None
                result_dict["peak_memory (MB)"] = None
                result_dict["peak_gpu_memory (MB)"] = None

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
    baseline_cols = [col for col in results_helper.performance_cols if col in gathered_results_df.columns]

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
    for col, col_type in results_helper.performance_cols.items():
        #print(f"Computing ratio for column: {col}")
        # if approach_ prefixed column exists merge it with performance_col temporarily and use it for ratio computation
        if col in gathered_results_df.columns:
            approach_metric = "approach_" + "_".join(col.split("_")[1:])
            if approach_metric in gathered_results_df.columns:
                # merge the values from old col and new approach_metric, if the column exists
                # only copy values where approach_metric is not null
                gathered_results_df = gathered_results_df.copy()
                gathered_results_df[col] = np.where(
                    gathered_results_df[approach_metric].notna(),
                    gathered_results_df[approach_metric],
                    gathered_results_df[col]
                )
                
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
    baseline_cols_prefixed = [f"baseline_{col}" for col in results_helper.performance_cols if col in gathered_results_df.columns]
    gathered_results_df = gathered_results_df.drop(columns=baseline_cols_prefixed)

    # drop all completely empty columns
    gathered_results_df = gathered_results_df.dropna(axis=1, how='all')

    grouping_columns = ["approach", "configuration", "task", "dataset"]
    results_df = prepare_results_for_reporting(gathered_results_df, grouping_columns)
    results_df.to_csv(results_folder/"all_results.csv", index=False)

    # create an excel sheet for each individual task with one sheet per dataset
    create_excel_files_per_dataset(results_df, results_folder=detailed_results_folder)

    ###########################################################################
    # compute elo scores per approach/configuration across all datasets
    ###########################################################################
    elo_df, elo_overall_df = ranking.compute_elo_scores(results_df)
    elo_df.to_csv(results_folder/"elo_scores_per_task.csv", index=False)
    elo_overall_df.to_csv(results_folder/"elo_scores_overall.csv", index=False)