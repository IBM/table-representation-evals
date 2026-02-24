import pandas as pd
from benchmark_src.results_processing import create_plots
import numpy as np
from pathlib import Path

### TODO: should be configured by user, not hardcoded here!
def approach_name_for_plot(approach_name: str, config_string: str):
    if approach_name in ["tabicl", "tabpfn", "sap_rpt_oss"]:
        if "predML_based_on=row_embeddings" in config_string:
            return f"{approach_name}-row_emb"

    if "GritLM" in config_string:
        return "GritLM-7B"
    elif "BAAI" in config_string:
        return "BAAI/bge-base-en-v1.5"
    elif "granite" in config_string:
        if "30m-english" in config_string:
            return "granite-30m"
        elif "english-r2" in config_string:
            return "granite-r2"
    elif "all-MiniLM-L6-v2" in config_string:
        return "all-MiniLM-L6-v2"
    elif "set_prios" in config_string:
        if "True" in config_string:
            if "row_sim_search=embeddings" in config_string:
                return "aidb_prio_emb"
            return "aidb_prio_custom"
        else:
            if "row_sim_search=embeddings" in config_string:
                return "aidb_default_emb"
            return "aidb_default_custom"
        
    return None


def get_task_df(results_folder: Path, task_name: str) -> pd.DataFrame:
    all_results_df = create_plots.gather_results_and_metrics(results_folder=results_folder)

    # group by task
    unique_tasks = all_results_df['task'].unique()

    # group by approach
    unique_approaches = all_results_df['Approach'].unique()
    #print(f"Got unique approaches: {unique_approaches}")

    # Filter data for the current task
    task_df = all_results_df[all_results_df['task'] == task_name].copy()
    # drop columns with all nans (result metrics from other tasks will be nan)
    task_df = task_df.dropna(axis=1, how="all")

    print(f"Unique datasets ({task_name}):", len(task_df['dataset'].unique()))
    return task_df

def get_list_of_all_runs(task_df: pd.DataFrame):
    # Get all unique runs
    unique_runs = task_df[['Approach', 'Configuration']].drop_duplicates()

    # Automatically create a Python list of tuples
    exclude_runs_list = [
        (row['Approach'], row['Configuration'])
        for _, row in unique_runs.iterrows()
    ]

    # Print it in a nicely formatted way for copying
    print("include_runs = [")
    for run in exclude_runs_list:
        print(f"    {run},")
    print("]")


def collect_data_for_plotting(df: pd.DataFrame, metric: str, is_aggregated: bool, resources: bool=False):
    if is_aggregated:
        metric += "_mean"
    orig_metric = metric

    data = {"Approach": [], "Performance": []}
    if is_aggregated:
        data["std"] = []
    else:
        data["Dataset"] = []
    if resources:
        data["OverallTime"] = []
        data["InferenceTime"] = []
        data["InferenceCPU"] = []
    for _, row in df.iterrows():
        approach_name = approach_name_for_plot(row["Approach"],row["Configuration"])
        if not approach_name:
            approach_name = row["Approach"]
        #print(f"Approach name {approach_name}")
        data["Approach"].append(approach_name)
        if np.isnan(row[orig_metric + "_mean"]):
            metric = "approach_" + "_".join(metric.split("_")[1:])
            print(f"Looking for {metric}")
        else:
            metric = orig_metric
        performance = row[metric + "_mean"]
        data["Performance"].append(performance)
        if is_aggregated:
            data["std"].append(row[metric + "_std"])
        else:
            data["Dataset"].append(row["dataset"])
        if resources:
            setup_time = row["model_setup---execution_time (s)_mean"]
            inference_time = row["task_inference---execution_time (s)_mean"]
            data["OverallTime"].append(setup_time + inference_time)
            data["InferenceTime"].append(inference_time)
            data["InferenceCPU"].append(row["task_inference---peak_cpu (%)_mean"])

    data = pd.DataFrame(data)

    return data
