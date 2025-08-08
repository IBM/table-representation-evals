import pandas as pd
import numpy as np


### TODO: should be configured by user, not hardcoded here!
def approach_name_for_plot(config_string: str):
    if "GritLM" in config_string:
        return "GritLM-7B"
    elif "BAAI" in config_string:
        return "BAAI/bge-base-en-v1.5"
    elif "granite" in config_string:
        return "IBM/granite-embedding-30m-english"
    if "embedding_model" in config_string:
        approach =config_string.split("=")[-1]
        return approach
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
        approach_name = approach_name_for_plot(row["Configuration"])
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
