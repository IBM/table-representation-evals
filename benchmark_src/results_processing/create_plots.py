import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import ast

from benchmark_src.results_processing.results_helper import get_setup_infos
from benchmark_src.results_processing.plots import plot_info, datatype_comparison, more_similar_than_plots, quadrant_charts, predML_plots


def gather_results_and_metrics(results_folder):
    # load all_results_aggregated.csv
    collected_results_df = pd.read_csv(results_folder / "all_results_aggregated.csv")

    gathered_resource_dfs = []

    # load resource metrics and insert into aggregated_results_df
    for results_file in results_folder.rglob("*/resource_metrics_formatted.csv"):
        dataset_name, task_name, configuration_str = get_setup_infos(results_file)
        resource_df = pd.read_csv(results_file, header=None)

        # Need tuple column names back as real tuples
        header_row = resource_df.iloc[0].tolist()
        tuple_column_names = [ast.literal_eval(col) if col.startswith('(') and col.endswith(')') else col for col in header_row]
        resource_df.columns = tuple_column_names
        # drop the first row (that conatined the header information)
        resource_df = resource_df.iloc[1:]

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
            if isinstance(col, tuple):
                if col[0] in replacement_map:
                    function_name = replacement_map[col[0]]
                else:
                    function_name = col[0]
                col_name = function_name + "---" + col[1]
                #col_name = (replacement_map[col[0]], col[1])
                new_columns.append(col_name)
            else:
                new_columns.append(col)
        resource_df.columns = new_columns
        
        resource_df["dataset"] = dataset_name
        resource_df["task"] = task_name
        resource_df["Configuration"] = configuration_str

        resource_df = resource_df.rename(columns={'approach': "Approach"})
        gathered_resource_dfs.append(resource_df)

    resource_df = pd.concat(gathered_resource_dfs)
    # convert numeric columns back to numeric types
    for col in resource_df.columns:
        try:
            resource_df[col] = pd.to_numeric(resource_df[col]) 
        except ValueError:
            pass # keep string columns as strings
    # Define the columns to merge on
    merge_cols = ['dataset', 'task', 'Approach', 'Configuration']

    collected_results_df = pd.merge(collected_results_df, resource_df, on=merge_cols, how='left')
    #collected_results_df.to_csv("all_results_with_metrics.csv")

    return collected_results_df

def predictive_ml_add_task_type(task_df: pd.DataFrame) -> pd.DataFrame:
    # add column to dataframe with task type
    dataset_to_type = {x: y["task_type"] for x, y in plot_info.predictiveML_dataset_info.items()}
    dataset_to_num_nonnumeric = {x: y["has_non_numeric_columns"] for x, y in plot_info.predictiveML_dataset_info.items()}
    dataset_to_num_cols = {x: y["num_cols"] for x, y in plot_info.predictiveML_dataset_info.items()}
    task_df['task_type'] = task_df['dataset'].map(dataset_to_type)
    task_df['num_cols'] = task_df['dataset'].map(dataset_to_num_cols)
    task_df['num_nonnumeric'] = task_df['dataset'].map(dataset_to_num_nonnumeric)

    return task_df

def filter_for_results_on_all_datasets(task_type, task_type_df):
    print(f"Unique datasets ({task_type}):", len(task_type_df['dataset'].unique()))

    all_approaches = task_type_df['Approach'].unique()
    num_expected_approaches = len(all_approaches)
    approach_counts = task_type_df.groupby('dataset')['Approach'].nunique()
    datasets_with_all_approaches = approach_counts[approach_counts == num_expected_approaches].index
    # Filter the original DataFrame to keep only these datasets
    complete_results_df = task_type_df[task_type_df['dataset'].isin(datasets_with_all_approaches)]
    print(f"Unique datasets with full results({task_type}): {len(complete_results_df['dataset'].unique())}")
    return complete_results_df


def main(results_folder: Path, plots_folder:Path):
    # read in results in metrics files
    all_results_df = gather_results_and_metrics(results_folder=results_folder)

    # group by task
    unique_tasks = all_results_df['task'].unique()
    for task in unique_tasks:
        # Filter data for the current task
        task_df = all_results_df[all_results_df['task'] == task].copy()
        # drop columns with all nans (result metrics from other tasks will be nan)
        task_df = task_df.dropna(axis=1, how="all")

        print(f"Unique datasets ({task}):", len(task_df['dataset'].unique()))

        if task == "row_similarity_search":
            pass
            # row_sim_plots_folder = plots_folder / "row_similarity"
            # row_sim_plots_folder.mkdir(exist_ok=True)
            # ####################################################################################################
            # # Create bar chart with all approaches (aggregated over datasets)
            # ####################################################################################################
            # row_similarity_plots.create_barplot(df=task_df, results_folder=row_sim_plots_folder)
            # current_df = task_df
            # current_df = current_df.loc[current_df['Approach'] != "tabula_8b"]
            # row_similarity_plots.create_barplot_datasets(df=current_df, results_folder=row_sim_plots_folder)

            # ####################################################################################################
            # # Create time / performance quadrant chart (with all approaches except aidb)
            # ####################################################################################################
            # #current_df = task_df.loc[task_df['Approach'] != "aidb"]
            # #current_df=task_df
            # # also exclude tabula
            # #current_df = current_df.loc[current_df['Approach'] != "tabula_8b"]
            # row_similarity_plots.quadrant_plot_aggregated_time(task_df=current_df, results_folder=row_sim_plots_folder)
            # row_similarity_plots.quadrant_plot_aggregated_cpu(task_df=current_df, results_folder=row_sim_plots_folder)

            # row_similarity_plots.plot_per_dataset(task_df=current_df, results_folder=row_sim_plots_folder)

            # current_df = current_df.loc[current_df['Approach'] != "GritLM"]
            # row_similarity_plots.plot_point_per_dataset(task_df=current_df, results_folder=row_sim_plots_folder)

            # ####################################################################################################
            # # If time: fex3 results
            # ####################################################################################################

        elif task == "predictive_ml":

            print("############# Predictive ML Plots")
            predML_plots_folder = plots_folder / "predictive_ml"
            predML_plots_folder.mkdir(exist_ok=True)

            task_df = predictive_ml_add_task_type(task_df)

            task_type_groups = task_df.groupby(["task_type"])

            for task_type, task_type_df in task_type_groups:
                task_type_df = task_type_df.dropna(axis=1, how="all")
                task_type = task_type[0]

                complete_results_df = filter_for_results_on_all_datasets(task_type, task_type_df)

                ####################################################################################################
                # Create time / performance quadrant chart (with all approaches except aidb)
                ####################################################################################################
                if task_type == "binary":
                    results_column = 'XGBoost_roc_auc_score (↑)_mean'
                    title = "Tabarena Binary Classification: Time Taken vs. Performance Reached"

                    predML_plots.create_binary_barplot(df=complete_results_df, results_folder=predML_plots_folder)
                    predML_plots.create_binary_barplot_altair(df=complete_results_df, results_folder=predML_plots_folder, aggregated=True, dataset_name="aggregated")
                    filename = "binary_nonnumeric_linechart.png"
                    predML_plots.create_nonnumeric_performance_plot(df=complete_results_df, task_type=task_type, metric_name=results_column, results_folder=predML_plots_folder, filename=filename)
                    filename = "binary_num_columns.png"
                    predML_plots.create_col_performance_plot(df=complete_results_df, task_type=task_type,metric_name=results_column, results_folder=predML_plots_folder, filename=filename)

                    # create barplots per dataset
                    unique_datasets = complete_results_df['dataset'].unique()

                    for dataset_name in unique_datasets:
                        # Filter data for the current dataset
                        dataset_df = complete_results_df[complete_results_df['dataset'] == dataset_name].copy()
                        # drop columns with all nans (result metrics from other tasks will be nan)
                        dataset_df = dataset_df.dropna(axis=1, how="all")
                        predML_plots.create_binary_barplot_altair(df=dataset_df, results_folder=predML_plots_folder, aggregated=False, dataset_name=dataset_name)

                elif task_type == "multiclass":
                    results_column = 'XGBoost_log_loss (↓)_mean'
                    title = "Tabarena Multiclass Classification: Time Taken vs. Performance Reached"

                    predML_plots.create_multiclass_barplot(df=complete_results_df, results_folder=predML_plots_folder)
                    filename = "multiclass_nonnumeric_linechart.png"
                    predML_plots.create_nonnumeric_performance_plot(df=complete_results_df, task_type=task_type,metric_name=results_column, results_folder=predML_plots_folder, filename=filename)
                    filename = "multiclass_num_columns.png"
                    predML_plots.create_col_performance_plot(df=complete_results_df, task_type=task_type,metric_name=results_column, results_folder=predML_plots_folder, filename=filename)

                elif task_type == "regression":
                    current_df = complete_results_df.loc[complete_results_df['Approach'] != "tabicl"] # tabicl does not support regression
                    results_column = 'XGBoost_rmse (↓)_mean'
                    title = "Tabarena Regression: Time Taken vs. Performance Reached"

                    predML_plots.create_regression_barplot(df=current_df, results_folder=predML_plots_folder)
                    filename = "regression_nonnumeric_linechart.png"
                    predML_plots.create_nonnumeric_performance_plot(df=current_df, task_type=task_type, metric_name=results_column, results_folder=predML_plots_folder, filename=filename)
                    filename = "regression_num_columns.png"
                    predML_plots.create_col_performance_plot(df=current_df, task_type=task_type,metric_name=results_column, results_folder=predML_plots_folder, filename=filename)

                else:
                    raise ValueError(f"Task type unkown: {task_type}")

                # TODO: idea: set markers consistent across plots per approach (colors/shapes)
                marker_labels = False
                # exclude aidb from the quadrant charts because they are run on different hardware!
                # current_df = task_type_df.loc[task_type_df['Approach'] != "aidb"]
                current_df = task_type_df
                filename = f"quadrant_plot_{task_type}_full_time.png"
                quadrant_charts.quadrant_plot_aggregated_over_datasets(filename=filename, task_df=current_df, task_type=task_type, results_folder=predML_plots_folder, results_column=results_column, title=title, marker_labels=marker_labels, inference_only=False)
                filename = f"quadrant_plot_{task_type}_inference_only.png"
                quadrant_charts.quadrant_plot_aggregated_over_datasets(filename=filename, task_df=current_df, task_type=task_type, results_folder=predML_plots_folder, results_column=results_column, title=title, marker_labels=marker_labels, inference_only=True)

                ####################################################################################################
                # Create time / performance quadrant chart (exclude tabula because it took so long)
                ####################################################################################################
                # also exclude tabula
                current_df = current_df.loc[task_type_df['Approach'] != "tabula_8b"]
                marker_labels = False
                filename = f"quadrant_plot_closer_{task_type}_full_time.png"
                quadrant_charts.quadrant_plot_aggregated_over_datasets(filename=filename, task_df=current_df, task_type=task_type, results_folder=predML_plots_folder, results_column=results_column, title=title, marker_labels=marker_labels, inference_only=False)
                filename = f"quadrant_plot_closer_{task_type}_inference_only.png"
                quadrant_charts.quadrant_plot_aggregated_over_datasets(filename=filename, task_df=current_df, task_type=task_type, results_folder=predML_plots_folder, results_column=results_column, title=title, marker_labels=marker_labels, inference_only=True)


                #############################################################
                # Create table_size / performance scatter plot per task_type
                #############################################################


                ########################################################
                # Create #rows / performance scatter plot per task_type
                ########################################################


                ########################################################
                # Create #cols / performance scatter plot per task_type
                ########################################################


            datatype_comparison.create_datatype_comparison_plot(task_df=task_df, plots_folder=predML_plots_folder)
        elif task == "more_similar_than":
            more_than_plots_folder = plots_folder / "more_similar_than"
            more_than_plots_folder.mkdir(exist_ok=True)

            print(task_df.columns)
            # Create bar chart
            more_similar_than_plots.create_barplot(df=task_df, results_folder=more_than_plots_folder)



if __name__ == "__main__":
    results_folder = Path("results")
    assert results_folder.exists(), f"Could not find results folder at {results_folder}"

    plots_folder = results_folder / "plots"
    plots_folder.mkdir(exist_ok=True)

    main(results_folder, plots_folder)
