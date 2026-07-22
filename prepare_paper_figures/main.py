######################################################
# Build all result tables and plots for the paper
######################################################
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from benchmark_src.utils import cfg_utils

import ranking_table
import row_sim_plot_aggregated, row_sim_linechart_topk, row_sim_results_table, quadrant_chart_row
import col_sim_results_table, col_sim_bar_plot_per_dataset
import tabular_prediction_result_tables, tabular_prediction_barchart_binary, tabular_prediction_barchart_multiclass, tabular_prediction_barchart_regression, tablular_prediction_elo_table
import cell_bar_plot, cell_bar_plot_stacked, cell_results_table, quadrant_chart_cell
import triplet_row_results_table, triplet_row_bar_plot_original, triplet_row_bar_plot_difficulty
import table_similarity_search_tables
import retrieval_main_table, retrieval_md_vs_csv_table, retrieval_per_dataset_bars, retrieval_recall_line, retrieval_rowlimit_bars
import shuffling_ecb_bars, shuffling_ecb_table, shuffling_magnitude_table, shuffling_main_table
import shuffling_perturbation_table, shuffling_row_col_scatter, shuffling_size_table, shuffling_variation_heatmap
import ttd_classifier_table
import serialization_deltas
import column_type_annotation_bar_plot, column_type_annotation_results_table
import nl2_bar_plot, nl2_results_table

# Shared with the general post-run plots in benchmark_src/results_processing/create_plots.py
# so both pipelines render the same approach with the same name/color; edit
# configs/approach_plotting.yaml (missing entries here raise a ValueError below,
# since paper figures require a complete, curated mapping rather than a fallback).
_approach_plotting = cfg_utils.load_approach_plotting()
color_mapping = {key: entry["color"] for key, entry in _approach_plotting.items() if "color" in entry}
name_mapping = {key: entry["name"] for key, entry in _approach_plotting.items() if "name" in entry}
marker_mapping = {key: entry["marker"] for key, entry in _approach_plotting.items() if "marker" in entry}


# CLI keys for --tasks, one per gated section below (ranking_table is a cross-task summary
# and always runs regardless of this selection).
VALID_TASKS = [
    "row_similarity_search",
    "column_similarity_search",
    "row_triplet_evaluation",
    "cell_similarity_search",
    "table_retrieval",
    "table_similarity_search",
    "table_shuffling",
    "table_type_detection",
    "serialization_deltas",
    "predictive_ml",
    "column_type_annotation",
    "schema_linking",
    "value_linking",
]


def _parse_tasks(tasks: str) -> set[str]:
    if tasks.strip().lower() == "all":
        return set(VALID_TASKS)
    selected = {t.strip() for t in tasks.split(",") if t.strip()}
    unknown = selected - set(VALID_TASKS)
    if unknown:
        raise typer.BadParameter(f"Unknown task(s): {', '.join(sorted(unknown))}. Valid tasks: {', '.join(VALID_TASKS)}")
    return selected


def main(
    results_folder: Annotated[str, typer.Argument(help="Path to the results folder containing all_results.csv")] = "results",
    tasks: Annotated[str, typer.Option(help=f"Comma-separated tasks to generate plots for, or 'all'. Choices: {', '.join(VALID_TASKS)}")] = "all",
):
    results_folder = Path(results_folder)
    assert results_folder.exists(), f"Could not find results folder at {results_folder}"

    selected_tasks = _parse_tasks(tasks)

    # keep each results folder's plots separate so different result sets don't clobber each other's output;
    # nested under generated_figures/ so the whole tree can be gitignored in one line regardless of folder name
    plots_folder = Path("./prepare_paper_figures/generated_figures") / results_folder.name
    plots_folder.mkdir(parents=True, exist_ok=True)

    all_results_df = pd.read_csv(results_folder / "all_results.csv")

    print(all_results_df["Approach"].unique())

    ## Filter approaches
    exclude_runs = [
    ('sentence_transformer', 'embedding_model=BAAI_bge-base-en-v1.5,table_row_limit=100'),
    ('sentence_transformer', 'embedding_model=ibm-granite_granite-embedding-30m-english,table_row_limit=100')
    ]
    all_results_df = all_results_df[
        ~all_results_df.set_index(['Approach', 'Configuration']).index.isin(exclude_runs)
    ]

    ## Set colors for the approaches
    all_results_df["color"] = all_results_df.apply(
        lambda row: color_mapping.get((row["Approach"], row["Configuration"]), "#000000"),  # fallback color if missing
        axis=1
    )
    ## Set chart names for the approaches
    all_results_df["chart_name"] = all_results_df.apply(
        lambda row: name_mapping.get((row["Approach"], row["Configuration"]), "TODO"),  # fallback color if missing
        axis=1
    )
    ## Set scatter markers for the approaches (default circle if not curated)
    all_results_df["marker"] = all_results_df.apply(
        lambda row: marker_mapping.get((row["Approach"], row["Configuration"]), "o"),
        axis=1
    )

    # print rows with fallback chart name to check if we missed any approach/configuration in the mapping
    missing_chart_names = all_results_df[all_results_df["chart_name"] == "TODO"]
    if not missing_chart_names.empty:
        print("Warning: Missing chart names for the following Approach + Configuration combinations:")
        # print each name configuration separately:
        for idx, row in missing_chart_names.iterrows():
            print(f"Approach: {row['Approach']}, Configuration: {row['Configuration']}")
        raise ValueError("Please update the name_mapping with chart names for the missing Approach + Configuration combinations!")
    else:
        print("All Approach + Configuration combinations have chart names ✅")

    # same for colors: 
    missing_colors = all_results_df[all_results_df["color"] == "#000000"]
    if not missing_colors.empty:    
        print("Warning: Missing colors for the following Approach + Configuration combinations:")
        for idx, row in missing_colors.iterrows():
            print(f"Approach: {row['Approach']}, Configuration: {row['Configuration']}")
        raise ValueError("Please update the color_mapping with colors for the missing Approach + Configuration combinations!")
    else:        print("All Approach + Configuration combinations have colors ✅")

    group_cols = ["Approach", "Configuration", "task", "dataset"]
    # Assert that we have every row of "Approach" "Configuration" "task" "dataset" only once
    duplicates = all_results_df.duplicated(subset=group_cols, keep=False)
    if duplicates.any():
        # If duplicates exist, raise an error or print them
        print("Duplicate rows found for the same Approach + Configuration + task:")
        print(all_results_df[duplicates])
        raise ValueError("Each Approach + Configuration + task combination must be unique!")

    # If no duplicates, you can safely continue
    print("All Approach + Configuration + task combinations are unique ✅")


    ##############################################################################################
    #
    #     Row Similarity Search
    # 
    ##############################################################################################
    if "row_similarity_search" in selected_tasks:
        # Filter data for the current task
        df = all_results_df[all_results_df['task'] == "row_similarity_search"].copy()
        if df.empty:
            print("No row_similarity_search results in this results folder, skipping.")
        else:
            # ----------------------------------------
            # Average MRR scores over all datasets
            # ----------------------------------------
            # TODO: (only 7 because some approaches were OOM? or on all 9 and exclude the two approaches?)
            row_sim_plot_aggregated.create_barplot(df, plots_folder)

            # ------------------------------------------------
            # Dataset-level bar plot (MRR as metric)
            # ------------------------------------------------
            # goal: show how approaches vary per dataset


            # ------------------------------------------------
            # Results table per dataset for all approaches
            # ------------------------------------------------
            row_sim_results_table.create_results_table(df, plots_folder)

            # TODO: only if time
            # ------------------------------------------------
            # Line Plot with k on the x-axis and Recall@k on the y-axis
            # ------------------------------------------------
            row_sim_linechart_topk.create_lineplot(df, plots_folder)

            # ------------------------------------------------
            # Resource vs. performance tradeoff
            # ------------------------------------------------
            quadrant_chart_row.build_quadrant_chart(df, plots_folder)
            quadrant_chart_row.build_quadrant_chart_vram(df, plots_folder)

    ##############################################################################################
    #
    #     Column Similarity Search
    # 
    ##############################################################################################
    if "column_similarity_search" in selected_tasks:
        # Filter data for the current task
        df = all_results_df[all_results_df['task'] == "column_similarity_search"].copy()
        if df.empty:
            print("No column_similarity_search results in this results folder, skipping.")
        else:
            # ----------------------------------------
            # Average MRR scores over all datasets ?
            # ----------------------------------------


            # ------------------------------------------------
            # Dataset-level bar plot (MRR as metric)?
            # ------------------------------------------------
            col_sim_bar_plot_per_dataset.create_barplot(df, plots_folder)


            # ------------------------------------------------
            # Results table per dataset for all approaches
            # ------------------------------------------------
            col_sim_results_table.create_results_table(df, plots_folder)


    ##############################################################################################
    #
    #     Triplet Tests
    # 
    ##############################################################################################
    if "row_triplet_evaluation" in selected_tasks:
        # Filter data for the current task
        df = all_results_df[all_results_df['task'] == "row_triplet_evaluation"].copy()
        # drop columns with all nans (result metrics from other tasks will be nan)
        df = df.dropna(axis=1, how="all")
        if df.empty:
            print("No row_triplet_evaluation results in this results folder, skipping.")
        else:
            # results table
            triplet_row_results_table.create_results_table(df, plots_folder)

            # plot original
            triplet_row_bar_plot_original.create_barplot(df, plots_folder)

            # plot easy vs. medium accuracy per approach on wikidata_books
            triplet_row_bar_plot_difficulty.create_barplot(df, plots_folder)

            # plot ablations for books and astronomical_objects

    ##############################################################################################
    #
    #     Cell Similarity Search
    # 
    ##############################################################################################
    if "cell_similarity_search" in selected_tasks:
        # Filter data for the current task
        df = all_results_df[all_results_df['task'] == "cell_similarity_search"].copy()
        if df.empty:
            print("No cell_similarity_search results in this results folder, skipping.")
        else:
            # ------------------------------------------------
            # Dataset-level bar plot
            # ------------------------------------------------
            cell_bar_plot.create_barplot(df, plots_folder)
            cell_bar_plot_stacked.create_barplot(df, plots_folder)

            # ------------------------------------------------
            # Results table per dataset for all approaches
            # ------------------------------------------------
            cell_results_table.create_results_table(df, plots_folder)

            # ------------------------------------------------
            # Quadrant chart
            # ------------------------------------------------
            quadrant_chart_cell.build_quadrant_chart(df, plots_folder)

    ##############################################################################################
    #
    #     Table Retrieval
    #
    ##############################################################################################
    if "table_retrieval" in selected_tasks:
        # Filter data for the current task
        df = all_results_df[all_results_df['task'] == "table_retrieval"].copy()
        # drop columns with all nans (result metrics from other tasks will be nan)
        df = df.dropna(axis=1, how="all")
        if df.empty:
            print("No table_retrieval results in this results folder, skipping.")
        else:
            # TARGET-based (multi-topk) retrieval tables/plots
            retrieval_main_table.create_table(df, plots_folder)
            retrieval_md_vs_csv_table.create_table(df, plots_folder)
            retrieval_per_dataset_bars.create_barplot(df, plots_folder)
            retrieval_recall_line.create_lineplot(df, plots_folder)
            retrieval_rowlimit_bars.create_barplot(df, plots_folder)

    ##############################################################################################
    #
    #     Table Similarity Search
    #
    ##############################################################################################
    if "table_similarity_search" in selected_tasks:
        # Filter data for the current task
        df = all_results_df[all_results_df['task'] == "table_similarity_search"].copy()
        # drop columns with all nans (result metrics from other tasks will be nan)
        df = df.dropna(axis=1, how="all")
        if df.empty:
            print("No table_similarity_search results in this results folder, skipping.")
        else:
            # results table for main text
            table_similarity_search_tables.create_results_table_small(df, plots_folder)

            # results table for appendix
            #table_similarity_search_tables.create_results_table_appendix(df, plots_folder)

    ##############################################################################################
    #
    #     Table Shuffling
    #
    ##############################################################################################
    if "table_shuffling" in selected_tasks:
        # Filter data for the current task
        df = all_results_df[all_results_df['task'] == "table_shuffling"].copy()
        # drop columns with all nans (result metrics from other tasks will be nan)
        df = df.dropna(axis=1, how="all")
        if df.empty:
            print("No table_shuffling results in this results folder, skipping.")
        else:
            shuffling_ecb_bars.create_barplot(df, plots_folder)
            shuffling_ecb_table.create_table(df, plots_folder)
            shuffling_magnitude_table.create_table(df, plots_folder)
            shuffling_main_table.create_accuracy_table(df, plots_folder)
            shuffling_main_table.create_bcs_table(df, plots_folder)
            shuffling_perturbation_table.create_table(df, plots_folder)
            shuffling_row_col_scatter.create_scatter(df, plots_folder)
            shuffling_size_table.create_table(df, plots_folder)
            shuffling_variation_heatmap.create_heatmap(df, plots_folder)

    ##############################################################################################
    #
    #     Table Type Detection
    #
    ##############################################################################################
    if "table_type_detection" in selected_tasks:
        # Filter data for the current task
        df = all_results_df[all_results_df['task'] == "table_type_detection"].copy()
        # drop columns with all nans (result metrics from other tasks will be nan)
        df = df.dropna(axis=1, how="all")
        if df.empty:
            print("No table_type_detection results in this results folder, skipping.")
        else:
            ttd_classifier_table.create_table(df, plots_folder)

    ##############################################################################################
    #
    #     Serialization Deltas (across table_retrieval / table_shuffling / table_type_detection)
    #
    ##############################################################################################
    if "serialization_deltas" in selected_tasks:
        serialization_deltas.create_plot(all_results_df, plots_folder)


    ##############################################################################################
    #
    #     Tabular Prediction
    # 
    ##############################################################################################
    # Falls back to an empty table (all approaches penalized) if predictive_ml wasn't
    # selected or has no data in this results folder, so ranking_table below always has
    # a valid frame to consume.
    elo_table = pd.DataFrame(columns=["chart_name", "elo_score_task"])

    if "predictive_ml" in selected_tasks:
        # Filter data for the current task
        df = all_results_df[all_results_df['task'] == "predictive_ml"].copy()
        # drop columns with all nans (result metrics from other tasks will be nan)
        df = df.dropna(axis=1, how="all")
        if df.empty:
            print("No predictive_ml results in this results folder, skipping.")
        else:
            print(df.columns)
            # ----------------------------------------
            # Result tables
            # ----------------------------------------
            tabular_prediction_result_tables.create_results_table_binary_classification(df, plots_folder)
            tabular_prediction_result_tables.create_results_table_multiclass_classification(df, plots_folder)
            tabular_prediction_result_tables.create_results_table_regression(df, plots_folder)

            # ----------------------------------------
            # Plots as percentage to baseline
            # ----------------------------------------
            tabular_prediction_barchart_binary.create_barplot(df, plots_folder)
            tabular_prediction_barchart_multiclass.create_barplot_multiclass(df, plots_folder)
            tabular_prediction_barchart_regression.create_barplot_regression(df, plots_folder)

            # ----------------------------------------
            # Elo table
            # ----------------------------------------
            # TODO: get elo scores to later use in ranking table!
            elo_table = tablular_prediction_elo_table.create_elo_table(df, plots_folder)

    ##############################################################################################
    #
    #     Column Type Annotation
    #
    ##############################################################################################
    if "column_type_annotation" in selected_tasks:
        df = all_results_df[all_results_df['task'] == "column_type_annotation"].copy()
        df = df.dropna(axis=1, how="all")
        if df.empty:
            print("No column_type_annotation results in this results folder, skipping.")
        else:
            column_type_annotation_bar_plot.create_barplot(df, plots_folder)
            column_type_annotation_results_table.create_results_table(df, plots_folder)

    ##############################################################################################
    #
    #     Schema Linking / Value Linking
    #
    ##############################################################################################
    for task_name, task_label, barplot_fn in [
        ("schema_linking", "column_mapping", nl2_bar_plot.create_barplot_single_dataset),
        ("value_linking", "cell2column_mapping", nl2_bar_plot.create_barplot_grouped),
    ]:
        if task_name not in selected_tasks:
            continue
        df = all_results_df[all_results_df['task'] == task_name].copy()
        df = df.dropna(axis=1, how="all")
        if df.empty:
            print(f"No {task_name} results in this results folder, skipping.")
        else:
            barplot_fn(df, plots_folder, task_label)
            nl2_results_table.create_results_table(df, plots_folder, task_label)

    ##############################################################################################
    #
    #     Overall Ranking
    #
    ##############################################################################################
    ranking_table.create_table(all_results_df, plots_folder, elo_table)


if __name__ == "__main__":
    typer.run(main)
