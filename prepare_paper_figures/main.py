######################################################
# Build all result tables and plots for the paper
######################################################
from pathlib import Path
from benchmark_src.results_processing import create_plots

import ranking_table

# ---- old TEmBed tasks (disabled by default) ----
import row_sim_plot_aggregated, row_sim_linechart_topk, row_sim_results_table, quadrant_chart_row
import col_sim_results_table, col_sim_bar_plot_per_dataset
import tabular_prediction_result_tables, tabular_prediction_barchart_binary, tabular_prediction_barchart_multiclass, tabular_prediction_barchart_regression, tablular_prediction_elo_table
import cell_bar_plot, cell_bar_plot_stacked, cell_results_table, quadrant_chart_cell
import triplet_row_results_table, triplet_row_bar_plot_original
import table_retrieval_tables as old_table_retrieval_tables

# ---- new paper modules ----
import retrieval_md_vs_csv_table
import shuffling_main_table
import shuffling_magnitude_table
import shuffling_size_table
import ttd_classifier_table
import ttd_classifier_bars
import headline_bars
import retrieval_per_dataset_bars
import retrieval_recall_line
import retrieval_rowlimit_bars
import shuffling_variation_heatmap
import shuffling_row_col_scatter
import shuffling_silhouette_bars
import cost_quality_quadrant
import serialization_deltas

# ---------------------------------------------------------------------------
# Color & name mappings for our 6 approaches × 2 serializations
# ---------------------------------------------------------------------------
color_mapping = {
    # GritLM
    ("GritLM", "embedding_model=GritLM_GritLM-7B,table_row_limit=0,table_serialization_format=markdown"):  "#1f77b4",
    ("GritLM", "embedding_model=GritLM_GritLM-7B,table_row_limit=100,table_serialization_format=markdown"): "#1f77b4",
    ("GritLM", "embedding_model=GritLM_GritLM-7B,table_row_limit=100,table_serialization_format=csv"):       "#aec7e8",
    # MiniLM
    ("sentence_transformer", "embedding_model=all-MiniLM-L6-v2,table_row_limit=0,table_serialization_format=markdown"):   "#ff7f0e",
    ("sentence_transformer", "embedding_model=all-MiniLM-L6-v2,table_row_limit=100,table_serialization_format=markdown"): "#ff7f0e",
    ("sentence_transformer", "embedding_model=all-MiniLM-L6-v2,table_row_limit=100,table_serialization_format=csv"):      "#ffbb78",
    # Granite-R2
    ("sentence_transformer", "embedding_model=ibm-granite_granite-embedding-english-r2,table_row_limit=0,table_serialization_format=markdown"):   "#9467bd",
    ("sentence_transformer", "embedding_model=ibm-granite_granite-embedding-english-r2,table_row_limit=100,table_serialization_format=markdown"): "#9467bd",
    ("sentence_transformer", "embedding_model=ibm-granite_granite-embedding-english-r2,table_row_limit=100,table_serialization_format=csv"):      "#c5b0d5",
    # HyTrel
    ("hytrel", "table_row_limit=0"):   "#17becf",
    ("hytrel", "table_row_limit=100"): "#17becf",
    # Hashing
    ("hashing", "n_features=32768,table_row_limit=0"):   "#d62728",
    ("hashing", "n_features=32768,table_row_limit=100"): "#d62728",
    ("hashing", "table_row_limit=100"):                  "#d62728",  # legacy config
    # # TF-IDF (disabled)
    # ("tfidf", "n_features=32768,table_row_limit=0"):   "#2ca02c",
    # ("tfidf", "n_features=32768,table_row_limit=100"): "#2ca02c",
    # ("tfidf", "table_row_limit=100"):                  "#2ca02c",  # legacy config
}

name_mapping = {
    # GritLM
    ("GritLM", "embedding_model=GritLM_GritLM-7B,table_row_limit=0,table_serialization_format=markdown"):  "GritLM (md)",
    ("GritLM", "embedding_model=GritLM_GritLM-7B,table_row_limit=100,table_serialization_format=markdown"): "GritLM (md)",
    ("GritLM", "embedding_model=GritLM_GritLM-7B,table_row_limit=100,table_serialization_format=csv"):       "GritLM (csv)",
    # MiniLM
    ("sentence_transformer", "embedding_model=all-MiniLM-L6-v2,table_row_limit=0,table_serialization_format=markdown"):   "MiniLM (md)",
    ("sentence_transformer", "embedding_model=all-MiniLM-L6-v2,table_row_limit=100,table_serialization_format=markdown"): "MiniLM (md)",
    ("sentence_transformer", "embedding_model=all-MiniLM-L6-v2,table_row_limit=100,table_serialization_format=csv"):      "MiniLM (csv)",
    # Granite-R2
    ("sentence_transformer", "embedding_model=ibm-granite_granite-embedding-english-r2,table_row_limit=0,table_serialization_format=markdown"):   "Granite-R2 (md)",
    ("sentence_transformer", "embedding_model=ibm-granite_granite-embedding-english-r2,table_row_limit=100,table_serialization_format=markdown"): "Granite-R2 (md)",
    ("sentence_transformer", "embedding_model=ibm-granite_granite-embedding-english-r2,table_row_limit=100,table_serialization_format=csv"):      "Granite-R2 (csv)",
    # HyTrel
    ("hytrel", "table_row_limit=0"):   "HyTrel",
    ("hytrel", "table_row_limit=100"): "HyTrel",
    # Hashing
    ("hashing", "n_features=32768,table_row_limit=0"):   "Hashing",
    ("hashing", "n_features=32768,table_row_limit=100"): "Hashing",
    ("hashing", "table_row_limit=100"):                  "Hashing",
    # # TF-IDF (disabled)
    # ("tfidf", "n_features=32768,table_row_limit=0"):   "TF-IDF",
    # ("tfidf", "n_features=32768,table_row_limit=100"): "TF-IDF",
    # ("tfidf", "table_row_limit=100"):                  "TF-IDF",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_FOLDER = Path("results_complete/results")
PLOTS_FOLDER = Path("./prepare_paper_figures/main_table_experiments")

# ---------------------------------------------------------------------------
# Toggle flags
# ---------------------------------------------------------------------------
# New paper tasks
TABLE_RETRIEVAL_PLOTS = True
TABLE_SHUFFLING_PLOTS = True
TABLE_TYPE_DETECTION_PLOTS = True
CROSS_TASK_PLOTS = True

# Old TEmBed tasks (disabled by default)
ROW_SIM_PLOTS = False
TRIPLET_PLOTS = False
COL_SIM_PLOTS = False
TABULAR_PREDICTION_PLOTS = False
CELL_SIM_PLOTS = False
OLD_TABLE_RETRIEVAL_PLOTS = False

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    results_folder = RESULTS_FOLDER
    assert results_folder.exists(), f"Could not find results folder at {results_folder}"

    plots_folder = PLOTS_FOLDER
    plots_folder.mkdir(exist_ok=True, parents=True)

    all_results_df = create_plots.gather_results_and_metrics(results_folder=results_folder)

    # Exclude TF-IDF from all tables, figures, and evaluations
    all_results_df = all_results_df[all_results_df["Approach"] != "tfidf"]

    print("Approaches:", all_results_df["Approach"].unique())
    print("Tasks:", all_results_df["task"].unique())

    all_results_df = all_results_df[
        ~all_results_df.set_index(['Approach', 'Configuration']).index.isin([
            ('sentence_transformer', 'embedding_model=BAAI_bge-base-en-v1.5,table_row_limit=100'),
            ('sentence_transformer', 'embedding_model=ibm-granite_granite-embedding-30m-english,table_row_limit=100'),
        ])
    ]

    all_results_df["color"] = all_results_df.apply(
        lambda row: color_mapping.get((row["Approach"], row["Configuration"]), "#000000"),
        axis=1,
    )
    all_results_df["chart_name"] = all_results_df.apply(
        lambda row: name_mapping.get((row["Approach"], row["Configuration"]), "TODO"),
        axis=1,
    )

    missing_chart_names = all_results_df[all_results_df["chart_name"] == "TODO"]
    if not missing_chart_names.empty:
        print("Warning: Missing chart names for:")
        for _, row in missing_chart_names.iterrows():
            print(f"  Approach: {row['Approach']}, Configuration: {row['Configuration']}")
        raise ValueError("Update name_mapping with chart names for missing combinations!")
    print("All Approach + Configuration combinations have chart names")

    missing_colors = all_results_df[all_results_df["color"] == "#000000"]
    if not missing_colors.empty:
        print("Warning: Missing colors for:")
        for _, row in missing_colors.iterrows():
            print(f"  Approach: {row['Approach']}, Configuration: {row['Configuration']}")
        raise ValueError("Update color_mapping with colors for missing combinations!")
    print("All Approach + Configuration combinations have colors")

    group_cols = ["Approach", "Configuration", "task", "dataset"]
    duplicates = all_results_df.duplicated(subset=group_cols, keep=False)
    if duplicates.any():
        print("Duplicate rows found:")
        print(all_results_df[duplicates])
        raise ValueError("Each Approach + Configuration + task combination must be unique!")
    print("All Approach + Configuration + task combinations are unique")

    # ======================================================================
    # OLD TASKS (disabled by default)
    # ======================================================================

    if ROW_SIM_PLOTS:
        df = all_results_df[all_results_df['task'] == "row_similarity_search"].copy()
        row_sim_plot_aggregated.create_barplot(df, plots_folder)
        row_sim_results_table.create_results_table(df, plots_folder)
        row_sim_linechart_topk.create_lineplot(df, plots_folder)
        quadrant_chart_row.build_quadrant_chart(df, plots_folder)
        quadrant_chart_row.build_quadrant_chart_vram(df, plots_folder)

    if COL_SIM_PLOTS:
        df = all_results_df[all_results_df['task'] == "column_similarity_search"].copy()
        col_sim_bar_plot_per_dataset.create_barplot(df, plots_folder)
        col_sim_results_table.create_results_table(df, plots_folder)

    if TRIPLET_PLOTS:
        df = all_results_df[all_results_df['task'] == "more_similar_than"].copy()
        df = df.dropna(axis=1, how="all")
        triplet_row_results_table.create_results_table(df, plots_folder)
        triplet_row_bar_plot_original.create_barplot(df, plots_folder)

    if CELL_SIM_PLOTS:
        df = all_results_df[all_results_df['task'] == "cell_task"].copy()
        cell_bar_plot.create_barplot(df, plots_folder)
        cell_bar_plot_stacked.create_barplot(df, plots_folder)
        cell_results_table.create_results_table(df, plots_folder)
        quadrant_chart_cell.build_quadrant_chart(df, plots_folder)

    if OLD_TABLE_RETRIEVAL_PLOTS:
        df = all_results_df[all_results_df['task'] == "table_retrieval"].copy()
        df = df.dropna(axis=1, how="all")
        old_table_retrieval_tables.create_results_table_small(df, plots_folder)

    if TABULAR_PREDICTION_PLOTS:
        df = all_results_df[all_results_df['task'] == "predictive_ml"].copy()
        df = df.dropna(axis=1, how="all")
        tabular_prediction_result_tables.create_results_table_binary_classification(df, plots_folder)
        tabular_prediction_result_tables.create_results_table_multiclass_classification(df, plots_folder)
        tabular_prediction_result_tables.create_results_table_regression(df, plots_folder)
        tabular_prediction_barchart_binary.create_barplot(df, plots_folder)
        tabular_prediction_barchart_multiclass.create_barplot_multiclass(df, plots_folder)
        tabular_prediction_barchart_regression.create_barplot_regression(df, plots_folder)
        elo_table = tablular_prediction_elo_table.create_elo_table(df, plots_folder)
    else:
        elo_table = None

    # ======================================================================
    # TASK 1 — Table Retrieval
    # ======================================================================
    if TABLE_RETRIEVAL_PLOTS:
        df = all_results_df[all_results_df['task'] == "table_retrieval"].copy()
        df = df.dropna(axis=1, how="all")

        # T7: Markdown vs CSV (rl=100, transformers only)
        retrieval_md_vs_csv_table.create_table(df, plots_folder)

        # P3: Per-dataset MRR bars
        retrieval_per_dataset_bars.create_barplot(df, plots_folder)

        # P4: Recall@k line chart
        retrieval_recall_line.create_lineplot(df, plots_folder)

        # P5: rl=0 vs rl=100 paired bars
        retrieval_rowlimit_bars.create_barplot(df, plots_folder)

    # ======================================================================
    # TASK 2 — Table Shuffling
    # ======================================================================
    if TABLE_SHUFFLING_PLOTS:
        df = all_results_df[all_results_df['task'] == "table_shuffling"].copy()
        df = df.dropna(axis=1, how="all")

        # T8: Shuffling main accuracy table (v0)
        shuffling_main_table.create_accuracy_table(df, plots_folder)

        # T10: Magnitude grid (v0/v1/v2, perturbation=both, avg datasets)
        shuffling_magnitude_table.create_table(df, plots_folder)

        # T11: Size ablation (v9-v14, exclude CKAN/ECB)
        shuffling_size_table.create_table(df, plots_folder)

        # P6: Variation heatmap
        shuffling_variation_heatmap.create_heatmap(df, plots_folder)

        # P9: Row vs column perturbation scatter
        shuffling_row_col_scatter.create_scatter(df, plots_folder)

        # P15: Silhouette score bars (v0, avg over datasets)
        shuffling_silhouette_bars.create_plot(df, plots_folder)

    # ======================================================================
    # TASK 3 — Table Type Detection
    # ======================================================================
    if TABLE_TYPE_DETECTION_PLOTS:
        df = all_results_df[all_results_df['task'] == "table_type_detection"].copy()
        df = df.dropna(axis=1, how="all")

        # T14: TTD classifier table
        ttd_classifier_table.create_table(df, plots_folder)

        # P11: TTD classifier faceted bars
        ttd_classifier_bars.create_barplot(df, plots_folder)

    # ======================================================================
    # CROSS-TASK PLOTS
    # ======================================================================
    if CROSS_TASK_PLOTS:
        # P1: Three-panel headline grouped bar
        headline_bars.create_plot(all_results_df, plots_folder)

        # P12: Cost vs quality quadrant
        cost_quality_quadrant.create_plot(all_results_df, plots_folder)

        # P13: Serialization deltas (all tasks)
        serialization_deltas.create_plot(all_results_df, plots_folder)

    # ======================================================================
    # OVERALL RANKING TABLE (T1)
    # ======================================================================
    ranking_table.create_table(all_results_df, plots_folder, elo_table)

    # ======================================================================
    # COMPILE .tex TABLES TO .pdf
    # ======================================================================
    import compile_tables
    print("Compiling .tex tables to PDF...")
    compile_tables.compile_all()
