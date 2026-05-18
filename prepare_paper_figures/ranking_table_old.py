import pandas as pd
from pathlib import Path
import numpy as np

# Maps task_key -> metric, or task_key -> (task_col_value, metric)
# for subtasks that share a task column value (predictive_ml).
task_metrics = {
    "table_retrieval": "MRR",
    "column_similarity_search": "MRR",
    "more_similar_than": "accuracy",
    "predictive_ml_regression": ("predictive_ml", "LinearRegression_rmse (↓)"),
    "predictive_ml_binary":     ("predictive_ml", "XGBoost_roc_auc_score (↑)"),
    "predictive_ml_multiclass": ("predictive_ml", "XGBoost_log_loss (↓)"),
    "row_similarity_search": "MRR",
    "cell_task": "accuracy",
}

TASK_NAME_MAP = {
    "table_retrieval": r"\makecell{Table \\ Retrieval}",
    "column_similarity_search": r"\makecell{Column \\ Similarity \\ Search}",
    "more_similar_than": r"\makecell{Triplet \\ Evaluation}",
    "predictive_ml_regression": r"\makecell{Tabular \\ Prediction\\ (Regression)}",
    "predictive_ml_binary": r"\makecell{Tabular \\ Prediction\\ (Binary)}",
    "predictive_ml_multiclass": r"\makecell{Tabular \\ Prediction\\ (Multiclass)}",
    "row_similarity_search": r"\makecell{Row \\ Similarity \\ Search}",
    "cell_task": r"\makecell{Cell Level\\ Retrieval}",
    "Overall": "Overall",
}

MISSING_MULTIPLIER = 2  # multiplier applied to worst observed value for lower-is-better metrics

def is_higher_better(metric_name: str) -> bool:
    return (
        "(↑)" in metric_name
        or "auc" in metric_name.lower()
        or "accuracy" in metric_name.lower()
        or "mrr" in metric_name.lower()
        or "silhouette" in metric_name.lower()
    )


def aggregate_per_task(
    task_df: pd.DataFrame,
    metric_col: str,
    higher_better: bool,
    best_config: bool = False,
) -> tuple[pd.Series, set]:
    """
    Aggregate metric_col for each base approach:

    1. Strip trailing * to get base_name (X and X* share a base_name).
    2. Per dataset:
       - If best_config=True (predictive ML): take the best value across all
         configurations and variants (X and X*).
       - Otherwise: average across all configurations and variants.
    3. Impute missing (approach, dataset) combinations with a sentinel:
       - higher-is-better: 0
       - lower-is-better: worst observed value across all approaches/datasets × MISSING_MULTIPLIER
    4. Average the per-dataset values across datasets.

    Returns:
        - Series indexed by base_name with averaged metric values
        - set of base_names that had at least one dataset imputed (partial completers)
    """
    df = task_df.copy()
    df["base_name"] = df["chart_name"].str.replace(r"\*$", "", regex=True)

    all_base_names = df["base_name"].unique()
    # Only consider datasets where at least one approach has a valid value for
    # this specific metric. This is critical for predictive ML subtasks: a dataset
    # that only has binary targets should not be counted as "missing" for the
    # multiclass metric — it simply doesn't apply to that subtask.
    all_datasets = df.loc[df[metric_col].notna(), "dataset"].unique()

    # Step 2: aggregate per (base_name, dataset)
    if best_config:
        if higher_better:
            per_dataset = df.groupby(["base_name", "dataset"])[metric_col].max()
        else:
            per_dataset = df.groupby(["base_name", "dataset"])[metric_col].min()
    else:
        per_dataset = df.groupby(["base_name", "dataset"])[metric_col].mean()

    # Step 3: reindex to full (base_name x dataset) grid so missing combos become NaN
    full_index = pd.MultiIndex.from_product(
        [all_base_names, all_datasets], names=["base_name", "dataset"]
    )
    per_dataset = per_dataset.reindex(full_index)

    # Identify partial completers: approaches missing at least one dataset but not all
    missing_mask = per_dataset.isna()
    missing_per_approach = missing_mask.groupby("base_name").sum()   # count of missing datasets
    total_datasets = len(all_datasets)
    partial_approaches = set(
        missing_per_approach[
            (missing_per_approach > 0) & (missing_per_approach < total_datasets)
        ].index
    )

    # Impute missing (approach, dataset) pairs with a sentinel worst-case value:
    # - higher-is-better: 0
    # - lower-is-better: worst observed value across all approaches/datasets × MISSING_MULTIPLIER
    if higher_better:
        worst_value = 0.0
    else:
        worst_observed = per_dataset.max()  # max = worst for ↓ metrics
        worst_value = worst_observed * MISSING_MULTIPLIER
    per_dataset = per_dataset.fillna(worst_value)

    # Step 4: average per-dataset values across datasets
    avg_across_datasets = per_dataset.groupby("base_name").mean()

    return avg_across_datasets, partial_approaches  # Series indexed by base_name


def create_table(all_results_df: pd.DataFrame, plots_folder: Path, predictive_ml_elo_ranking_df: pd.DataFrame):

    df = all_results_df[all_results_df["chart_name"] != "Baseline"].copy()

    # All base approach names (strip * for canonical names)
    all_approaches = sorted(
        df["chart_name"].str.replace(r"\*$", "", regex=True).unique()
    )
    n_approaches = len(all_approaches)
    PENALTY_RANK = n_approaches + 1

    rankings = {}      # task_key -> Series[base_name -> rank]
    mean_values = {}   # task_key -> Series[base_name -> mean metric value]
    cannot_do = {}     # task_key -> set of base approach names
    partial_do = {}    # task_key -> set of base approach names that had imputed datasets

    # ------------------------------------------------------------------
    # Loop over tasks
    # ------------------------------------------------------------------
    for task_key, spec in task_metrics.items():

        # Unpack spec
        if isinstance(spec, tuple):
            task_filter, metric = spec
        else:
            task_filter, metric = task_key, spec

        metric_col = f"{metric}_mean"

        # Filter to rows for this task
        task_df = df[df["task"] == task_filter].copy()

        if task_df.empty or metric_col not in task_df.columns:
            cannot_do[task_key] = set(all_approaches)
            rankings[task_key] = pd.Series(PENALTY_RANK, index=all_approaches)
            continue

        higher_better = is_higher_better(metric)

        # Predictive ML: best config per dataset; all other tasks: mean across configs
        is_predictive_ml = task_filter == "predictive_ml"
        agg, partial_approaches = aggregate_per_task(task_df, metric_col, higher_better, best_config=is_predictive_ml)

        # Reindex to all approaches so missing ones become NaN
        agg = agg.reindex(all_approaches)

        missing = set(agg[agg.isna()].index)
        cannot_do[task_key] = missing
        partial_do[task_key] = partial_approaches

        # Store raw mean values for debugging
        mean_values[task_key] = agg.copy()

        # Rank among approaches that have data
        ranks = agg.rank(
            ascending=not higher_better,
            method="average",
            na_option="keep",
        )

        # Fixed penalty for missing approaches
        for approach in missing:
            ranks[approach] = PENALTY_RANK

        rankings[task_key] = ranks

    # ------------------------------------------------------------------
    # Build ranking DataFrame
    # ------------------------------------------------------------------
    ranking_df = pd.DataFrame(rankings, index=all_approaches)
    ranking_df.index.name = "Approach"

    ranking_df["Overall"] = ranking_df.mean(axis=1)
    ranking_df = ranking_df.rename(columns=TASK_NAME_MAP)
    ranking_df = ranking_df.sort_values("Overall").round(2)
    ranking_df = ranking_df.reset_index()  # brings "Approach" in as a column

    # ------------------------------------------------------------------
    # Save mean values DataFrame for debugging
    # ------------------------------------------------------------------
    mean_df = pd.DataFrame(mean_values, index=all_approaches)
    mean_df.index.name = "Approach"
    mean_df = mean_df.reset_index()
    mean_df.to_csv(plots_folder / "overall_ranking_mean_values.csv", index=False)

    # ------------------------------------------------------------------
    # Write LaTeX table
    # ------------------------------------------------------------------
    df_out = ranking_df.copy()
    value_cols = df_out.columns[1:]  # everything after "Approach"

    display_name_to_task = {v: k for k, v in TASK_NAME_MAP.items()}
    cannot_do_display = {
        col: cannot_do.get(display_name_to_task.get(col, ""), set())
        for col in value_cols
    }
    partial_do_display = {
        col: partial_do.get(display_name_to_task.get(col, ""), set())
        for col in value_cols
    }

    # Best / second-best per column, excluding cannot-do approaches
    col_best, col_second = {}, {}
    for col in value_cols:
        missing_names = cannot_do_display.get(col, set())
        eligible = df_out.loc[~df_out["Approach"].isin(missing_names), col]
        best = eligible.min() if not eligible.empty else None
        col_best[col] = best
        remaining = eligible[eligible != best] if best is not None else eligible
        col_second[col] = remaining.min() if not remaining.empty else None

    with open(plots_folder / "overall_ranking_table.tex", "w") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            f"\\begin{{tabular*}}{{\\textwidth}}{{l"
            + "c" * len(value_cols)
            + "}\n"
            "\\hline\n"
        )
        f.write("Approach & " + " & ".join(value_cols) + " \\\\\n")
        f.write("\\hline\n")

        for _, row in df_out.iterrows():
            approach = row["Approach"]
            formatted = []

            for col in value_cols:
                if approach in cannot_do_display.get(col, set()):
                    formatted.append("---$^\\dagger$")
                    continue

                v = row[col]
                if pd.isna(v):
                    formatted.append("-")
                    continue

                val_str = f"{v:.2f}"
                if col_best[col] is not None and v == col_best[col]:
                    val_str = f"\\textbf{{{val_str}}}"
                elif col_second[col] is not None and v == col_second[col]:
                    val_str = f"\\underline{{{val_str}}}"

                # Mark approaches that had some datasets imputed
                if approach in partial_do_display.get(col, set()):
                    val_str = val_str + "$^\\dagger$"

                formatted.append(val_str)

            f.write(f"{approach} & {' & '.join(formatted)} \\\\\n")

        f.write("\\hline\n\\end{tabular*}\n")
        f.write(
            "\\caption{Overall Ranking of approaches across tasks. "
            "\\text{---} indicates the approach does not support the task. "
            "$^\\dagger$ indicates the approach could not complete all datasets for the task; "
            "missing datasets were imputed with a worst-case value.}\n"
        )
        f.write("\\label{tab:overall_ranking}\n")
        f.write("\\end{table*}\n")

    return ranking_df