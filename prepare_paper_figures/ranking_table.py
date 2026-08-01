import pandas as pd
from pathlib import Path
import numpy as np

from benchmark_src.results_processing.ranking import INITIAL_RATING

# Maps task_key -> metric, or task_key -> (task_col_value, metric)
# for subtasks that share a task column value (predictive_ml).
task_metrics = {
    "row_similarity_search": "MAP",
    "row_triplet_evaluation": "accuracy",
    "predictive_ml_regression": ("predictive_ml", "XGBoost_rmse (↓)"),
    "predictive_ml_multiclass": ("predictive_ml", "XGBoost_log_loss (↓)"),
    "predictive_ml_binary":     ("predictive_ml", "XGBoost_roc_auc_score (↑)"),
    "column_similarity_search": "MAP",
    "column_type_annotation": "macro_f1 (↑)",
    "schema_linking": "mean_map",
    "table_retrieval": "MAP@10",
    "table_similarity_search": "MAP",
    "table_shuffling": "TripletAccuracy",
    "cell_similarity_search": "MAP",
    "value_linking": "mean_map",
}

TASK_NAME_MAP = {
    "row_similarity_search": r"\makecell{Row \\ Sim. Search \\ (MAP)}",
    "row_triplet_evaluation": r"\makecell{Triplet \\ Evaluation \\ (Accuracy) }",
    "predictive_ml": r"\makecell{Tabular \\ Prediction\\ (ELO)}",
    "predictive_ml_regression": r"\makecell{Tabular \\ Prediction\\ (Regression)}",
    "predictive_ml_binary": r"\makecell{Tabular \\ Prediction\\ (Binary)}",
    "predictive_ml_multiclass": r"\makecell{Tabular \\ Prediction\\ (Multiclass)}",
    "column_similarity_search": r"\makecell{Column \\ Sim. Search \\ (MAP)}",
    "column_type_annotation": r"\makecell{Column Type \\ Annotation \\ (Macro F1)}",
    "schema_linking": r"\makecell{Schema \\ Linking \\ (MAP)}",
    "table_retrieval": r"\makecell{Table \\ Retrieval \\ (MAP@10)}",
    "table_similarity_search": r"\makecell{Table \\ Sim. Search \\ (MAP)}",
    "table_shuffling": r"\makecell{Table \\ Shuffling \\ (Accuracy)}",
    "cell_similarity_search": r"\makecell{Cell \\ Sim. Search \\ (MAP)}",
    "value_linking": r"\makecell{Value \\ Linking \\ (MAP)}",
    "Overall": "Overall",
}

MISSING_MULTIPLIER = 2  # multiplier applied to worst observed value for lower-is-better metrics

# row_triplet_evaluation's dataset list (configs/global_datasets.yaml) also includes several
# column-count/ablation variants used for robustness analyses elsewhere; the ranking table
# only reflects the two original datasets, matching the main-text figure.
ROW_TRIPLET_CORE_DATASETS = frozenset({"wikidata_books", "astronomical_objects"})

# Task-column values create_table() reads from all_results_df, derived from task_metrics
# (predictive_ml subtasks are skipped there and pulled in separately via the elo ranking
# argument instead, so they're excluded here).
RANKING_TASKS = frozenset(
    (spec[0] if isinstance(spec, tuple) else task_key)
    for task_key, spec in task_metrics.items()
    if task_key not in ("predictive_ml_regression", "predictive_ml_binary", "predictive_ml_multiclass")
)

def is_higher_better(metric_name: str) -> bool:
    return (
        "(↑)" in metric_name
        or "auc" in metric_name.lower()
        or "accuracy" in metric_name.lower()
        or "mrr" in metric_name.lower()
        or "map" in metric_name.lower()
        or "silhouette" in metric_name.lower()
    )


def compute_zscore(agg: pd.Series, higher_better: bool) -> pd.Series:
    """
    Convert a per-approach task-average metric into a z-score across the approaches
    that support the task (mean-centered, scaled by the cross-approach std), sign-flipped
    so a higher z-score always means better performance.

    Approaches with no data at all for the task (NaN in `agg`) are assigned a z-score
    exactly 1 standard deviation below the worst supporting approach -- the z-score
    analog of PENALTY_RANK's "one rank worse than the worst observed rank", expressed
    in the z-score's own natural unit, so narrow task coverage is penalized the same
    way it already is for the rank-based Overall column.
    """
    valid = agg.dropna()
    if valid.empty:
        return pd.Series(0.0, index=agg.index)

    std = valid.std()
    if not std or pd.isna(std):
        z = pd.Series(0.0, index=valid.index)
    else:
        z = (valid - valid.mean()) / std
        if not higher_better:
            z = -z

    z = z.reindex(agg.index)
    missing = z.isna()
    if missing.any() and (~missing).any():
        z[missing] = z[~missing].min() - 1.0
    return z


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

    # All base approach names
    all_approaches = sorted(df["chart_name"].str.replace(r"\*$", "", regex=True).unique())
    n_approaches = len(all_approaches)
    PENALTY_RANK = n_approaches + 1

    rankings = {}
    zscores = {}
    mean_values = {}
    cannot_do = {}
    partial_do = {}

    # ------------------------------------------------------------------
    # Loop over tasks
    # ------------------------------------------------------------------
    for task_key, spec in task_metrics.items():

        # For predictive ML tasks, we will skip individual subtasks
        if task_key in ["predictive_ml_regression", "predictive_ml_binary", "predictive_ml_multiclass"]:
            continue

        # Unpack spec
        if isinstance(spec, tuple):
            task_filter, metric = spec
        else:
            task_filter, metric = task_key, spec

        metric_col = metric

        # Filter to rows for this task
        task_df = df[df["task"] == task_filter].copy()
        if task_key == "row_triplet_evaluation":
            task_df = task_df[task_df["dataset"].isin(ROW_TRIPLET_CORE_DATASETS)]

        if task_df.empty or metric_col not in task_df.columns:
            cannot_do[task_key] = set(all_approaches)
            rankings[task_key] = pd.Series(PENALTY_RANK, index=all_approaches)
            zscores[task_key] = pd.Series(0.0, index=all_approaches)
            continue

        higher_better = is_higher_better(metric)
        is_predictive_ml_task = False  # predictive ML subtasks handled separately

        agg, partial_approaches = aggregate_per_task(
            task_df, metric_col, higher_better, best_config=is_predictive_ml_task
        )

        # Reindex to all approaches
        agg = agg.reindex(all_approaches)

        missing = set(agg[agg.isna()].index)
        cannot_do[task_key] = missing
        partial_do[task_key] = partial_approaches

        mean_values[task_key] = agg.copy()
        zscores[task_key] = compute_zscore(agg, higher_better)

        # Rank among approaches with data
        ranks = agg.rank(ascending=not higher_better, method="average", na_option="keep")
        for approach in missing:
            ranks[approach] = PENALTY_RANK
        rankings[task_key] = ranks

    # ------------------------------------------------------------------
    # Add combined predictive ML ELO ranking
    # ------------------------------------------------------------------
    # predictive_ml_elo_ranking_df: expects columns ['chart_name', 'elo_score_task'].
    # Unlike other task columns, ELO scores from the default and '*' (row-embeddings)
    # configs of the same approach are two distinct tournament participants and aren't
    # meaningfully averaged -- use the default config's score, falling back to the '*'
    # config only for approaches with no default entry at all (currently just TabuLa-8B).
    elo_df = predictive_ml_elo_ranking_df.copy()
    is_starred = elo_df["chart_name"].str.endswith("*")
    elo_df["base_name"] = elo_df["chart_name"].str.replace(r"\*$", "", regex=True)

    # ELO: higher better
    default_series = elo_df.loc[~is_starred].set_index("base_name")["elo_score_task"]
    starred_series = elo_df.loc[is_starred].set_index("base_name")["elo_score_task"]
    elo_series = default_series.combine_first(starred_series)
    elo_series = elo_series.reindex(all_approaches)

    missing = set(elo_series[elo_series.isna()].index)
    cannot_do["predictive_ml"] = missing
    partial_do["predictive_ml"] = set()  # assume no partial info
    mean_values["predictive_ml"] = elo_series.copy()
    zscores["predictive_ml"] = compute_zscore(elo_series, higher_better=True)

    # Rank ELO descending (higher is better)
    ranks = elo_series.rank(ascending=False, method="average", na_option="keep")
    for approach in missing:
        ranks[approach] = PENALTY_RANK
    rankings["predictive_ml"] = ranks

    # ------------------------------------------------------------------
    # Build ranking DataFrame
    # ------------------------------------------------------------------
    ranking_df = pd.DataFrame(rankings, index=all_approaches)
    ranking_df.index.name = "Approach"

    ranking_df["Overall"] = ranking_df.mean(axis=1)

    zscore_df = pd.DataFrame(zscores, index=all_approaches)
    ranking_df["Overall Z-Score"] = zscore_df.mean(axis=1)

    # Replace task keys with display names (TASK_NAME_MAP)
    task_keys_for_display = {
        k: TASK_NAME_MAP.get(k, k)
        for k in ranking_df.columns
        if k not in ("Overall", "Overall Z-Score")
    }
    ranking_df = ranking_df.rename(columns=task_keys_for_display)
    ranking_df = ranking_df.sort_values("Overall").round(2)
    ranking_df = ranking_df.reset_index()

    # ------------------------------------------------------------------
    # Export the raw per-task metric values behind the rank/z-score table above,
    # so they can be spot-checked independently of the table's formatting.
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

    # sort value cols (order: same as TASK_NAME_MAP)
    value_cols = sorted(value_cols, key=lambda x: list(TASK_NAME_MAP.values()).index(x) if x in TASK_NAME_MAP.values() else len(TASK_NAME_MAP))


    display_name_to_task = {v: k for k, v in TASK_NAME_MAP.items()}
    cannot_do_display = {
        col: cannot_do.get(display_name_to_task.get(col, ""), set())
        for col in value_cols
    }
    partial_do_display = {
        col: partial_do.get(display_name_to_task.get(col, ""), set())
        for col in value_cols
    }

    # Every column except "Overall Z-Score" holds a rank (lower is better); the
    # z-score column holds a magnitude where higher is better.
    col_best, col_second = {}, {}
    for col in value_cols:
        missing_names = cannot_do_display.get(col, set())
        eligible = df_out.loc[~df_out["Approach"].isin(missing_names), col]
        col_lower_better = col != "Overall Z-Score"
        if eligible.empty:
            best = None
        else:
            best = eligible.min() if col_lower_better else eligible.max()
        col_best[col] = best
        remaining = eligible[eligible != best] if best is not None else eligible
        if remaining.empty:
            second = None
        else:
            second = remaining.min() if col_lower_better else remaining.max()
        col_second[col] = second

    # Precompute the widest ELO delta (digits only, sign excluded) so narrower
    # deltas can be padded with an invisible zero and stay right-aligned with
    # wider ones in the same column, mirroring the rank zero-padding below.
    elo_raw_series = mean_values.get("predictive_ml")
    elo_delta_max_digits = 0
    if elo_raw_series is not None:
        valid_deltas = (elo_raw_series - INITIAL_RATING).dropna()
        if not valid_deltas.empty:
            elo_delta_max_digits = valid_deltas.abs().round().astype(int).astype(str).str.len().max()

    with open(plots_folder / "overall_ranking_table.tex", "w") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l {'c '*len(value_cols)}@{{}}}}\n"
            "\\toprule\n"
        )
        # "Overall"/"Overall Z-Score" stay as plain column names throughout (used as
        # lookup keys above); only the header cell text gets the two-line makecell form.
        header_display = {
            "Overall": r"\makecell{Overall \\ Rank}",
            "Overall Z-Score": r"\makecell{Overall \\ Z-Score}",
        }
        header_cells = [header_display.get(col, col) for col in value_cols]
        f.write("Approach & " + " & ".join(header_cells) + " \\\\\n")
        f.write("\\midrule\n")

        for _, row in df_out.iterrows():
            approach = row["Approach"]
            formatted = []

            for col in value_cols:
                if approach in cannot_do_display.get(col, set()):
                    formatted.append("---")
                    continue

                v = row[col]
                if pd.isna(v):
                    formatted.append("-")
                    continue

                # Individual task ranks are whole numbers unless tied (average-rank
                # method), so drop the trailing ".00" to keep cells narrow; the
                # Overall/Overall Z-Score summary columns are means and always shown
                # to 2 decimals.
                is_rank_col = col != "Overall Z-Score"
                if col in ("Overall", "Overall Z-Score"):
                    val_str = f"{v:.2f}"
                else:
                    val_str = f"{v:.2f}".rstrip("0").rstrip(".")

                # Pad single-digit ranks with an invisible leading zero so ranks stay
                # right-aligned with double-digit ranks in the same column. Kept outside
                # the bold/underline wrap below so \underline doesn't draw its stroke
                # under the invisible padding too.
                needs_zero_pad = is_rank_col and "." not in val_str and len(val_str) == 1

                if col_best[col] is not None and v == col_best[col]:
                    val_str = f"\\textbf{{{val_str}}}"
                elif col_second[col] is not None and v == col_second[col]:
                    val_str = f"\\underline{{{val_str}}}"

                if needs_zero_pad:
                    val_str = r"\phantom{0}" + val_str

                # Mark approaches that had some datasets imputed with a dagger; pad
                # with an invisible dagger otherwise so cell width in a column doesn't
                # depend on whether the dagger is actually shown.
                if is_rank_col:
                    if approach in partial_do_display.get(col, set()):
                        val_str = val_str + "$^\\dagger$"
                    else:
                        val_str = val_str + r"\phantom{$^\dagger$}"

                # For individual task columns (not the Overall/Overall Z-Score
                # summary columns), append the raw metric value next to the rank
                # so within-task magnitude isn't lost.
                task_key = display_name_to_task.get(col, "")
                if task_key not in ("", "Overall"):
                    raw_series = mean_values.get(task_key)
                    if raw_series is not None and approach in raw_series.index:
                        raw_val = raw_series[approach]
                        if not pd.isna(raw_val):
                            # ELO's absolute value is anchored to the arbitrary
                            # INITIAL_RATING starting point, so report the delta from
                            # it (signed) instead -- matches the standalone ELO table.
                            pad_str = ""
                            if task_key == "predictive_ml":
                                delta = raw_val - INITIAL_RATING
                                raw_str = f"{delta:+.0f}"
                                digit_count = len(str(int(round(abs(delta)))))
                                pad = elo_delta_max_digits - digit_count
                                if pad > 0:
                                    pad_str = r"\phantom{0}" * pad
                            else:
                                raw_str = f"{raw_val:.2f}"
                            val_str = f"{val_str} {pad_str}({raw_str})"

                formatted.append(val_str)

            f.write(f"{approach} & {' & '.join(formatted)} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular*}\n")
        f.write(
            "\\caption{Overall Ranking of approaches across tasks. Each task cell shows the "
            "approach's rank followed by its raw metric value in parentheses; for Tabular "
            "Prediction this is the ELO delta from the initial rating of 1500. "
            "\\text{---} indicates the approach does not support the task. "
            "$^\\dagger$ indicates the approach could not complete all datasets for the task; "
            "missing datasets were imputed with a worst-case value. "
            "Overall Z-Score reports the mean, per-task z-score of the raw metric across "
            "approaches (sign-flipped so higher is always better), preserving the magnitude of "
            "cross-approach differences that the rank-based Overall column collapses; approaches "
            "that do not support a task are assigned a z-score 1 standard deviation below the "
            "worst supporting approach on that task.}\n"
        )
        f.write("\\label{tab:overall_ranking}\n")
        f.write("\\end{table*}\n")

    return ranking_df