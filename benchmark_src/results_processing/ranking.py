import random
import pandas as pd
import numpy as np
import hashlib
from collections import defaultdict
from typing import Tuple, List, Dict
import itertools

from benchmark_src.results_processing import results_helper
from benchmark_src.utils import cfg_utils

INITIAL_RATING = 1500.0

def compute_dominance_and_avg_rank(df: pd.DataFrame, metric_col: str = 'In top-1 [%]_mean') -> pd.DataFrame:
    """
    Compute for each task/approach/configuration:
      - num_datasets_best: number of datasets where it reached the best score (including ties)
      - num_datasets_sole_best: number of datasets where it was the sole best performer
      - datasets_sole_best: sorted list of datasets where it was the sole best performer
      - avg_rank: average rank across datasets (lower rank = better)
    """
    records = []

    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        for dataset in task_df['dataset'].unique():
            dataset_df = task_df[task_df['dataset'] == dataset].copy()
            max_val = dataset_df[metric_col].max()
            best_rows = dataset_df[dataset_df[metric_col] == max_val]

            # Compute ranks
            dataset_df['rank'] = dataset_df[metric_col].rank(ascending=False, method='min')

            for _, row in dataset_df.iterrows():
                # Check if sole best
                is_sole_best = int(len(best_rows) == 1 and row[metric_col] == max_val)
                # Check if best (tie included)
                is_best = int(row[metric_col] == max_val)

                # Append record
                records.append({
                    'task': task,
                    'dataset': dataset,
                    'Approach': row['Approach'],
                    'Configuration': row['Configuration'],
                    'is_best': is_best,
                    'is_sole_best': is_sole_best,
                    'rank': row['rank']
                })

    df_records = pd.DataFrame(records)

    # Aggregate per approach/configuration
    agg = (df_records.groupby(['task', 'Approach', 'Configuration'])
           .agg(
               num_datasets_best=('is_best', 'sum'),
               num_datasets_sole_best=('is_sole_best', 'sum'),
               datasets_sole_best=('dataset', lambda x: sorted(list(x[df_records.loc[x.index, 'is_sole_best'] == 1]))),
               avg_rank=('rank', 'mean')
           )
           .reset_index())

    return agg



def compute_pairwise_wins(df: pd.DataFrame, metric_col: str = 'In top-1 [%]_mean') -> pd.DataFrame:
    """
    Compute pairwise wins for all approach/configuration combinations.
    Output: matrix of #datasets where row beats column.
    """
    approaches = df[['Approach', 'Configuration']].drop_duplicates()
    approach_keys = [tuple(x) for x in approaches.values]

    # initialize dictionary
    pairwise_counts = defaultdict(lambda: defaultdict(int))

    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        for dataset in task_df['dataset'].unique():
            dataset_df = task_df[task_df['dataset'] == dataset]
            vals = [(row['Approach'], row['Configuration'], row[metric_col]) for _, row in dataset_df.iterrows()]
            for (a_name, a_cfg, a_val), (b_name, b_cfg, b_val) in itertools.combinations(vals, 2):
                if a_val > b_val:
                    pairwise_counts[(a_name, a_cfg)][(b_name, b_cfg)] += 1
                elif b_val > a_val:
                    pairwise_counts[(b_name, b_cfg)][(a_name, a_cfg)] += 1
                # ties are ignored or could count as 0.5

    # convert to DataFrame
    rows = []
    for a_key in approach_keys:
        for b_key in approach_keys:
            if a_key == b_key:
                continue
            rows.append({
                'Approach_A': a_key[0],
                'Config_A': a_key[1],
                'Approach_B': b_key[0],
                'Config_B': b_key[1],
                'num_datasets_A_beats_B': pairwise_counts[a_key].get(b_key, 0)
            })
    return pd.DataFrame(rows)

def update_elo(rating_a: float, rating_b: float, score_a: float, score_b: float, k_factor: float = 32.0) -> Tuple[float, float]:
    # Validate inputs
    if not isinstance(rating_a, (int, float)) or not isinstance(rating_b, (int, float)):
        raise TypeError("Ratings must be numeric")
    if not (isinstance(score_a, (int, float)) and isinstance(score_b, (int, float))):
        raise TypeError("Scores must be numeric")
    if not (0.0 <= score_a <= 1.0 and 0.0 <= score_b <= 1.0):
        raise ValueError("Scores must be between 0 and 1")
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    new_a = rating_a + k_factor * (score_a - expected_a)
    new_b = rating_b + k_factor * (score_b - expected_b)
    return new_a, new_b

def yield_matches(dataset_df: pd.DataFrame, metric: str, higher_is_better: bool, eps: float = 1e-6):
    """
    Yield pairwise match outcomes for rows in dataset_df based on metric column.
    Yields ((approach, config), (approach, config), score_a, score_b)
    """
    vals = dataset_df[['Approach', 'Configuration', metric]].dropna().values
    for i in range(len(vals)):
        a_name, a_cfg, a_val = vals[i]
        for j in range(i + 1, len(vals)):
            b_name, b_cfg, b_val = vals[j]
            # choose winner according to higher_is_better
            diff = (a_val - b_val) if higher_is_better else (b_val - a_val)
            if abs(diff) < eps:
                yield (a_name, a_cfg), (b_name, b_cfg), 0.5, 0.5
            elif diff > 0:
                yield (a_name, a_cfg), (b_name, b_cfg), 1.0, 0.0
            else:
                yield (a_name, a_cfg), (b_name, b_cfg), 0.0, 1.0


def build_task_metrics_map(df: pd.DataFrame) -> Dict[str, List[Tuple[str, bool]]]:
    performance_cols = results_helper.performance_cols  # dict metric -> 'higher_is_better'|'lower_is_better'

    # Build mapping task -> list of (metric_mean_col, higher_is_better)
    task_metrics: Dict[str, List[Tuple[str, bool]]] = {}
    for task in df['task'].unique():
        task_cfg = cfg_utils.load_task_config(task)
        elo_metric = task_cfg.get('elo_metric', None)
        if elo_metric is None:
            continue
        if isinstance(elo_metric, str):
            elo_metric = [elo_metric]
        metric_infos: List[Tuple[str, bool]] = []
        for m in elo_metric:
            if m not in performance_cols:
                raise ValueError(f"ELO metric '{m}' for task '{task}' not found in performance columns")
            higher = performance_cols[m] == 'higher_is_better'
            metric_infos.append((f"{m}_mean", higher))
        if metric_infos:
            task_metrics[task] = metric_infos

    return task_metrics

def get_elo_scores_for_task(
    task: str,
    df: pd.DataFrame,
    task_metrics: Dict[str, List[Tuple[str, bool]]],
    seed: int = 123,
    k_factor: float = 32.0,
) -> pd.DataFrame:

    if task not in task_metrics:
        print(f"No ELO metrics defined for task {task}")
        return pd.DataFrame()

    metrics_for_task = task_metrics[task]

    print(f"Computing ELO for task {task} using metrics: {metrics_for_task}")

    task_df = df[df["task"] == task]

    combos = task_df[["Approach", "Configuration"]].drop_duplicates()
    if len(combos) < 2:
        print(f"Not enough approach/config combos for task {task}")
        return pd.DataFrame()

    per_dataset_records = []

    for dataset in sorted(task_df["dataset"].unique()):
        dataset_df = task_df[task_df["dataset"] == dataset].copy()

        # initialize ratings for all combos in this dataset
        present_combos = {
            (row["Approach"], row["Configuration"])
            for _, row in dataset_df[["Approach", "Configuration"]]
            .drop_duplicates()
            .iterrows()
        }
        ratings = {combo: INITIAL_RATING for combo in present_combos}

        matches = []

        for metric_col, higher_is_better in metrics_for_task:

            if metric_col not in dataset_df.columns:
                print(f"Metric {metric_col} missing in dataset {dataset}, skipping metric")
                continue

            # if task is "predicitive_ml", make a temporary copy and write the value from the column "approach_" + metric_col without the first part before _ into the metric col
            if task == "predictive_ml":
                metric_has_data = dataset_df[metric_col].notna().any()
                if metric_has_data:
                    approach_metric_col = "approach_" + metric_col.split("_", 1)[1]

                    # merge the values from old metric_col and new approach_metric_col, if the column exists
                    # only copy values where approach_metric_col is not null
                    if approach_metric_col in dataset_df.columns:
                        dataset_df.loc[
                            dataset_df[approach_metric_col].notna(), metric_col
                        ] = dataset_df.loc[
                            dataset_df[approach_metric_col].notna(), approach_metric_col
                        ]


            subset = dataset_df[["Approach", "Configuration", metric_col]].dropna()
            if len(subset) < 2:
                continue

            matches.extend(
                yield_matches(subset, metric_col, higher_is_better)
            )

        if not matches:
            print(f"No valid matches for dataset {dataset}, using initial ratings")
        else:
            dataset_hash = int(
                hashlib.md5(f"{task}-{dataset}".encode()).hexdigest(), 16
            ) % 10000
            rng = random.Random(seed + dataset_hash)
            rng.shuffle(matches)

            for (a_name, a_cfg), (b_name, b_cfg), sa, sb in matches:
                ra, rb = ratings[(a_name, a_cfg)], ratings[(b_name, b_cfg)]
                ra_new, rb_new = update_elo(ra, rb, sa, sb, k_factor=k_factor)
                ratings[(a_name, a_cfg)] = ra_new
                ratings[(b_name, b_cfg)] = rb_new

        for (approach, config), rating in ratings.items():
            per_dataset_records.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "Approach": approach,
                    "Configuration": config,
                    "elo_score_dataset": float(rating),
                }
            )

    if not per_dataset_records:
        return pd.DataFrame()

    task_dataset_df = pd.DataFrame(per_dataset_records)

    grouped = (
        task_dataset_df
        .groupby(["task", "Approach", "Configuration"], as_index=False)
        ["elo_score_dataset"]
        .mean()
        .rename(columns={"elo_score_dataset": "elo_score_task"})
    )

    return grouped


def compute_elo_scores(df: pd.DataFrame, seed: int = 123, k_factor: float = 32.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute ELO scores and return DataFrames:
      - per_task_df:   columns [task, Approach, Configuration, elo_score_task]
      - overall_df:    columns [Approach, Configuration, elo_score_overall]
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    print("%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Overview of all results:")
    overview = df.groupby(['task', 'Approach', 'Configuration'])['dataset'].nunique().reset_index(name='num_datasets')
    print(overview)

    task_metrics = build_task_metrics_map(df)

    print(f"Build task metrics: {task_metrics}")

    per_task_records = []

    for task in sorted(df['task'].unique()):
        print(f"Computing ELO scores for task: {task}")
        grouped = get_elo_scores_for_task(task, df, task_metrics, seed=seed, k_factor=k_factor)
        # if df is empty, skip
        if grouped.empty:
            print(f"  No ELO scores computed for task {task} (insufficient data?)")
            continue

        for _, row in grouped.iterrows():
            per_task_records.append({
                'task': row['task'],
                'Approach': row['Approach'],
                'Configuration': row['Configuration'],
                'elo_score_task': float(row['elo_score_task'])
            })

    # build DataFrames
    per_task_df = pd.DataFrame(per_task_records)

    if per_task_df.empty:
        overall_df = pd.DataFrame()
    else:
        # overall: average elo_score_task across tasks per approach/config
        overall_df = per_task_df.groupby(['Approach', 'Configuration'])['elo_score_task'].mean().reset_index()
        overall_df = overall_df.rename(columns={'elo_score_task': 'elo_score_overall'}).sort_values(by='elo_score_overall', ascending=False).reset_index(drop=True)


    if not per_task_df.empty:
        per_task_df = per_task_df.sort_values(['task', 'elo_score_task'], ascending=[True, False]).reset_index(drop=True)

    print("%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Overview ELO Scores:")
    print(overall_df)

    return per_task_df, overall_df