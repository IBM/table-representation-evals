import random
import pandas as pd
import numpy as np
import hashlib
from collections import defaultdict
from typing import Tuple, List, Dict

from benchmark_src.results_processing import results_helper
from benchmark_src.utils import cfg_utils

INITIAL_RATING = 1500.0

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

    per_dataset_records = []
    per_task_records = []

    for task in sorted(df['task'].unique()):
        if task not in task_metrics:
            # no elo metrics defined for this task
            continue
        metrics_for_task = task_metrics[task]
        task_df = df[df['task'] == task]
        # get all approach/config combos present for this task
        combos = task_df[['Approach', 'Configuration']].drop_duplicates()
        if combos.shape[0] < 2:
            # nothing to compare
            continue

        # collect dataset-level ELOs for this task
        dataset_level_scores: Dict[str, Dict[Tuple[str, str], float]] = {}

        for dataset in sorted(task_df['dataset'].unique()):
            dataset_df = task_df[task_df['dataset'] == dataset]
            # initialize ratings for combos present in this dataset
            ratings = defaultdict(lambda: INITIAL_RATING)
            matches = []
            for metric_col, higher_is_better in metrics_for_task:
                if metric_col not in dataset_df.columns:
                    continue
                subset = dataset_df[['Approach', 'Configuration', metric_col]].dropna()
                if subset.shape[0] < 2:
                    continue
                matches.extend(list(yield_matches(subset, metric_col, higher_is_better)))
            # deterministic shuffle
            task_hash = int(hashlib.md5(task.encode("utf-8")).hexdigest(), 16) % 10000
            rng = random.Random(seed + task_hash)
            rng.shuffle(matches)
            # apply ELO updates
            for (a_name, a_cfg), (b_name, b_cfg), sa, sb in matches:
                key_a = (a_name, a_cfg)
                key_b = (b_name, b_cfg)
                ra = ratings[key_a]
                rb = ratings[key_b]
                ra_new, rb_new = update_elo(ra, rb, sa, sb, k_factor=k_factor)
                ratings[key_a] = ra_new
                ratings[key_b] = rb_new
            # store final ratings for this dataset
            dataset_level_scores[dataset] = dict(ratings)
            # append records (ensure all combos that appeared get a row; absent combos get INITIAL_RATING)
            for (approach, config), rating in dataset_level_scores[dataset].items():
                per_dataset_records.append({
                    'task': task,
                    'dataset': dataset,
                    'Approach': approach,
                    'Configuration': config,
                    'elo_score_dataset': float(rating)
                })
            # also ensure approaches present in dataset but without matches get initial rating row
            present_combos = {(row['Approach'], row['Configuration']) for _, row in dataset_df[['Approach', 'Configuration']].drop_duplicates().iterrows()}
            for approach, config in present_combos:
                if (approach, config) not in dataset_level_scores[dataset]:
                    per_dataset_records.append({
                        'task': task,
                        'dataset': dataset,
                        'Approach': approach,
                        'Configuration': config,
                        'elo_score_dataset': float(INITIAL_RATING)
                    })

        # compute per-task ELO by averaging dataset-level ELOs for each (Approach, Configuration)
        # collect all dataset ratings for this task
        task_dataset_df = pd.DataFrame(per_dataset_records)
        if task_dataset_df.empty:
            continue
        task_specific = task_dataset_df[task_dataset_df['task'] == task]
        if task_specific.empty:
            continue
        grouped = task_specific.groupby(['Approach', 'Configuration'])['elo_score_dataset'].mean().reset_index()
        grouped = grouped.rename(columns={'elo_score_dataset': 'elo_score_task'})
        grouped.insert(0, 'task', task)
        for _, row in grouped.iterrows():
            per_task_records.append({
                'task': row['task'],
                'Approach': row['Approach'],
                'Configuration': row['Configuration'],
                'elo_score_task': float(row['elo_score_task'])
            })

    # build DataFrames
    per_dataset_df = pd.DataFrame(per_dataset_records)
    per_task_df = pd.DataFrame(per_task_records)

    if per_task_df.empty:
        overall_df = pd.DataFrame()
    else:
        # overall: average elo_score_task across tasks per approach/config
        overall_df = per_task_df.groupby(['Approach', 'Configuration'])['elo_score_task'].mean().reset_index()
        overall_df = overall_df.rename(columns={'elo_score_task': 'elo_score_overall'}).sort_values(by='elo_score_overall', ascending=False).reset_index(drop=True)

    # sort and return
    if not per_dataset_df.empty:
        per_dataset_df = per_dataset_df.sort_values(['task', 'dataset', 'elo_score_dataset'], ascending=[True, True, False]).reset_index(drop=True)
    if not per_task_df.empty:
        per_task_df = per_task_df.sort_values(['task', 'elo_score_task'], ascending=[True, False]).reset_index(drop=True)

    print("%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Overview ELO Scores:")
    print(overall_df)

    return per_task_df, overall_df