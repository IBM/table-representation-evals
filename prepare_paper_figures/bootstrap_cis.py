"""Bootstrap confidence intervals from per-instance scores in full_results.json.

Computes 95% bootstrap CIs for retrieval (MRR, MAP, Recall, Precision at k=1,3,5,10)
and shuffling (Accuracy, Silhouette, Contrastive). Writes one CSV with per-dataset
and __aggregate__ (pooled across datasets) rows.

Run once before main.py:
    python prepare_paper_figures/bootstrap_cis.py
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_ROOT = Path("results_complete/results/table_paper_experiments")
OUTPUT_PATH = Path(__file__).parent / "bootstrap_cis.csv"
N_BOOTSTRAP = 10_000
RANDOM_SEED = 42
K_VALUES = [1, 3, 5, 10]

RETRIEVAL_METRIC_MAP = {
    "MRR": "reciprocal_rank",
    "MAP": "average_precision",
    "Recall": "recall",
    "Precision": "precision",
}


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
                 ci: float = 95.0, random_seed: int = RANDOM_SEED) -> dict:
    """Compute bootstrap confidence intervals for a mean.

    Args:
        scores: 1D array of per-instance values.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (percent).
        random_seed: Random seed for reproducibility.

    Returns:
        dict with keys: mean, se, ci_lower, ci_upper, n.
    """
    rng = np.random.default_rng(random_seed)
    n = len(scores)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        means[i] = sample.mean()
    alpha = (100 - ci) / 2
    return {
        "mean": float(means.mean()),
        "se": float(means.std(ddof=0)),
        "ci_lower": float(np.percentile(means, alpha)),
        "ci_upper": float(np.percentile(means, 100 - alpha)),
        "n": n,
    }


def parse_path(file_path: Path) -> dict:
    """Extract Approach, Configuration, Task, Dataset from a full_results.json path.

    Path structure:
        .../table_paper_experiments/{Approach}/{Config}/{task}/{dataset}/full_results.json
    """
    parts = file_path.parts
    # Find the index of 'table_paper_experiments' and take the 4 components after it
    try:
        idx = parts.index("table_paper_experiments")
    except ValueError:
        raise ValueError(f"Cannot find 'table_paper_experiments' in path: {file_path}")

    return {
        "Approach": parts[idx + 1],
        "Configuration": parts[idx + 2],
        "Task": parts[idx + 3],
        "Dataset": parts[idx + 4],
    }


def process_retrieval(file_path: Path) -> list[dict]:
    """Extract per-query scores from a retrieval full_results.json and bootstrap them.

    Returns a list of row dicts, one per (metric, k) combination.
    """
    with open(file_path) as f:
        data = json.load(f)

    per_query = data.get("per_query_results", [])
    if not per_query:
        logger.warning(f"No per_query_results in {file_path}")
        return []

    n_queries = len(per_query)
    info = parse_path(file_path)
    rows = []

    for k in K_VALUES:
        k_str = str(k)
        for metric_name, field in RETRIEVAL_METRIC_MAP.items():
            scores = np.empty(n_queries)
            missing = 0
            for i, q in enumerate(per_query):
                try:
                    scores[i] = q["metrics"]["metrics_per_k"][k_str][field]
                except (KeyError, TypeError):
                    scores[i] = np.nan
                    missing += 1

            if missing > 0:
                logger.warning(f"{file_path}: {missing}/{n_queries} missing {metric_name}@{k}")

            valid = scores[~np.isnan(scores)]
            if len(valid) == 0:
                continue

            ci = bootstrap_ci(valid)
            rows.append({
                **info,
                "Metric": f"{metric_name}@{k}",
                "Mean": ci["mean"],
                "SE": ci["se"],
                "CI_lower": ci["ci_lower"],
                "CI_upper": ci["ci_upper"],
                "N": ci["n"],
            })

    return rows


def process_shuffling(file_path: Path) -> list[dict]:
    """Extract per-triplet scores from a shuffling full_results.json and bootstrap them.

    Returns a list of row dicts for Accuracy, Silhouette, Contrastive.
    """
    with open(file_path) as f:
        data = json.load(f)

    triplets = data.get("triplets", [])
    if not triplets:
        logger.warning(f"No triplets in {file_path}")
        return []

    n = len(triplets)
    info = parse_path(file_path)

    # Accuracy: derived binary (d_pos < d_neg)
    acc_scores = np.array([
        1.0 if t["d_pos"] < t["d_neg"] else 0.0
        for t in triplets
    ])
    sil_scores = np.array([t["silhouette"] for t in triplets])
    con_scores = np.array([t["contrastive"] for t in triplets])

    rows = []
    for metric_name, scores in [("Accuracy", acc_scores),
                                 ("Silhouette", sil_scores),
                                 ("Contrastive", con_scores)]:
        ci = bootstrap_ci(scores)
        rows.append({
            **info,
            "Metric": metric_name,
            "Mean": ci["mean"],
            "SE": ci["se"],
            "CI_lower": ci["ci_lower"],
            "CI_upper": ci["ci_upper"],
            "N": ci["n"],
        })

    return rows


def collect_all_scores() -> tuple[list[dict], dict]:
    """Walk all full_results.json files and collect per-instance scores.

    Returns:
        all_rows: list of per-dataset bootstrap rows.
        pooled_scores: dict keyed by (Approach, Config, Task, Metric) -> list of scalars
                       for computing __aggregate__ CIs.
    """
    all_rows = []
    pooled_scores: dict[tuple, list[float]] = {}  # (Approach, Config, Task, Metric) -> scores

    files = sorted(RESULTS_ROOT.glob("**/full_results.json"))
    total = len(files)
    logger.info(f"Found {total} full_results.json files")

    for i, fp in enumerate(files):
        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{total} files...")

        try:
            info = parse_path(fp)
        except ValueError as e:
            logger.warning(f"Skipping {fp}: {e}")
            continue

        task = info["Task"]
        if task == "table_retrieval":
            rows = process_retrieval(fp)
        elif task == "table_shuffling":
            rows = process_shuffling(fp)
        else:
            continue  # TTD or other — no per-instance data

        all_rows.extend(rows)

        # Collect per-instance scores for __aggregate__ pooling
        # We need to re-read the file to get raw scores per instance
        with open(fp) as f:
            data = json.load(f)

        if task == "table_retrieval":
            per_query = data.get("per_query_results", [])
            if not per_query:
                continue
            n_queries = len(per_query)
            for k in K_VALUES:
                k_str = str(k)
                for metric_name, field in RETRIEVAL_METRIC_MAP.items():
                    key = (info["Approach"], info["Configuration"], task, f"{metric_name}@{k}")
                    if key not in pooled_scores:
                        pooled_scores[key] = []
                    for q in per_query:
                        try:
                            pooled_scores[key].append(
                                q["metrics"]["metrics_per_k"][k_str][field]
                            )
                        except (KeyError, TypeError):
                            pass

        elif task == "table_shuffling":
            triplets = data.get("triplets", [])
            if not triplets:
                continue
            acc = [1.0 if t["d_pos"] < t["d_neg"] else 0.0 for t in triplets]
            sil = [t["silhouette"] for t in triplets]
            con = [t["contrastive"] for t in triplets]
            for metric_name, scores in [("Accuracy", acc), ("Silhouette", sil),
                                         ("Contrastive", con)]:
                key = (info["Approach"], info["Configuration"], task, metric_name)
                if key not in pooled_scores:
                    pooled_scores[key] = []
                pooled_scores[key].extend(scores)

    logger.info(f"Processed {total}/{total} files. {len(all_rows)} per-dataset rows collected.")
    return all_rows, pooled_scores


def compute_aggregate_rows(pooled_scores: dict) -> list[dict]:
    """Bootstrap pooled per-instance scores across all datasets for __aggregate__ rows."""
    rows = []
    for (approach, config, task, metric), scores in pooled_scores.items():
        if not scores:
            continue
        ci = bootstrap_ci(np.array(scores))
        rows.append({
            "Approach": approach,
            "Configuration": config,
            "Task": task,
            "Dataset": "__aggregate__",
            "Metric": metric,
            "Mean": ci["mean"],
            "SE": ci["se"],
            "CI_lower": ci["ci_lower"],
            "CI_upper": ci["ci_upper"],
            "N": ci["n"],
        })
    logger.info(f"Computed {len(rows)} __aggregate__ rows")
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not RESULTS_ROOT.exists():
        logger.error(f"Results root not found: {RESULTS_ROOT}")
        return

    all_rows, pooled_scores = collect_all_scores()
    aggregate_rows = compute_aggregate_rows(pooled_scores)

    df = pd.DataFrame(all_rows + aggregate_rows)
    df = df.sort_values(["Approach", "Configuration", "Task", "Dataset", "Metric"]).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
