"""Bootstrap confidence intervals from per-instance scores in full_results.json.

Computes 95% bootstrap CIs for retrieval (MRR, MAP, Recall, Precision at k=1,3,5,10)
and shuffling (Accuracy, Silhouette, Contrastive). Writes one CSV with per-dataset
and __aggregate__ (pooled across datasets) rows.

Uses multiprocessing (16 workers) with batched bootstrap to stay within memory limits.

Run once before main.py:
    python prepare_paper_figures/bootstrap_cis.py
"""

import json
import logging
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_ROOT = Path("results_complete/results/table_paper_experiments")
OUTPUT_PATH = Path(__file__).parent / "bootstrap_cis.csv"
N_BOOTSTRAP = 10_000
BATCH_SIZE = 1000  # bootstrap iterations per batch — keeps memory low
RANDOM_SEED = 42
K_VALUES = [1, 3, 5, 10]
N_WORKERS = 16

RETRIEVAL_METRIC_MAP = {
    "MRR": "reciprocal_rank",
    "MAP": "average_precision",
    "Recall": "recall",
    "Precision": "precision",
}


def bootstrap_ci_batched(scores: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
                         batch_size: int = BATCH_SIZE,
                         ci: float = 95.0, random_seed: int = RANDOM_SEED) -> dict:
    """Batched bootstrap — processes n_bootstrap iterations in batches to limit memory.

    Instead of allocating (n_bootstrap × n) at once, allocates (batch_size × n)
    and accumulates mean + sum-of-squares for the bootstrap distribution.
    """
    rng = np.random.default_rng(random_seed)
    n = len(scores)
    n_batches = (n_bootstrap + batch_size - 1) // batch_size

    # We need percentiles, so we must store all means. But we batch the sampling.
    all_means = np.empty(n_bootstrap)
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_bootstrap)
        bs = end - start
        indices = rng.integers(0, n, size=(bs, n))
        all_means[start:end] = scores[indices].mean(axis=1)

    alpha = (100 - ci) / 2
    return {
        "mean": float(all_means.mean()),
        "se": float(all_means.std(ddof=0)),
        "ci_lower": float(np.percentile(all_means, alpha)),
        "ci_upper": float(np.percentile(all_means, 100 - alpha)),
        "n": n,
    }


def parse_path(file_path: str) -> dict:
    """Extract Approach, Configuration, Task, Dataset from a full_results.json path."""
    parts = Path(file_path).parts
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


def process_one_file(file_path: str) -> tuple[list[dict], dict]:
    """Read one full_results.json, bootstrap per-dataset CIs, return (rows, pool_arrays).

    pool_arrays maps pool_key -> np.ndarray of scores (instead of individual tuples).
    This avoids millions of tiny Python objects in the main process.
    """
    with open(file_path) as f:
        data = json.load(f)

    info = parse_path(file_path)
    task = info["Task"]
    rows = []
    pool_arrays: dict[tuple, np.ndarray] = {}

    if task == "table_retrieval":
        per_query = data.get("per_query_results", [])
        if not per_query:
            return rows, pool_arrays

        n_queries = len(per_query)

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

                valid = scores[~np.isnan(scores)]
                if len(valid) == 0:
                    continue

                metric_key = f"{metric_name}@{k}"
                ci = bootstrap_ci_batched(valid)
                rows.append({
                    **info,
                    "Metric": metric_key,
                    "Mean": ci["mean"],
                    "SE": ci["se"],
                    "CI_lower": ci["ci_lower"],
                    "CI_upper": ci["ci_upper"],
                    "N": ci["n"],
                })

                pool_key = (info["Approach"], info["Configuration"], task, metric_key)
                pool_arrays[pool_key] = valid

    elif task == "table_shuffling":
        triplets = data.get("triplets", [])
        if not triplets:
            return rows, pool_arrays

        acc_scores = np.array([1.0 if t["d_pos"] < t["d_neg"] else 0.0 for t in triplets])
        sil_scores = np.array([t["silhouette"] for t in triplets])
        con_scores = np.array([t["contrastive"] for t in triplets])

        for metric_name, scores in [("Accuracy", acc_scores),
                                     ("Silhouette", sil_scores),
                                     ("Contrastive", con_scores)]:
            ci = bootstrap_ci_batched(scores)
            rows.append({
                **info,
                "Metric": metric_name,
                "Mean": ci["mean"],
                "SE": ci["se"],
                "CI_lower": ci["ci_lower"],
                "CI_upper": ci["ci_upper"],
                "N": ci["n"],
            })

            pool_key = (info["Approach"], info["Configuration"], task, metric_name)
            pool_arrays[pool_key] = scores

    return rows, pool_arrays


def _bootstrap_one_aggregate(item: tuple) -> dict:
    """Bootstrap a single aggregate key (called from worker pool)."""
    (approach, config, task, metric), scores = item
    ci = bootstrap_ci_batched(np.array(scores))
    return {
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
    }


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S",
                        stream=sys.stderr)
    print("bootstrap_cis: starting up...", flush=True)

    if not RESULTS_ROOT.exists():
        logger.error(f"Results root not found: {RESULTS_ROOT.resolve()}")
        return

    files = sorted(RESULTS_ROOT.glob("**/full_results.json"))
    files = [str(f) for f in files]
    total = len(files)
    print(f"bootstrap_cis: found {total} full_results.json files", flush=True)
    logger.info(f"Found {total} full_results.json files under {RESULTS_ROOT}")

    retrieval_count = sum(1 for f in files if "/table_retrieval/" in f)
    shuffling_count = sum(1 for f in files if "/table_shuffling/" in f)
    logger.info(f"  Retrieval: {retrieval_count}  |  Shuffling: {shuffling_count}"
                f"  |  Other: {total - retrieval_count - shuffling_count}")

    # ---- Phase 1: process files in parallel ----
    t_start = time.monotonic()
    all_rows = []
    pooled: dict[tuple, list[np.ndarray]] = defaultdict(list)
    skipped = 0

    print(f"bootstrap_cis: processing {total} files with {N_WORKERS} workers"
          f" (batched bootstrap, batch={BATCH_SIZE})...", flush=True)

    with Pool(N_WORKERS) as pool:
        for i, (rows, pool_arrays) in enumerate(pool.imap_unordered(process_one_file, files)):
            all_rows.extend(rows)
            for key, arr in pool_arrays.items():
                pooled[key].append(arr)

            n_done = i + 1
            if n_done % 50 == 0 or n_done == total:
                elapsed = time.monotonic() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                logger.info(f"[{n_done}/{total}] {n_done / total * 100:.0f}%  "
                            f"rows={len(all_rows)}  "
                            f"elapsed={elapsed:.0f}s  rate={rate:.1f} files/s  "
                            f"ETA={eta:.0f}s")
                sys.stderr.flush()

    t_collect = time.monotonic() - t_start
    logger.info(f"File processing done in {t_collect:.0f}s — "
                f"{len(all_rows)} per-dataset rows, {len(pooled)} aggregate keys")

    # ---- Merge pooled arrays into single arrays per key ----
    logger.info("Merging pooled score arrays...")
    pooled_merged: dict[tuple, np.ndarray] = {}
    for key, arrays in pooled.items():
        pooled_merged[key] = np.concatenate(arrays)
    del pooled  # free memory
    logger.info(f"Merged {len(pooled_merged)} aggregate keys, "
                f"total scores: {sum(len(v) for v in pooled_merged.values()):,}")

    # ---- Phase 2: bootstrap aggregate rows in parallel ----
    logger.info(f"Bootstrapping {len(pooled_merged)} aggregate keys with {N_WORKERS} workers...")
    t_agg = time.monotonic()

    aggregate_rows = []
    with Pool(N_WORKERS) as pool:
        for i, row in enumerate(pool.imap_unordered(_bootstrap_one_aggregate,
                                                     pooled_merged.items())):
            aggregate_rows.append(row)
            n_done = i + 1
            if n_done % 50 == 0 or n_done == len(pooled_merged):
                logger.info(f"  aggregate bootstrapping: {n_done}/{len(pooled_merged)} keys")
                sys.stderr.flush()

    t_agg_elapsed = time.monotonic() - t_agg
    logger.info(f"Aggregate bootstrapping done in {t_agg_elapsed:.0f}s — {len(aggregate_rows)} rows")

    # ---- Write CSV ----
    df = pd.DataFrame(all_rows + aggregate_rows)
    df = df.sort_values(["Approach", "Configuration", "Task", "Dataset", "Metric"]).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

    total_time = time.monotonic() - t_start
    logger.info(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    logger.info(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}m)")


if __name__ == "__main__":
    main()
