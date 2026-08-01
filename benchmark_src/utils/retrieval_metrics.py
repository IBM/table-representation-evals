"""Scoring utilities for table-retrieval-style tasks (a query mapped to a ranked list of tables).

Shared between benchmark_src/tasks/run_table_retrieval_benchmark.py (approaches that plug into the
TableEmbeddingInterface, ranked via Qdrant ANN search) and standalone_comparisons/ scripts for
approaches that produce a ranked list directly through their own retrieval pipeline (e.g. Pneuma),
so both compute MRR/MAP/Recall/Precision identically.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RetrievalHit:
    database_id: str
    table_id: str
    score: float


def build_hits_info(
    hits: List[RetrievalHit],
    gold_tables_set: Set[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    hits_info = []

    for i, hit in enumerate(hits):
        retrieved_table_tuple = (hit.database_id, hit.table_id)
        is_match = retrieved_table_tuple in gold_tables_set

        hits_info.append({
            "database_id": hit.database_id,
            "table_id": hit.table_id,
            "score": hit.score,
            "is_match": is_match,
            "rank": i + 1
        })

    return hits_info


def compute_metrics_for_slice(
    hits_slice: List[Dict[str, Any]],
    num_gold_tables: int
) -> Dict[str, float]:
    """
    Calculates MRR, AP, Overlap, Recall, and Precision for a specific slice of hits (top-k).
    """
    query_reciprocal_rank = 0.0
    query_precision_at_k_sum = 0.0
    query_hits = 0

    for hit in hits_slice:
        if hit["is_match"]:
            # MRR: 1 / rank of first match
            if query_hits == 0:
                query_reciprocal_rank = 1.0 / hit["rank"]

            query_hits += 1
            precision_at_rank = query_hits / hit["rank"]
            query_precision_at_k_sum += precision_at_rank

    if num_gold_tables > 0:
        query_average_precision = query_precision_at_k_sum / num_gold_tables
    else:
        query_average_precision = 0.0

    overlap_count = sum(1 for h in hits_slice if h["is_match"])
    retrieved_count = len(hits_slice)

    recall = (overlap_count / num_gold_tables) if num_gold_tables > 0 else 0.0
    precision = (overlap_count / retrieved_count) if retrieved_count > 0 else 0.0

    return {
        "reciprocal_rank": query_reciprocal_rank,
        "average_precision": query_average_precision,
        "overlap": overlap_count,
        "retrieved_tables_count": retrieved_count,
        "recall": recall,
        "precision": precision
    }


def process_search_results(
    hits: List[RetrievalHit],
    gold_tables_set: Set[Tuple[str, str]],
    top_ks: List[int]
) -> Dict[str, Any]:
    """
    Calculates metrics for multiple k values based on a single (deepest) search result list.
    """
    hits_info = build_hits_info(hits, gold_tables_set)

    num_gold_tables = len(gold_tables_set)
    metrics_per_k = {}

    for k in top_ks:
        # Slice the hits info to the current k depth
        metrics_per_k[k] = compute_metrics_for_slice(hits_info[:k], num_gold_tables)

    return {
        "gold_tables_count": num_gold_tables,
        "metrics_per_k": metrics_per_k,
        "retrieved_items_full_list": hits_info
    }


def calculate_summary_metrics(query_results: List[Dict[str, Any]], top_ks: List[int]) -> Dict[str, Any]:
    total_queries = len(query_results)
    summary_metrics: Dict[str, Any] = {"total_queries": total_queries}

    for k in top_ks:
        total_reciprocal_rank = 0.0
        total_average_precision = 0.0
        total_overlap = 0
        total_gold_tables = 0
        total_retrieved_items = 0

        for res in query_results:
            m = res["metrics"]["metrics_per_k"][k]
            total_reciprocal_rank += m["reciprocal_rank"]
            total_average_precision += m["average_precision"]
            total_overlap += m["overlap"]
            total_gold_tables += res["metrics"]["gold_tables_count"]
            total_retrieved_items += m["retrieved_tables_count"]

        mrr = total_reciprocal_rank / total_queries
        map_score = total_average_precision / total_queries

        recall = (total_overlap / total_gold_tables) if total_gold_tables > 0 else 0.0
        precision = (total_overlap / total_retrieved_items) if total_retrieved_items > 0 else 0.0

        summary_metrics[f"k={k}"] = {
            "MRR": mrr,
            "MAP": map_score,
            "Recall": recall,
            "Precision": precision,
            "total_overlap": total_overlap,
        }

    logger.info(f"Evaluation summary: {summary_metrics}")
    return summary_metrics


def flatten_summary_metrics(summary_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens the nested summary metrics by appending @k to the metric keys.
    Example input:  "k=1": {"MRR": 0.5}
    Example output: "MRR@1": 0.5
    """
    flattened = {}
    for key, value in summary_metrics.items():
        # Check if the key represents a specific k-slice
        if isinstance(value, dict) and key.startswith("k="):
            try:
                k_suffix = key.split("=")[1]
                for metric_name, metric_val in value.items():
                    new_key = f"{metric_name}@{k_suffix}"
                    flattened[new_key] = metric_val
            except IndexError:
                raise ValueError(f"Invalid key format in summary metrics: {key}, expected 'k=<value>'")
        else:
            # Keep top-level scalar metrics (e.g., "total_queries") as is
            flattened[key] = value
    return flattened
