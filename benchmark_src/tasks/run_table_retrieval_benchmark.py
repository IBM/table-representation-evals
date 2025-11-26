import logging
import json
from pathlib import Path
from typing import Set, Dict, Any, Tuple, List

import datasets
import numpy as np
from datasets import Dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from tqdm import tqdm

from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface
from benchmark_src.dataset_creation.target.collect_all_target_datasets import get_target_dataset_by_name
from benchmark_src.utils import result_utils
from benchmark_src.utils.framework import get_approach_class
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk
from benchmark_src.tasks import component_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_embedder(cfg: DictConfig) -> Tuple[TableEmbeddingInterface, Dict[str, Any]]:
    approach_cls = get_approach_class(cfg)
    embedder = approach_cls(cfg)
    table_component: TableEmbeddingInterface = embedder._load_component(
        "table_embedding_component", "TableEmbeddingComponent", TableEmbeddingInterface
    )
    _, resource_metrics_setup = component_utils.run_model_setup(component=table_component)
    return table_component, resource_metrics_setup


def get_qdrant_client(cfg: DictConfig) -> QdrantClient:
    qdrant_path = Path(get_original_cwd()) / cfg.cache_dir / "qdrant_storage"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    logger.info(f"Initialized Qdrant client with persistent storage at {qdrant_path}")
    return client


def infer_embedder_output_dim(
        table_component: TableEmbeddingInterface,
        corpus: datasets.arrow_dataset.Dataset
) -> int:
    """Infer the vector dimension by embedding a single sample from the corpus."""
    if len(corpus) == 0:
        raise ValueError("Corpus is empty; cannot infer embedding dimension.")

    sample_embedding = table_component.create_table_embedding(corpus[0]["table"])
    return int(np.array(sample_embedding).shape[-1])


def embed_corpus(
    table_component: TableEmbeddingInterface,
    corpus: Dataset
) -> tuple[list[np.ndarray], list[dict]]:
    vectors = []
    payloads = []
    required_keys = ["table", "database_id", "table_id"]

    # TODO: For testing purposes, limiting to a subset of corpus
    corpus_subset = corpus.select(range(min(200, len(corpus))))
    logger.warning(f"Embedding only {len(corpus_subset)} tables from the corpus.")

    for row in tqdm(corpus_subset, desc="Embedding tables"):
        missing = [k for k in required_keys if k not in row or row.get(k) is None]
        if missing:
            logger.error(f"Row missing required fields: {missing}. Skipping.")
            continue

        vec = table_component.create_table_embedding(row["table"])
        payload = {"database_id": row.get("database_id"), "table_id": row.get("table_id")}

        vectors.append(np.array(vec))
        payloads.append(payload)

    logger.info(f"Embedded {len(vectors)} tables from corpus")
    return vectors, payloads


def upload_corpus(
    client: QdrantClient,
    collection_name: str,
    vectors: list[np.ndarray],
    payloads: list[dict]
):
    if not vectors:
        logger.warning("No vectors to upload. Skipping upload.")
        return

    client.upload_collection(
        collection_name=collection_name,
        vectors=np.stack(vectors, axis=0),
        payload=payloads,
    )

    logger.info(
        f"Uploaded {len(vectors)} tables to Qdrant collection '{collection_name}'"
    )


def _search_query(
    query_text: str,
    table_component: TableEmbeddingInterface,
    client: QdrantClient,
    collection_name: str,
    top_k: int
) -> List[rest.ScoredPoint]:
    query_vector = table_component.create_query_embedding(query_text)
    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )


def _build_hits_info(
    search_results: List[rest.ScoredPoint],
    gold_tables_set: Set[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    hits_info = []

    for i, hit in enumerate(search_results):
        retrieved_db_id = hit.payload.get("database_id")
        retrieved_table_id = hit.payload.get("table_id")
        retrieved_table_tuple = (retrieved_db_id, retrieved_table_id)

        is_match = retrieved_table_tuple in gold_tables_set

        hits_info.append({
            "database_id": retrieved_db_id,
            "table_id": retrieved_table_id,
            "score": hit.score,
            "is_match": is_match,
            "rank": i + 1
        })

    return hits_info


def _compute_metrics_for_slice(
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


def _process_search_results(
    search_results: List[rest.ScoredPoint],
    gold_tables_set: Set[Tuple[str, str]],
    top_ks: List[int]
) -> Dict[str, Any]:
    """
    Calculates metrics for multiple k values based on a single (deepest) search result list.
    """
    hits_info = _build_hits_info(search_results, gold_tables_set)

    num_gold_tables = len(gold_tables_set)
    metrics_per_k = {}

    for k in top_ks:
        # Slice the hits info to the current k depth
        metrics_per_k[k] = _compute_metrics_for_slice(hits_info[:k], num_gold_tables)

    return {
        "gold_tables_count": num_gold_tables,
        "metrics_per_k": metrics_per_k,
        "retrieved_items_full_list": hits_info
    }


def _calculate_summary_metrics(query_results: List[Dict[str, Any]], top_ks: List[int]) -> Dict[str, Any]:
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


def _evaluate_retrieval(
    client: QdrantClient,
    collection_name: str,
    table_component: TableEmbeddingInterface,
    queries_dataset: Dataset,
    top_ks: List[int]
) -> Dict[str, Any]:
    """
    Runs the retrieval loop over all queries, computes MRR, MAP, Recall, and Precision for all k in top_ks.
    """
    if len(queries_dataset) == 0:
        logger.error("Queries dataset is empty. No evaluation will be performed.")
        return {"summary_metrics": {}, "per_query_results": []}

    if top_ks is None or len(top_ks) == 0:
        logger.error("Top ks are empty. No evaluation will be performed.")
        return {"summary_metrics": {}, "per_query_results": []}

    # Determine the maximum k needed for the actual Qdrant query
    max_k = max(top_ks)

    per_query_results = []

    for query_row in tqdm(queries_dataset, desc="Evaluating queries"):
        query_text = query_row["query"]
        query_id = query_row["query_id"]
        gt_database_id = query_row["database_id"]
        gt_table_id_list = query_row["table_id"]

        if isinstance(gt_table_id_list, str):
            gt_table_id_list = [gt_table_id_list]

        # Create the normalized set of ground truth (db_id, table_id) tuples
        gold_tables_set: Set[Tuple[str, str]] = set(
            (gt_database_id, t) for t in gt_table_id_list
        )

        # Fetch the maximum required items
        search_results = _search_query(query_text, table_component, client, collection_name, max_k)

        processed_data = _process_search_results(search_results, gold_tables_set, top_ks)

        per_query_results.append({
            "query_id": query_id,
            "query": query_text,
            "ground_truth": {"database_id": gt_database_id, "table_ids": gt_table_id_list},
            "retrieved": processed_data["retrieved_items_full_list"],
            "metrics": {
                "gold_tables_count": processed_data["gold_tables_count"],
                "metrics_per_k": processed_data["metrics_per_k"]
            }
        })

    summary_metrics = _calculate_summary_metrics(per_query_results, top_ks)

    return {"summary_metrics": summary_metrics, "per_query_results": per_query_results}


def embeddings_exist(client: QdrantClient, collection_name: str) -> bool:
    """Return True if the collection exists and has a non-zero number of vectors/points."""
    try:
        exists = client.collection_exists(collection_name=collection_name)
    except Exception as e:
        logger.warning(f"Could not check collection existence: {e}")
        return False

    if not exists:
        return False

    try:
        count = client.count(collection_name=collection_name).count
    except Exception as e:
        logger.warning(f"Collection exists but count() failed: {e}")
        return False

    return count > 0


def _populate_vectordb(
    client: QdrantClient,
    collection_name: str,
    table_embedding_component: TableEmbeddingInterface,
    corpus_dataset: Dataset
) -> None:
    try:
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted existing collection '{collection_name}' due to force_embed flag or because it was empty.")
    except Exception:
        pass

    vector_size = infer_embedder_output_dim(table_embedding_component, corpus_dataset)

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )
        logger.info(f"Created collection '{collection_name}' with vector size {vector_size}.")
    except Exception as e:
        logger.warning(f"Failed to create collection '{collection_name}': {e}")

    vectors, payloads = embed_corpus(table_embedding_component, corpus_dataset)
    upload_corpus(client, collection_name, vectors, payloads)
    logger.info("Completed corpus embedding and upload to qdrant.")


@monitor_resources()
def run_table_retrieval(
    cfg: DictConfig,
    dataset_bundle,
    client: QdrantClient,
    table_embedding_component: TableEmbeddingInterface,
) -> Dict[str, Any]:
    try:
        force_embed = bool(cfg.benchmark_tasks.table_retrieval.task_parameters.force_embed_corpus)
    except Exception:
        force_embed = False

    # Embed corpus before evaluation if forced or not already embedded
    if force_embed or not embeddings_exist(client, cfg.run_identifier):
        _populate_vectordb(
            client=client,
            collection_name=cfg.run_identifier,
            table_embedding_component=table_embedding_component,
            corpus_dataset=dataset_bundle.corpus,
        )
    else:
        logger.info(
            f"Skipping corpus embedding as vectorDB collection '{cfg.run_identifier}' is already populated."
            f" Set force_embed_corpus to True to re-embed."
        )

    logger.info("Starting retrieval evaluation...")

    evaluation_results = _evaluate_retrieval(
        client=client,
        collection_name=cfg.run_identifier,
        table_component=table_embedding_component,
        queries_dataset=dataset_bundle.queries,
        top_ks=list(cfg.task.top_ks)
    )

    logger.info("Retrieval evaluation complete, saving results...")
    return evaluation_results


def _flatten_summary_metrics(summary_metrics: Dict[str, Any]) -> Dict[str, Any]:
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


def _save_full_results_to_disk(
        results: Dict[str, Any],
        filename: str = "full_results.json"
) -> None:
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved full detailed results to '{filename}'")
    except Exception as e:
        logger.error(f"Failed to save full results to disk: {e}")


def main(cfg: DictConfig):
    logger.debug("Started run_table_retrieval_benchmark.main")
    logger.debug("Received cfg:")
    logger.debug(cfg)

    try:
        dataset_bundle = get_target_dataset_by_name(cfg.dataset_name)
    except Exception as e:
        logger.error(f"Failed to load dataset '{cfg.dataset_name}': {e}")
        raise e

    logger.info(
        f"Dataset '{cfg.dataset_name}': Corpus has {len(dataset_bundle.corpus)} rows, "
        f"Queries has {len(dataset_bundle.queries)} rows."
    )

    client = get_qdrant_client(cfg)
    table_embedding_component, resource_metrics_setup = get_embedder(cfg)

    evaluation_results, resource_metrics_task = run_table_retrieval(
        cfg=cfg,
        dataset_bundle=dataset_bundle,
        client=client,
        table_embedding_component=table_embedding_component,
    )

    if resource_metrics_setup:
        save_resource_metrics_to_disk(
            cfg=cfg,
            resource_metrics_setup=resource_metrics_setup,
            resource_metrics_task=resource_metrics_task,
        )

    # Dump the full, raw results (including per_query_results) to disk
    _save_full_results_to_disk(evaluation_results, filename="full_results.json")

    flattened_summary_metrics = _flatten_summary_metrics(evaluation_results["summary_metrics"])

    result_utils.save_results(cfg=cfg, metrics=flattened_summary_metrics)

    client.close()
