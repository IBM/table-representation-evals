import logging
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_embedder(cfg: DictConfig) -> TableEmbeddingInterface:
    approach_cls = get_approach_class(cfg)
    embedder = approach_cls(cfg)
    table_component: TableEmbeddingInterface = embedder._load_component(
        "table_embedding_component", "TableEmbeddingComponent", TableEmbeddingInterface
    )
    table_component.setup_model_for_task()
    return table_component


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


def _process_search_results(
    search_results: List[rest.ScoredPoint],
    gold_tables_set: Set[Tuple[str, str]]
) -> Dict[str, Any]:
    retrieved_items_list = []
    retrieved_tables_set = set()

    query_reciprocal_rank = 0.0
    query_precision_at_k_sum = 0.0
    query_hits = 0
    num_gold_tables = len(gold_tables_set)

    for i, hit in enumerate(search_results):
        current_rank = i + 1
        retrieved_db_id = hit.payload.get("database_id")
        retrieved_table_id = hit.payload.get("table_id")

        retrieved_table_tuple = (retrieved_db_id, retrieved_table_id)
        retrieved_tables_set.add(retrieved_table_tuple)
        is_match = (retrieved_table_tuple in gold_tables_set)

        retrieved_items_list.append({
            "database_id": retrieved_db_id,
            "table_id": retrieved_table_id,
            "score": hit.score,
            "is_match": is_match
        })

        if is_match:
            # 1 / rank of first match
            if query_hits == 0:
                query_reciprocal_rank = 1.0 / current_rank
            query_hits += 1
            precision_at_k = query_hits / current_rank
            query_precision_at_k_sum += precision_at_k

    if num_gold_tables > 0:
        query_average_precision = query_precision_at_k_sum / num_gold_tables
    else:
        query_average_precision = 0.0

    overlap = len(gold_tables_set.intersection(retrieved_tables_set))

    return {
        "reciprocal_rank": query_reciprocal_rank,
        "average_precision": query_average_precision,
        "overlap": overlap,
        "retrieved_tables_count": len(search_results),
        "retrieved_items_list": retrieved_items_list
    }


def _calculate_summary_metrics(query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_queries = len(query_results)

    total_reciprocal_rank = sum(r['metrics']['reciprocal_rank'] for r in query_results)
    total_average_precision = sum(r['metrics']['average_precision'] for r in query_results)
    total_num_overlap = sum(r['metrics']['overlap'] for r in query_results)
    total_gold_tables = sum(r['metrics']['gold_tables_count'] for r in query_results)
    total_retrieved_items = sum(r['metrics']['retrieved_tables_count'] for r in query_results)

    mrr = total_reciprocal_rank / total_queries
    map_score = total_average_precision / total_queries

    # Micro-averaged Recall and Precision
    recall = (total_num_overlap / total_gold_tables) if total_gold_tables > 0 else 0.0
    precision = (total_num_overlap / total_retrieved_items) if total_retrieved_items > 0 else 0.0

    summary_metrics = {
        "MRR": mrr,
        "MAP": map_score,
        "Recall": recall,
        "Precision": precision,
        "total_queries": total_queries,
        "total_overlap": total_num_overlap,
        "total_gold_tables": total_gold_tables,
        "total_retrieved_items": total_retrieved_items
    }

    logger.info(f"Evaluation summary: {summary_metrics}")
    return summary_metrics


def _evaluate_retrieval(
    client: QdrantClient,
    collection_name: str,
    table_component: TableEmbeddingInterface,
    queries_dataset: Dataset,
    top_k: int
) -> Dict[str, Any]:
    """
    Runs the retrieval loop over all queries, computes MRR, MAP, Recall, and Precision.
    """
    if len(queries_dataset) == 0:
        logger.error("Queries dataset is empty. No evaluation will be performed.")
        return {"summary_metrics": {}, "per_query_results": []}

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

        search_results = _search_query(query_text, table_component, client, collection_name, top_k)

        query_metrics = _process_search_results(search_results, gold_tables_set)

        per_query_results.append({
            "query_id": query_id,
            "query": query_text,
            "ground_truth": {"database_id": gt_database_id, "table_ids": gt_table_id_list},
            "retrieved": query_metrics["retrieved_items_list"],
            "metrics": {
                "reciprocal_rank": query_metrics["reciprocal_rank"],
                "average_precision": query_metrics["average_precision"],
                "overlap": query_metrics["overlap"],
                "gold_tables_count": len(gold_tables_set),
                "retrieved_tables_count": query_metrics["retrieved_tables_count"]
            }
        })

    summary_metrics = _calculate_summary_metrics(per_query_results)

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
        logger.info(f"Deleted existing collection '{collection_name}' due to force_embed flag.")
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


def main(cfg: DictConfig):
    logger.debug("Started run_table_retrieval_benchmark.main")
    logger.debug("Received cfg:")
    logger.debug(cfg)

    try:
        dataset_bundle = get_target_dataset_by_name(cfg.dataset_name)
    except Exception as e:
        logger.error(f"Failed to load dataset '{cfg.dataset_name}': {e}")
        return

    logger.info(
        f"Dataset '{cfg.dataset_name}': Corpus has {len(dataset_bundle.corpus)} rows, "
        f"Queries has {len(dataset_bundle.queries)} rows."
    )

    client = get_qdrant_client(cfg)
    table_embedding_component = get_embedder(cfg)

    try:
        force_embed = bool(cfg.benchmark_tasks.table_retrieval.task_parameters.force_embed_corpus)
    except Exception:
        force_embed = False

    # Decide whether to embed using the simple rule you requested
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
        top_k=cfg.task.top_k
    )

    logger.info("Retrieval evaluation complete, saving results...")
    result_utils.save_results(cfg=cfg, metrics=evaluation_results)
    client.close()
