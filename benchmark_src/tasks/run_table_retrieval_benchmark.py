import logging
import json
from pathlib import Path
from typing import Set, List, Dict, Any, Tuple

import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from datasets import Dataset
from tqdm import tqdm

from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface
from benchmark_src.dataset_creation.target.collect_all_target_datasets import get_target_dataset_by_name
from benchmark_src.utils.framework import get_approach_class

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_embedder(cfg: DictConfig) -> TableEmbeddingInterface:
    approach_cls = get_approach_class(cfg)
    embedder = approach_cls(cfg)
    table_component: TableEmbeddingInterface = embedder._load_component(
        "table_embedding_component", "TableEmbeddingComponent", TableEmbeddingInterface
    )
    table_component.setup_model_for_task()
    return table_component


def setup_qdrant(cfg: DictConfig, collection_name: str, vector_size: int) -> QdrantClient:
    qdrant_path = Path(get_original_cwd()) / cfg.cache_dir / "qdrant_storage"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    
    logger.info(f"Initialized Qdrant client with persistent storage at {qdrant_path}")
    
    # Delete existing collection if it exists and create a new one
    client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )
    logger.info(f"Created new collection: {collection_name}")

    return client


def infer_embedder_output_dim(
    table_component: TableEmbeddingInterface, 
    corpus: Dataset
) -> int:
    """Infer the vector dimension by embedding a single sample from the corpus."""
    sample_vec = None
    for row in corpus:
        sample_vec = table_component.create_table_embedding(row["table"])
        break
    
    if sample_vec is None:
        raise ValueError("Corpus is empty; nothing to embed.")

    return int(np.array(sample_vec).shape[-1])


def embed_corpus(
    table_component: TableEmbeddingInterface,
    corpus: Dataset
) -> tuple[list[np.ndarray], list[dict]]:
    vectors = []
    payloads = []
    required_keys = ["table", "database_id", "table_id"]
    
    corpus_subset = corpus.select(range(min(50, len(corpus))))
    logger.warning(f"Embedding only {len(corpus_subset)} tables from the corpus.")

    for row in tqdm(corpus_subset, desc="Embedding tables"):
        missing = [k for k in required_keys if k not in row or row.get(k) is None]
        if missing:
            logger.error(f"Row missing required fields: {missing}. Skipping.")
            continue

        vec = table_component.create_table_embedding(row["table"])
        payload = { "database_id": row.get("database_id"), "table_id": row.get("table_id")}

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
    client.upload_collection(
        collection_name=collection_name,
        vectors=np.stack(vectors, axis=0),
        payload=payloads,
    )

    logger.info(
        f"Uploaded {len(vectors)} tables to Qdrant"
        f"collection '{collection_name}'"
    )

def evaluate_retrieval(
    client: QdrantClient,
    collection_name: str,
    table_component: TableEmbeddingInterface,
    queries_dataset: Dataset,
    top_k_values: List[int]
) -> Dict[str, Any]:
    per_query_results = []

    total_num_overlap = 0
    total_gold_tables = 0
    total_tables_capped = 0

    total_hits_at_k = {k: 0 for k in top_k_values}
    max_k = max(top_k_values)

    for query_row in tqdm(queries_dataset, desc="Evaluating queries"):
        query_text = query_row["query"]
        query_id = query_row["query_id"]
        gt_database_id = query_row["database_id"]
        gt_table_id_list = query_row["table_id"]

        if not isinstance(gt_table_id_list, list):
            gt_table_id_list = [gt_table_id_list]

        # Create the normalized set of ground truth (db_id, table_id) tuples
        gold_tables_set: Set[Tuple[str, str]] = set(
            (gt_database_id, t) for t in gt_table_id_list
        )

        query_vector = table_component.create_query_embedding(query_text)

        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=max_k,
            with_payload=True
        )

        retrieved_items_list = []
        retrieved_tables_set: Set[Tuple[str, str]] = set()
        hits_at_k = {k: 0 for k in top_k_values}
        found_match = False

        for i, hit in enumerate(search_results):
            payload = hit.payload
            retrieved_db_id = payload.get("database_id")
            retrieved_table_id = payload.get("table_id")

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
                if not found_match:
                    found_match = True

                for k in top_k_values:
                    if (i + 1) <= k and hits_at_k[k] == 0:
                        hits_at_k[k] = 1 # 1 indicates a hit within the top k

        overlap = len(gold_tables_set.intersection(retrieved_tables_set))
        total_num_overlap += overlap
        total_gold_tables += len(gold_tables_set)
        total_tables_capped += min(len(gold_tables_set), len(retrieved_tables_set))

        for k in top_k_values:
            total_hits_at_k[k] += hits_at_k[k]

        per_query_results.append({
            "query_id": query_id,
            "query": query_text,
            "ground_truth": {
                "database_id": gt_database_id,
                "table_ids": gt_table_id_list
            },
            "retrieved": retrieved_items_list,
            "metrics": {
                "overlap": overlap,
                "gold_tables_count": len(gold_tables_set),
                "retrieved_tables_count": len(retrieved_tables_set),
                "accuracy_at_k": hits_at_k
            }
        })


    # Calculate final aggregate metrics
    total_queries = len(queries_dataset)
    if total_queries == 0:
        logger.warning("No queries were processed.")
        return {"summary_metrics": {}, "per_query_results": []}

    recall = (total_num_overlap / total_gold_tables) if total_gold_tables > 0 else 0.0
    capped_recall = (total_num_overlap / total_tables_capped) if total_tables_capped > 0 else 0.0

    accuracy_at_k = {f"Accuracy@{k}": total_hits_at_k[k] / total_queries for k in top_k_values}

    summary_metrics = {
        "Recall": recall,
        "Capped_Recall": capped_recall,
        **accuracy_at_k,
        "total_queries": total_queries,
        "total_overlap": total_num_overlap,
        "total_gold_tables": total_gold_tables,
        "total_tables_capped": total_tables_capped
    }

    logger.info(f"Evaluation summary: {summary_metrics}")

    return {
        "summary_metrics": summary_metrics,
        "per_query_results": per_query_results
    }


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

    table_embedding_component = get_embedder(cfg)
    
    try:
        vector_size = infer_embedder_output_dim(table_embedding_component, dataset_bundle.corpus)
    except ValueError as e:
        logger.error(f"FATAL: {str(e)}. Could not infer vector dimension.")
        return
    logger.info(f"Inferred vector dimension from the embedder: {vector_size}")

    collection_name = f"{cfg.task.task_name}:{cfg.dataset_name},{cfg.approach.approach_name}"
    client = setup_qdrant(cfg, collection_name, vector_size)

    vectors, payloads = embed_corpus(table_embedding_component, dataset_bundle.corpus)
    
    upload_corpus(client, collection_name, vectors, payloads)


    logger.info("Corpus upload complete. Starting retrieval evaluation...")

    # top_k_values = cfg.task.get("top_k_values")
    top_k_values = [1, 3, 5, 10]

    evaluation_results = evaluate_retrieval(
        client=client,
        collection_name=collection_name,
        table_component=table_embedding_component,
        queries_dataset=dataset_bundle.queries,
        top_k_values=top_k_values
    )

    try:
        output_dir = Path(get_original_cwd()) / cfg.benchmark_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = collection_name.replace(":", "_").replace(",", "_") + "_results.json"
        output_path = output_dir / safe_filename

        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        logger.info(f"Successfully saved evaluation results to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save evaluation results {e}")

    logger.info("Completed table_retrieval_benchmark successfully.")
