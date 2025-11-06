import logging
from pathlib import Path

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
    
    return client


def infer_embedder_vector_output_dim(
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
    
    vector_size = int(np.array(sample_vec).shape[-1])
    logger.info(f"Inferred vector dimension from the embedder: {vector_size}")
    return vector_size


def embed_corpus(
    table_component: TableEmbeddingInterface,
    corpus: Dataset
) -> tuple[list[np.ndarray], list[dict]]:
    vectors = []
    payloads = []
    required_keys = ["table", "database_id", "table_id"]
    
    corpus_subset = corpus.select(range(min(50, len(corpus))))
    
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

    table_component = get_embedder(cfg)
    
    try:
        vector_size = infer_embedder_vector_output_dim(table_component, dataset_bundle.corpus)
    except ValueError as e:
        logger.error(str(e))
        return

    collection_name = f"{cfg.task.task_name}:{cfg.dataset_name},{cfg.approach.approach_name}"
    client = setup_qdrant(cfg, collection_name, vector_size)

    vectors, payloads = embed_corpus(table_component, dataset_bundle.corpus)
    
    upload_corpus(client, collection_name, vectors, payloads)
    logger.info("Completed table_retrieval_benchmark successfully.")
