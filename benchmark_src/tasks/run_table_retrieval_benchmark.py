import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import multiprocessing
import logging
from hydra.utils import get_original_cwd
import json
import pickle
import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from benchmark_src.dataset_creation.target.collect_all_target_datasets import (
    get_target_dataset_by_name,
    DatasetBundle,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(cfg: DictConfig):
    logger.debug("Started run_table_retrieval_benchmark.main")
    logger.debug("Received cfg:")
    logger.debug(cfg)

    ds_name = cfg.dataset_name
    try:
        dataset_bundle = get_target_dataset_by_name(ds_name)
    except Exception as e:
        logger.error(f"Failed to load dataset '{ds_name}': {e}")
        return

    logger.info(
        f"Dataset '{ds_name}': Corpus has {len(dataset_bundle.corpus)} rows, "
        f"Queries has {len(dataset_bundle.queries)} rows."
    )

    # --- Simple dummy embedder (deterministic) ---
    def embed_item(item: dict, vector_size: int = 32) -> list:
        def _accumulate(val, acc):
            s = str(val)
            for i, ch in enumerate(s):
                acc[i % vector_size] = (acc[i % vector_size] + ord(ch)) % 1000

        acc = [0] * vector_size
        # Expecting keys: table (List[List[Any]]), database_id (str), table_id (str), context (dict)
        _accumulate(item.get("database_id", ""), acc)
        _accumulate(item.get("table_id", ""), acc)
        table = item.get("table", []) or []
        for row in table:
            if not isinstance(row, list):
                continue
            for cell in row:
                _accumulate(cell, acc)
        context = item.get("context", {}) or {}
        for k in sorted(context.keys()):
            _accumulate(k, acc)
            _accumulate(context[k], acc)
        return [x / 1000.0 for x in acc]

    # --- Prepare Qdrant persistent storage ---
    qdrant_path = Path(get_original_cwd()) / cfg.cache_dir / "qdrant_storage"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))

    collection_name = f"{ds_name}_tables"
    vector_size = 32

    client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )

    # --- Embed corpus and persist ---
    vectors = []
    payloads = []

    required_keys = ["table", "database_id", "table_id"]
    for row in dataset_bundle.corpus:
        # Validate required fields; if missing, log and skip
        # logger.info(f"typeof row: {type(row)}, keys: {list(row.keys())}")
        missing = [k for k in required_keys if k not in row or row.get(k) is None]
        if missing:
            logger.error(f"Row {row} missing required fields: {missing}. Skipping.")
            continue

        # Each row is expected to have the fields we discussed
        vector = embed_item(row, vector_size=vector_size)
        payload = {
            "database_id": row.get("database_id"),
            "table_id": row.get("table_id"),
        }

        vectors.append(vector)
        payloads.append(payload)

    client.upload_collection(
        collection_name=collection_name,
        vectors=np.array(vectors),
        payload=payloads,
    )

    logger.info(
        f"Embedded and persisted {len(vectors)} tables into Qdrant at {qdrant_path} collection '{collection_name}'"
    )