"""
NL-to-Column Mapping Benchmark for Text2SQL (Concept Mapping)

This benchmark evaluates how well column embeddings can identify relevant columns
for SQL query generation from natural language questions. It uses the BIRD benchmark
dataset with pure_concept_mapping_queries.json.

Task: Given a natural language query, use column embeddings to retrieve the database
columns that are relevant for answering the query (concept-based, no cell values).

Uses Qdrant for efficient ANN search over column embeddings.
"""

from omegaconf import DictConfig, OmegaConf
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import logging
import multiprocessing
import numpy as np
import sqlite3
from typing import List, Dict, Tuple
import statistics

import gc
import shutil
import time
import torch

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from benchmark_src.tasks import component_utils
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface
from benchmark_src.utils import framework, result_utils, load_benchmark
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk

logger = logging.getLogger(__name__)


def get_qdrant_client(cfg: DictConfig) -> Tuple[QdrantClient, Path]:
    """
    Initialize Qdrant client with persistent storage.
    
    Cache key is based on approach, embedding model, and dataset to enable reuse across experiments.
    """
    # Create a stable cache identifier based on approach, model, and dataset
    approach_name = cfg.approach.approach_name
    embedding_model = cfg.approach.get("embedding_model", "default")
    dataset_name = cfg.dataset_name
    
    # Sanitize names for filesystem (replace / and : with _)
    embedding_model_safe = embedding_model.replace("/", "_").replace(":", "_")
    cache_key = f"{approach_name}_{embedding_model_safe}_{dataset_name}"
    
    qdrant_path = Path(cfg.cache_dir) / "qdrant_storage" / f"qdrant_nl2column_{cache_key}"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    logger.info(f"Initialized Qdrant client with persistent storage at {qdrant_path}")
    logger.info(f"Cache key: {cache_key} (enables reuse across experiments)")
    return client, qdrant_path


def cleanup_stale_collection_dirs(qdrant_path: Path) -> None:
    """Best-effort removal of '.stale-*' collection dirs left behind by a previous run.

    These are collection directories renamed aside (see run_nl2column_mapping_benchmark) instead
    of deleted in place, since cache/qdrant_storage lives on NFS where Qdrant's local segment
    files can stay busy well past delete_collection() returning. By the time a later run calls
    this, enough time has passed that the files are no longer busy.
    """
    collections_path = qdrant_path / "collection"
    if not collections_path.exists():
        return
    for stale_dir in collections_path.glob("*.stale-*"):
        try:
            shutil.rmtree(stale_dir)
            logger.info(f"Cleaned up stale collection directory {stale_dir.name}")
        except OSError as e:
            logger.warning(f"Could not clean up stale collection directory {stale_dir.name}: {e}")


def load_benchmark_data(cfg: DictConfig) -> Tuple[Path, List[Dict]]:
    """Load the BIRD benchmark data for NL-to-column mapping."""
    bird_path_override = cfg.dataset.get("bird_path", None)
    bird_path = Path(bird_path_override) if bird_path_override else Path(cfg.cache_dir) / "datasets" / "bird"

    benchmark_file = cfg.dataset.get("benchmark_file", "pure_concept_mapping_queries.json")
    queries_file = bird_path / benchmark_file
    assert queries_file.exists(), f"Could not find queries file at {queries_file}"

    with open(queries_file, "r") as f:
        queries = json.load(f)

    databases_path = bird_path / "train" / "train_databases"
    assert databases_path.exists(), f"Could not find databases at {databases_path}. Please unzip train_databases.zip"
    
    logger.info(f"Loaded {len(queries)} queries from {queries_file}")
    logger.info(f"Using benchmark file: {benchmark_file}")
    logger.info(f"Database path: {databases_path}")
    
    return databases_path, queries


def load_database_schema(db_path: Path) -> Dict[str, List[str]]:
    """Load database schema (table names and column names)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row[0] for row in cursor.fetchall()]
    
    # Filter out SQLite system tables
    tables = [t for t in all_tables if not t.startswith('sqlite_')]
    
    schema = {}
    for table in tables:
        cursor.execute(f'PRAGMA table_info("{table}");')
        columns = [row[1] for row in cursor.fetchall()]
        schema[table] = columns
    
    conn.close()
    return schema


def load_table_data(db_path: Path, table: str, limit: int = 1000) -> pd.DataFrame:
    """Load data from a database table."""
    conn = sqlite3.connect(db_path)
    try:
        query = f'SELECT * FROM "{table}" LIMIT {limit}'
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        logger.warning(f"Error loading table {table}: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    
    return df


def embed_query_text(
    column_embedding_component: ColumnEmbeddingInterface,
    query_text: str
) -> np.ndarray:
    """
    Embed a natural language query using the column embedding component so the
    query vector lives in the same space as the stored column embeddings.
    """
    query_df = pd.DataFrame({"query": [query_text]})
    column_embeddings, _ = column_embedding_component.create_column_embeddings_for_table(query_df)

    if column_embeddings is None:
        raise ValueError("Failed to create query embedding")

    if hasattr(column_embeddings, 'cpu'):
        column_embeddings = column_embeddings.cpu().numpy()

    # The single column "query" is at index 0
    return column_embeddings[0]


def create_qdrant_collection_for_database(
    client: QdrantClient,
    collection_name: str,
    column_embedding_component: ColumnEmbeddingInterface,
    db_path: Path,
    schema: Dict[str, List[str]],
    max_rows_per_table: int = 1000
) -> int:
    """
    Create Qdrant collection with all column embeddings from the database.
    
    Returns:
        Number of columns embedded
    """
    # Get embedding dimension from a sample
    first_table = next(iter(schema.keys()))
    sample_df = load_table_data(db_path, first_table, limit=max_rows_per_table)
    if sample_df.empty:
        raise ValueError(f"Could not load sample data from {first_table}")
    
    sample_embeddings, _ = column_embedding_component.create_column_embeddings_for_table(sample_df)
    if hasattr(sample_embeddings, 'cpu'):
        sample_embeddings = sample_embeddings.cpu().numpy()
    embedding_dim = sample_embeddings[0].size
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )
    logger.info(f"Created Qdrant collection {collection_name} with vector size {embedding_dim}")
    
    # Embed and upload all columns
    points = []
    current_id = 0
    total_columns = 0
    
    for table in tqdm(schema.keys(), desc=f"Embedding tables for {collection_name}"):
        df = load_table_data(db_path, table, limit=max_rows_per_table)
        
        if df.empty or len(df) == 0:
            continue
        
        try:
            # Create column embeddings for the table
            column_embeddings, column_names = column_embedding_component.create_column_embeddings_for_table(df)
            
            if column_embeddings is None:
                logger.warning(f"No embeddings returned for table {table}")
                continue
            
            # Convert to numpy if needed
            if hasattr(column_embeddings, 'cpu'):
                column_embeddings = column_embeddings.cpu().numpy()
            elif isinstance(column_embeddings, list):
                column_embeddings = np.array([emb.cpu().numpy() if hasattr(emb, 'cpu') else emb for emb in column_embeddings])
            
            # Check for NaNs
            if np.isnan(column_embeddings).any():
                column_embeddings = np.nan_to_num(column_embeddings, nan=0.0)
                logger.warning(f"Found NaNs in embeddings for table {table}, converted to 0s")
            
            # Store each column embedding in Qdrant with metadata
            for col_idx, col_name in enumerate(column_names):
                points.append(PointStruct(
                    id=current_id,
                    vector=column_embeddings[col_idx].tolist(),
                    payload={
                        "table": table,
                        "column": col_name,
                        "table_column": f"{table}.{col_name}"
                    }
                ))
                current_id += 1
                total_columns += 1
                
                # Upload in batches
                if len(points) >= 500:
                    client.upsert(collection_name=collection_name, points=points)
                    points = []
            
            del df, column_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error embedding table {table}: {e}")
            continue
    
    # Upload remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    logger.info(f"Embedded {total_columns} columns into collection {collection_name}")
    return total_columns


@monitor_resources()
def run_nl2column_mapping_benchmark(
    cfg: DictConfig,
    column_embedding_component: ColumnEmbeddingInterface,
    databases_path: Path,
    queries: List[Dict]
) -> List[Dict]:
    """Run the NL-to-column mapping benchmark using Qdrant for ANN search."""
    
    # Setup Qdrant client
    client, qdrant_path = get_qdrant_client(cfg)
    cleanup_stale_collection_dirs(qdrant_path)

    results = []
    db_collections = {}  # Track which collections have been created

    # Limit number of queries if specified
    max_queries = cfg.task.get("max_queries", len(queries))
    queries = queries[:max_queries]

    force_embed_corpus = cfg.task.get("force_embed_corpus", False)

    for query_idx, query in enumerate(tqdm(queries, desc="Processing queries")):
        db_id = query["db_id"]
        question = query["question"]
        gold_columns = query["gold_columns"]
        
        if not gold_columns:
            continue
        
        # Setup database collection if not already done
        if db_id not in db_collections:
            db_path = databases_path / db_id / f"{db_id}.sqlite"
            if not db_path.exists():
                logger.warning(f"Database not found: {db_path}")
                continue
            
            collection_name = f"bird_columns_{db_id}"
            collection_dir = qdrant_path / "collection" / collection_name
            completed_file_path = collection_dir / "COMPLETED"

            # Check if collection already exists and is complete
            if client.collection_exists(collection_name=collection_name) and completed_file_path.exists() and not force_embed_corpus:
                logger.info(f"Collection {collection_name} already exists and is complete")
            else:
                # Delete if exists but incomplete, or if a re-embed was forced
                if client.collection_exists(collection_name=collection_name):
                    client.delete_collection(collection_name=collection_name)
                    logger.info(
                        f"Deleted {'existing' if force_embed_corpus else 'incomplete'} collection {collection_name}"
                    )
                # Move aside any leftover collection directory (e.g. a stale COMPLETED marker)
                # rather than relying on delete_collection to have fully cleaned it up. Renaming
                # is metadata-only and doesn't wait on open file handles, unlike shutil.rmtree,
                # which can hit EBUSY for a long time on NFS if Qdrant's background segment
                # cleanup hasn't released a file handle yet. The stale directory is reaped later
                # by cleanup_stale_collection_dirs(), once enough time has passed that it's safe.
                if collection_dir.exists():
                    stale_dir = collection_dir.with_name(f"{collection_dir.name}.stale-{time.time_ns()}")
                    collection_dir.rename(stale_dir)
                    logger.info(f"Moved leftover collection directory aside to {stale_dir.name}")

                # Create new collection
                logger.info(f"Creating column embeddings for database {db_id}")
                schema = load_database_schema(db_path)
                
                num_columns = create_qdrant_collection_for_database(
                    client=client,
                    collection_name=collection_name,
                    column_embedding_component=column_embedding_component,
                    db_path=db_path,
                    schema=schema,
                    max_rows_per_table=cfg.task.get("max_rows_per_table", 1000)
                )
                
                # Mark as complete
                completed_file_path.parent.mkdir(parents=True, exist_ok=True)
                completed_file_path.write_text(f"Embedded {num_columns} columns")
                logger.info(f"Marked collection {collection_name} as complete")
            
            db_collections[db_id] = collection_name
        
        collection_name = db_collections[db_id]
        
        # Prepare ground truth
        gold_column_set = {f"{gc['table']}.{gc['column']}" for gc in gold_columns}
        
        # Embed the natural language query
        try:
            query_embedding = embed_query_text(column_embedding_component, question)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            continue
        
        # Search Qdrant for top-K nearest columns
        # Use a large K to ensure we get all relevant columns
        top_k = cfg.task.get("top_k_columns", 50)
        
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        
        # Get ranked columns
        ranked_columns = [hit.payload["table_column"] for hit in hits]
        
        # Compute metrics at different K values
        k_values = [1, 3, 5, 10, 20]
        metrics = {}
        
        for k in k_values:
            top_k_cols = ranked_columns[:k]
            correct = sum(1 for col in top_k_cols if col in gold_column_set)
            
            precision = correct / k if k > 0 else 0.0
            recall = correct / len(gold_column_set) if gold_column_set else 0.0
            
            metrics[f"precision@{k}"] = precision
            metrics[f"recall@{k}"] = recall
        
        # Compute MRR
        mrr = 0.0
        for rank, col in enumerate(ranked_columns, 1):
            if col in gold_column_set:
                mrr = 1.0 / rank
                break
        
        # Compute MAP
        average_precision = 0.0
        num_relevant = 0
        for rank, col in enumerate(ranked_columns, 1):
            if col in gold_column_set:
                num_relevant += 1
                precision_at_rank = num_relevant / rank
                average_precision += precision_at_rank
        
        if len(gold_column_set) > 0:
            average_precision /= len(gold_column_set)
        
        metrics["mrr"] = mrr
        metrics["map"] = average_precision
        
        results.append({
            "query_idx": query_idx,
            "db_id": db_id,
            "question": question,
            "num_gold_columns": len(gold_column_set),
            "num_retrieved_columns": len(ranked_columns),
            **metrics
        })
    
    return results


def compute_aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute aggregate metrics across all queries."""
    if not results:
        return {}
    
    aggregate = {}
    
    metric_keys = [k for k in results[0].keys() 
                   if k not in ["query_idx", "db_id", "question", 
                                "num_gold_columns", "num_retrieved_columns"]]
    
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            aggregate[f"mean_{key}"] = statistics.mean(values)
            if len(values) > 1:
                aggregate[f"std_{key}"] = statistics.stdev(values)
    
    return aggregate


def main(cfg: DictConfig):
    """Main entry point for the NL-to-column mapping benchmark."""
    logger.info("Started run_NL2column_mapping_benchmark")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True)
    
    # Load dataset config
    dataset_config_path = Path(cfg.project_root) / "configs" / "dataset" / f"{cfg.dataset_name}.yaml"
    dataset_cfg = OmegaConf.load(str(dataset_config_path))
    OmegaConf.set_struct(cfg, False)
    cfg.dataset = dataset_cfg.dataset
    OmegaConf.set_struct(cfg, True)
    
    # Load benchmark data
    databases_path, queries = load_benchmark_data(cfg)
    
    # Instantiate the embedding approach class
    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)
    
    # Load column embedding component
    column_embedding_component = embedder._load_component(
        "column_embedding_component",
        "ColumnEmbeddingComponent",
        ColumnEmbeddingInterface
    )

    # Setup model - column embedding needs a sample table
    sample_df = None
    if queries:
        first_db_id = queries[0]["db_id"]
        first_db_path = databases_path / first_db_id / f"{first_db_id}.sqlite"
        if first_db_path.exists():
            schema = load_database_schema(first_db_path)
            if schema:
                first_table = next(iter(schema.keys()))
                sample_df = load_table_data(first_db_path, first_table, limit=100)

    if sample_df is None or sample_df.empty:
        sample_df = pd.DataFrame({"col1": ["sample"], "col2": [1]})
        logger.warning("Could not load sample table from database, using dummy table for setup")

    _, resource_metrics_setup = component_utils.run_model_setup(
        component=column_embedding_component,
        input_table=sample_df,
        dataset_information=None
    )
    
    # Run benchmark
    all_results, resource_metrics_task = run_nl2column_mapping_benchmark(
        cfg=cfg,
        column_embedding_component=column_embedding_component,
        databases_path=databases_path,
        queries=queries
    )
    
    # Save resource metrics
    save_resource_metrics_to_disk(
        cfg=cfg,
        resource_metrics_setup=resource_metrics_setup,
        resource_metrics_task=resource_metrics_task
    )
    
    # Compute aggregate metrics
    aggregate_metrics = compute_aggregate_metrics(all_results)
    
    # Log summary
    logger.info(f"Processed {len(all_results)} queries")
    logger.info("Aggregate Metrics:")
    for key, value in aggregate_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save results
    result_utils.save_results(cfg=cfg, metrics=aggregate_metrics)
    
    # Save detailed per-query results
    with open(Path(cfg.output_dir) / "results_per_query.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("Benchmark complete!")

# Made with Bob
