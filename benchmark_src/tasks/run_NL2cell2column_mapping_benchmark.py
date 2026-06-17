"""
Cell-to-Column Mapping Benchmark for Text2SQL

This benchmark evaluates how well cell embeddings can identify relevant columns
for SQL query generation from natural language questions. It uses the BIRD benchmark
dataset with cell_value_matching_queries.json or fuzzy_cell_matching.json.

Task: Given a natural language query, use cell embeddings to retrieve the database
columns that are relevant for answering the query.

Two modes supported:
1. Full NL Query: Embed the entire natural language question
2. Extracted Values: Embed extracted values from the query (when available)

Key Features:
- Uses SQL DISTINCT to get unique values per column before sampling
- ALWAYS includes matched values from benchmark queries in embeddings
- Supports both exact matching (cell_value_matching_queries.json) and
  fuzzy/semantic matching (fuzzy_cell_matching.json) benchmarks
- Uses Qdrant for efficient ANN search over cell embeddings
"""

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import logging
import multiprocessing
import numpy as np
import sqlite3
from typing import List, Dict, Tuple, Optional
import statistics
from collections import defaultdict, Counter
import gc
import torch

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from benchmark_src.tasks import component_utils
from benchmark_src.approach_interfaces.cell_embedding_interface import CellEmbeddingInterface
from benchmark_src.utils import framework, result_utils
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk

logger = logging.getLogger(__name__)


def get_qdrant_client(cfg: DictConfig) -> Tuple[QdrantClient, Path]:
    """Initialize Qdrant client with persistent storage."""
    qdrant_path = Path(get_original_cwd()) / cfg.cache_dir / "qdrant_storage" / f"qdrant_cell_to_column_{cfg.run_identifier}"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    logger.info(f"Initialized Qdrant client with persistent storage at {qdrant_path}")
    return client, qdrant_path


def load_benchmark_data(cfg: DictConfig) -> Tuple[Path, List[Dict]]:
    """Load the BIRD benchmark data for cell-to-column mapping."""
    bird_path = Path(cfg.dataset.bird_path).expanduser()
    
    # Get benchmark file from config, default to cell_value_matching_queries.json
    benchmark_file = cfg.dataset.get("nl2cell2column_benchmark_file", "cell_value_matching_queries.json")
    queries_file = bird_path / benchmark_file
    assert queries_file.exists(), f"Could not find queries file at {queries_file}"
    
    with open(queries_file, "r") as f:
        queries = json.load(f)
    
    databases_path = bird_path / "bird" / "train" / "train_databases"
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
    tables = [row[0] for row in cursor.fetchall()]
    
    schema = {}
    for table in tables:
        # Quote table name to handle spaces and special characters
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


def get_unique_column_values(db_path: Path, table: str, column: str, limit: int = 1000) -> List[str]:
    """
    Get unique values from a specific column using SQL.
    
    Args:
        db_path: Path to the SQLite database
        table: Table name
        column: Column name
        limit: Maximum number of unique values to retrieve
        
    Returns:
        List of unique string values from the column
    """
    conn = sqlite3.connect(db_path)
    try:
        # Use DISTINCT to get unique values, filter out NULLs
        query = f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit}'
        cursor = conn.cursor()
        cursor.execute(query)
        values = [str(row[0]) for row in cursor.fetchall()]
    except Exception as e:
        logger.warning(f"Error getting unique values from {table}.{column}: {e}")
        values = []
    finally:
        conn.close()
    
    return values


def collect_matched_values_for_database(queries: List[Dict], db_id: str) -> Dict[str, set]:
    """
    Collect all matched values from queries for a specific database.
    Groups values by table.column for efficient lookup.
    
    Args:
        queries: List of query dictionaries
        db_id: Database ID to filter queries
        
    Returns:
        Dictionary mapping "table.column" to set of matched values
    """
    matched_values_by_column = defaultdict(set)
    
    for query in queries:
        if query.get("db_id") != db_id:
            continue
            
        # Get matched values and gold columns from the query
        # matched_values is a list of strings: ["83373278", "1"]
        # gold_columns is a list of dicts: [{"table": "lists", "column": "user_id"}, ...]
        matched_values = query.get("matched_values", [])
        gold_columns = query.get("gold_columns", [])
        
        # Match values with their corresponding columns
        # Assumes matched_values and gold_columns are aligned by index
        for i, value in enumerate(matched_values):
            if i < len(gold_columns):
                col_info = gold_columns[i]
                table = col_info.get("table")
                column = col_info.get("column")
                
                if table and column and value is not None:
                    table_column = f"{table}.{column}"
                    matched_values_by_column[table_column].add(str(value))
    
    return matched_values_by_column


def embed_query_text(
    cell_embedding_component: CellEmbeddingInterface,
    query_text: str
) -> np.ndarray:
    """Embed a natural language query as if it were a cell value."""
    query_df = pd.DataFrame({"query": [query_text]})
    cell_embeddings = cell_embedding_component.create_cell_embeddings_for_table(query_df)
    
    if cell_embeddings is None:
        raise ValueError("Failed to create query embedding")
    
    # Return the data row embedding (skip header at index 0)
    return cell_embeddings[1, 0, :]


def embed_extracted_values(
    cell_embedding_component: CellEmbeddingInterface,
    extracted_values: List[str]
) -> np.ndarray:
    """Embed extracted values from a query and aggregate them."""
    if not extracted_values:
        raise ValueError("No extracted values provided")
    
    values_df = pd.DataFrame({"values": extracted_values})
    cell_embeddings = cell_embedding_component.create_cell_embeddings_for_table(values_df)
    
    if cell_embeddings is None:
        raise ValueError("Failed to create embeddings for extracted values")
    
    # Skip header row and aggregate data rows
    value_embeddings = cell_embeddings[1:, 0, :]
    return np.mean(value_embeddings, axis=0)


def create_qdrant_collection_for_database(
    client: QdrantClient,
    collection_name: str,
    cell_embedding_component: CellEmbeddingInterface,
    db_path: Path,
    schema: Dict[str, List[str]],
    matched_values_by_column: Dict[str, set],
    max_unique_values_per_column: int = 1000
) -> int:
    """
    Create Qdrant collection with cell embeddings from unique values in the database.
    Always includes matched values from the benchmark queries.
    
    Args:
        client: Qdrant client
        collection_name: Name for the collection
        cell_embedding_component: Component for creating embeddings
        db_path: Path to SQLite database
        schema: Database schema (table -> columns)
        matched_values_by_column: Dictionary of matched values from queries by "table.column"
        max_unique_values_per_column: Maximum unique values to sample per column
    
    Returns:
        Number of cells embedded
    """
    # Get embedding dimension from a sample
    first_table = next(iter(schema.keys()))
    sample_df = load_table_data(db_path, first_table, limit=1)
    if sample_df.empty:
        raise ValueError(f"Could not load sample data from {first_table}")
    
    sample_embeddings = cell_embedding_component.create_cell_embeddings_for_table(sample_df)
    embedding_dim = sample_embeddings.shape[2]
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )
    logger.info(f"Created Qdrant collection {collection_name} with vector size {embedding_dim}")
    
    # Embed and upload all cells
    points = []
    current_id = 0
    total_cells = 0
    
    for table, columns in tqdm(schema.items(), desc=f"Embedding tables for {collection_name}"):
        for col_name in columns:
            table_column = f"{table}.{col_name}"
            
            try:
                # Get unique values from the database using SQL
                unique_values = get_unique_column_values(
                    db_path, table, col_name, limit=max_unique_values_per_column
                )
                
                if not unique_values:
                    continue
                
                # Get matched values for this column from the benchmark
                matched_values = matched_values_by_column.get(table_column, set())
                
                # Combine unique values with matched values, ensuring matched values are included
                all_values = set(unique_values)
                all_values.update(matched_values)
                
                # Convert to list for embedding
                values_to_embed = list(all_values)
                
                if not values_to_embed:
                    continue
                
                # Create a DataFrame with these values
                values_df = pd.DataFrame({col_name: values_to_embed})
                
                # Create cell embeddings for this column
                # Shape: [rows+1, 1, embedding_dim]
                cell_embeddings = cell_embedding_component.create_cell_embeddings_for_table(values_df)
                
                if cell_embeddings is None:
                    logger.warning(f"No embeddings returned for {table_column}")
                    continue
                
                # Store each cell embedding in Qdrant with metadata
                for row_idx, value in enumerate(values_to_embed):
                    # Get cell embedding (skip header row at index 0)
                    cell_emb = cell_embeddings[row_idx + 1, 0, :]
                    
                    # Mark if this value was from the matched values
                    is_matched = str(value) in matched_values
                    
                    # Create point with metadata
                    points.append(PointStruct(
                        id=current_id,
                        vector=cell_emb.tolist(),
                        payload={
                            "table": table,
                            "column": col_name,
                            "table_column": table_column,
                            "value": str(value),
                            "is_matched_value": is_matched
                        }
                    ))
                    current_id += 1
                    total_cells += 1
                    
                    # Upload in batches
                    if len(points) >= 1000:
                        client.upsert(collection_name=collection_name, points=points)
                        points = []
                
                del values_df, cell_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error embedding column {table_column}: {e}")
                continue
    
    # Upload remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    logger.info(f"Embedded {total_cells} unique cell values into collection {collection_name}")
    return total_cells


@monitor_resources()
def run_cell_to_column_mapping_benchmark(
    cfg: DictConfig,
    cell_embedding_component: CellEmbeddingInterface,
    databases_path: Path,
    queries: List[Dict]
) -> List[Dict]:
    """Run the cell-to-column mapping benchmark using Qdrant for ANN search."""
    
    # Setup Qdrant client
    client, qdrant_path = get_qdrant_client(cfg)
    
    results = []
    db_collections = {}  # Track which collections have been created
    
    # Limit number of queries if specified
    max_queries = cfg.task.get("max_queries", len(queries))
    queries = queries[:max_queries]
    
    # Query mode: "full_nl" or "extracted_values" or "both"
    query_mode = cfg.task.get("query_mode", "both")
    
    # Top-K cells to retrieve
    top_k_cells = cfg.task.get("top_k_cells", 100)
    
    for query_idx, query in enumerate(tqdm(queries, desc="Processing queries")):
        db_id = query["db_id"]
        question = query["question"]
        gold_columns = query["gold_columns"]
        extracted_values = query.get("extracted_values_from_NL", None)
        
        if not gold_columns:
            continue
        
        # Setup database collection if not already done
        if db_id not in db_collections:
            db_path = databases_path / db_id / f"{db_id}.sqlite"
            if not db_path.exists():
                logger.warning(f"Database not found: {db_path}")
                continue
            
            collection_name = f"bird_{db_id}"
            completed_file_path = qdrant_path / "collection" / collection_name / "COMPLETED"
            
            # Check if collection already exists and is complete
            if client.collection_exists(collection_name=collection_name) and completed_file_path.exists():
                logger.info(f"Collection {collection_name} already exists and is complete")
            else:
                # Delete if exists but incomplete
                if client.collection_exists(collection_name=collection_name):
                    client.delete_collection(collection_name=collection_name)
                    logger.info(f"Deleted incomplete collection {collection_name}")
                
                # Create new collection
                logger.info(f"Creating cell embeddings for database {db_id}")
                schema = load_database_schema(db_path)
                
                # Collect matched values for this database from all queries
                matched_values_by_column = collect_matched_values_for_database(queries, db_id)
                logger.info(f"Collected matched values for {len(matched_values_by_column)} columns in {db_id}")
                
                num_cells = create_qdrant_collection_for_database(
                    client=client,
                    collection_name=collection_name,
                    cell_embedding_component=cell_embedding_component,
                    db_path=db_path,
                    schema=schema,
                    matched_values_by_column=matched_values_by_column,
                    max_unique_values_per_column=cfg.task.get("max_unique_values_per_column", 1000)
                )
                
                # Mark as complete
                completed_file_path.parent.mkdir(parents=True, exist_ok=True)
                completed_file_path.write_text(f"Embedded {num_cells} cells")
                logger.info(f"Marked collection {collection_name} as complete")
            
            db_collections[db_id] = collection_name
        
        collection_name = db_collections[db_id]
        
        # Prepare ground truth
        gold_column_set = {f"{gc['table']}.{gc['column']}" for gc in gold_columns}
        
        # Process based on query mode
        modes_to_run = []
        if query_mode == "both":
            modes_to_run = ["full_nl", "extracted_values"]
        else:
            modes_to_run = [query_mode]
        
        for mode in modes_to_run:
            # Skip extracted_values mode if no extracted values available
            if mode == "extracted_values" and not extracted_values:
                continue
            
            # Create query embedding
            try:
                if mode == "full_nl":
                    query_embedding = embed_query_text(cell_embedding_component, question)
                    mode_suffix = "_full_nl"
                else:  # extracted_values
                    query_embedding = embed_extracted_values(cell_embedding_component, extracted_values)
                    mode_suffix = "_extracted"
            except Exception as e:
                logger.error(f"Error embedding query in mode {mode}: {e}")
                continue
            
            # Search Qdrant for top-K nearest cells
            hits = client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k_cells,
                with_payload=True
            )
            
            # Count which columns the retrieved cells belong to
            column_counts = Counter()
            for hit in hits:
                table_column = hit.payload["table_column"]
                column_counts[table_column] += 1
            
            # Rank columns by count (how many of their cells were retrieved)
            ranked_columns = [col for col, count in column_counts.most_common()]
            
            # Compute metrics at different K values
            k_values = [1, 3, 5, 10, 20]
            metrics = {}
            
            for k in k_values:
                top_k = ranked_columns[:k]
                correct = sum(1 for col in top_k if col in gold_column_set)
                
                precision = correct / k if k > 0 else 0.0
                recall = correct / len(gold_column_set) if gold_column_set else 0.0
                
                metrics[f"precision@{k}{mode_suffix}"] = precision
                metrics[f"recall@{k}{mode_suffix}"] = recall
            
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
            
            metrics[f"mrr{mode_suffix}"] = mrr
            metrics[f"map{mode_suffix}"] = average_precision
            
            results.append({
                "query_idx": query_idx,
                "db_id": db_id,
                "question": question,
                "query_mode": mode,
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
    
    # Group by query mode
    by_mode = defaultdict(list)
    for r in results:
        mode = r.get("query_mode", "unknown")
        by_mode[mode].append(r)
    
    # Compute metrics for each mode
    for mode, mode_results in by_mode.items():
        metric_keys = [k for k in mode_results[0].keys() 
                       if k not in ["query_idx", "db_id", "question", "query_mode", 
                                    "num_gold_columns", "num_retrieved_columns"]]
        
        for key in metric_keys:
            values = [r[key] for r in mode_results if key in r]
            if values:
                aggregate[f"mean_{key}"] = statistics.mean(values)
                if len(values) > 1:
                    aggregate[f"std_{key}"] = statistics.stdev(values)
    
    return aggregate


def main(cfg: DictConfig):
    """Main entry point for the cell-to-column mapping benchmark."""
    logger.info("Started run_cell_to_column_mapping_benchmark")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True)
    
    # Load dataset config
    dataset_config_path = Path(get_original_cwd()) / "benchmark_src" / "config" / "dataset" / f"{cfg.dataset_name}.yaml"
    dataset_cfg = OmegaConf.load(str(dataset_config_path))
    OmegaConf.set_struct(cfg, False)
    cfg.dataset = dataset_cfg.dataset
    OmegaConf.set_struct(cfg, True)
    
    # Load benchmark data
    databases_path, queries = load_benchmark_data(cfg)
    
    # Instantiate the embedding approach class
    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)
    
    # Load cell embedding component
    cell_embedding_component = embedder._load_component(
        "cell_embedding_component",
        "CellEmbeddingComponent",
        CellEmbeddingInterface
    )
    
    # Setup model
    _, resource_metrics_setup = component_utils.run_model_setup(
        component=cell_embedding_component,
        dataset_information=None
    )
    
    # Run benchmark
    all_results, resource_metrics_task = run_cell_to_column_mapping_benchmark(
        cfg=cfg,
        cell_embedding_component=cell_embedding_component,
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
    logger.info(f"Processed {len(all_results)} query evaluations")
    logger.info("Aggregate Metrics:")
    for key, value in aggregate_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save results
    result_utils.save_results(cfg=cfg, metrics=aggregate_metrics)
    
    # Save detailed per-query results
    with open("results_per_query.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("Benchmark complete!")

# Made with Bob
