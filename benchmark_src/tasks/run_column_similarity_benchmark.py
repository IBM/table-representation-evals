from omegaconf import DictConfig, OmegaConf
import multiprocessing
import logging
import json
import pickle
import os
import sys
from pathlib import Path
import statistics
from tqdm import tqdm
from hydra.utils import get_original_cwd
import numpy as np
import glob
import gc
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Add ContextAwareJoin to Python path
# Path to src folder in ContextAwareJoin submodule
context_aware_join_src = Path(get_original_cwd()) / "ContextAwareJoin" / "src"

# Make sure it exists
if not context_aware_join_src.exists():
    raise FileNotFoundError(f"{context_aware_join_src} does not exist!")

# Add to sys.path if not already present
if str(context_aware_join_src) not in sys.path:
    sys.path.insert(0, str(context_aware_join_src))
    print(f"Added {context_aware_join_src} to sys.path")


from myutils.evaluation import compute_mrr_from_list, compute_map_from_list, compute_ndcg, compute_precision_recall_at_k
from myutils.utilities import load_dataframe, convert_to_dict_of_list, get_groundtruth_with_scores

from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk
from benchmark_src.utils import framework, result_utils
from benchmark_src.tasks import component_utils

logger = logging.getLogger(__name__)


def get_qdrant_client(cfg: DictConfig) -> QdrantClient:
    qdrant_path = Path(get_original_cwd()) / cfg.cache_dir / "qdrant_storage"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    logger.info(f"Initialized Qdrant client with persistent storage at {qdrant_path}")
    return client

def load_benchmark_data(cfg):
    if cfg.dataset_name == 'wikijoin_small':
        dataset_dir = str(Path(get_original_cwd()) / "ContextAwareJoin" / "datasets" / "wikijoin")
    else:
        dataset_dir = str(Path(get_original_cwd()) / "ContextAwareJoin" / "datasets" / cfg.dataset_name)
    logger.debug(f"Looking for datasets in dir: {dataset_dir}")

    assert Path(dataset_dir).exists(), f"Could not find dataset dir: {dataset_dir}"

    if cfg.dataset_name == 'opendata':
        file_format = '.df'
    else:
        file_format = '.csv'

    # Helper to find all lowest-level subfolders
    def get_leaf_dirs(root_dir, keep=None):
        leaf_dirs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if keep and keep not in dirpath:
                continue
            # If a directory has no subdirectories, it is a leaf
            if not dirnames:
                leaf_dirs.append(dirpath)
        return leaf_dirs


    test_cases = {}
    if cfg.dataset_name.lower() == "valentine":
        leaf_dirs = get_leaf_dirs(dataset_dir, keep='Joinable')
    elif cfg.dataset_name.lower() == "nextia":
        leaf_dirs = get_leaf_dirs(dataset_dir)
    else:
        leaf_dirs = [dataset_dir]

    if len(leaf_dirs) == 0:
        raise ValueError(f"Did not find the dataset. leaf_dirs={leaf_dirs}")

    print('LEAF_DIRS', leaf_dirs)
    
    for dataset in leaf_dirs:
        ############################
        # Load GT first
        ############################
        print("Dataset name", cfg.dataset_name.lower())

        if cfg.dataset_name.lower() == "valentine":
            gt = glob.glob(f"{dataset}/*mapping.json", recursive=True)
        elif cfg.dataset_name.lower() == "wikijoin_small":
            gt = glob.glob(f"{dataset}/gt_small.*", recursive=True)
        elif cfg.dataset_name.lower() == "nextia":
            d = dataset.replace('/datalake', '')
            print(d)
            gt = glob.glob(f"{d}/**/gt.*", recursive=True)
        else:
            gt = glob.glob(f"{dataset}/**/gt.*", recursive=True)
            gt = [x for x in gt if x.endswith('json') or x.endswith('jsonl') or x.endswith('pickle')]
            print(f"else case. Found gt: {gt}")

        assert len(gt) == 1, f"Error: gt is {gt}"
        gt = gt[0]

        if gt.endswith('.json'):
            gt_data = json.load(open(gt, 'r'))
        elif gt.endswith('.jsonl'):
            gt_data = convert_to_dict_of_list(gt)
        elif gt.endswith('.pickle'):
            raise NotImplementedError
        else:
            raise NotImplementedError

        ############################
        # Load datalake tables - STORE PATHS ONLY
        ############################
        datalake_tables = glob.glob(f"{dataset}/**/*{file_format}", recursive=True)
        
        # Filter tables for wikijoin_small upfront
        if cfg.dataset_name.lower() == "wikijoin_small":
            l = [x.split('.')[0] + '.csv' for x in gt_data.keys()]
            alx = []
            for x in gt_data.values():
                for y in x:
                    alx.append(y.split('.')[0] + '.csv')
            fls = set(l + alx)  # Use set for faster lookup
            
            filtered_tables = []
            for table in datalake_tables:
                x = table.split('/')[-1]
                if x in fls:
                    filtered_tables.append(table)
            
            assert len(filtered_tables) < len(datalake_tables)
            assert len(filtered_tables) > 0, filtered_tables
            print('created a small version of wikijoin', len(filtered_tables))
            datalake_tables = filtered_tables

        # Validate tables can be loaded (optional check)
        use_tqdm = len(datalake_tables) > 20
        iterator = tqdm(datalake_tables, desc="Validating Datasets") if use_tqdm else datalake_tables
        valid_tables = []
        for table in iterator:
            try:
                logger.debug(f'validating table: {table}')
                df = load_dataframe(table, file_format=file_format)
                # make sure there is at least one row
                if len(df) == 0:
                    continue
                if len(df.columns) == 1:
                    raise Exception(f"Table {table} loaded with {len(df.columns)} columns: {df.columns}")
                valid_tables.append(table)
                del df  # Free memory immediately
            except Exception as e:
                logger.error(e)

        # Store table paths instead of loaded dataframes
        table_paths = {table: file_format for table in valid_tables}
        test_cases[dataset] = table_paths, gt_data, dataset.replace('/', '_')

    return test_cases


@monitor_resources()
def run_inference_based_on_column_embeddings(cluster_ranges, cfg):

    # Setup Qdrant client
    client = get_qdrant_client(cfg)

    metric_res = {}
    resource_metrics_setup = None

    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)

    # Load test cases (only paths + GT)
    test_cases = load_benchmark_data(cfg)
    logger.info("Done loading the test cases.")

    column_embedding_component = embedder._load_component(
        "column_embedding_component",
        "ColumnEmbeddingComponent",
        ColumnEmbeddingInterface
    )

    # Model setup
    _, resource_metrics_setup = component_utils.run_model_setup(
        component=column_embedding_component,
        input_table=None,
        dataset_information=None
    )

    for testcase in test_cases:
        logger.info(f"Processing testcase: {testcase}")
        table_paths, gt_data, dataset_name = test_cases[testcase]

        # First, infer embedding dimension using first available table
        first_table = next(iter(table_paths))
        df = load_dataframe(first_table, file_format=table_paths[first_table])
        sample_embeddings, column_names = column_embedding_component.create_column_embeddings_for_table(df)
        if hasattr(sample_embeddings, 'cpu'):
            sample_embeddings = sample_embeddings.cpu().numpy()
        embedding_dim = sample_embeddings[0].size
        del df, sample_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Create Qdrant collection if not exists
        if client.collection_exists(collection_name=cfg.run_identifier + "_" + dataset_name):
            client.delete_collection(collection_name=cfg.run_identifier + "_" + dataset_name)
            logger.info(f"Deleted existing collection {cfg.run_identifier}_{dataset_name}")

        client.create_collection(
            collection_name=cfg.run_identifier + "_" + dataset_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
        logger.info(f"Created Qdrant collection {dataset_name} with vector size {embedding_dim}")

        # Streaming embedding and upsert
        points = []
        current_id = 0

        for table_path in tqdm(table_paths, desc=f"Embedding tables for {dataset_name}"):
            df = load_dataframe(table_path, file_format=table_paths[table_path])
            tname = os.path.basename(table_path).replace('.csv', '').replace('.df', '')

            column_embeddings, column_names = column_embedding_component.create_column_embeddings_for_table(df)

            # Convert embeddings to numpy if needed
            if hasattr(column_embeddings, 'cpu'):
                column_embeddings = column_embeddings.cpu().numpy()
            elif isinstance(column_embeddings, list):
                column_embeddings = np.array([emb.cpu().numpy() if hasattr(emb, 'cpu') else emb for emb in column_embeddings])

            # Create Qdrant Points
            for idx, col_name in enumerate(column_names):
                full_col_name = f"{tname}.{col_name}"
                points.append(PointStruct(
                    id=current_id,
                    vector=column_embeddings[idx].tolist(),
                    payload={"table": tname, "column": full_col_name}
                ))
                current_id += 1

            # Upload in batches to save memory
            if len(points) >= 1000:
                client.upsert(collection_name=cfg.run_identifier + "_" + dataset_name, points=points)
                points = []

            del df, column_embeddings, column_names
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Upload any remaining points
        if points:
            client.upsert(collection_name=cfg.run_identifier + "_" + dataset_name, points=points)
            points = []

        # Prepare queries and ground truth
        new_gt = {}
        search_queries = []
        top_k = 0

        if cfg.dataset_name.lower() == "valentine":
            for match in gt_data['matches']:
                col = f"{match['source_table']}.{match['source_column']}"
                search_queries.append(col)
                if col not in new_gt:
                    new_gt[col] = []
                new_gt[col].append(f"{match['target_table']}.{match['target_column']}")
                top_k = max(top_k, len(new_gt[col]))
        else:
            for k in gt_data:
                table = k.split('.')[0]
                search_queries.append(k)
                new_gt[k] = gt_data[k]
                top_k = max(top_k, len(gt_data[k]))

        # Search each query in Qdrant
        result = {}
        for query_col in tqdm(search_queries, desc="Searching columns"):
            # Fetch query embedding
            table_name, col_name = query_col.split('.', 1)
            table_path = next((p for p in table_paths if os.path.basename(p).startswith(table_name)), None)
            df = load_dataframe(table_path, file_format=table_paths[table_path])
            col_idx = list(df.columns).index(col_name)
            query_embedding = column_embedding_component.create_column_embeddings_for_table(df)[0][col_idx]
            if hasattr(query_embedding, 'cpu'):
                query_embedding = query_embedding.cpu().numpy()

            # Search in Qdrant
            hits = client.search(
                collection_name=dataset_name,
                query_vector=query_embedding.tolist(),
                limit=top_k + 1,
                with_payload=True
            )

            retrieved_cols = [hit.payload["column"] for hit in hits if hit.payload["column"] != query_col]
            result[query_col] = retrieved_cols

            del df
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Compute metrics
        MRR = compute_mrr_from_list(new_gt, result, top_k)
        MAP = compute_map_from_list(new_gt, result, top_k)
        Precision, Recall = compute_precision_recall_at_k(new_gt, result, top_k)

        metric_res[dataset_name] = {
            "MRR": MRR,
            "MAP": MAP,
            "Precision": Precision,
            "Recall": Recall
        }

        logger.info(f"{dataset_name} -> MRR: {MRR}, MAP: {MAP}, Precision: {Precision}, Recall: {Recall}")

    # Summarize results
    summary_result = {}
    for key in metric_res:
        for metric in metric_res[key]:
            summary_result.setdefault(metric, []).append(metric_res[key][metric])

    final_summary = {}
    for metric in summary_result:
        final_summary[metric] = statistics.mean(summary_result[metric])
        if len(summary_result[metric]) > 1:
            final_summary[metric + "_std"] = statistics.stdev(summary_result[metric])

    return final_summary, resource_metrics_setup

def main(cfg: DictConfig):
    logger.debug(f"Started run_column_similarity_benchmark")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True) 

    # run inference with model
    logger.info(f"Running column similarity based on column embeddings")
    cluster_ranges = [1000]

    result, resource_metrics_task = run_inference_based_on_column_embeddings(cluster_ranges=cluster_ranges, cfg=cfg)
    result_metrics, resource_metrics_setup = result
    
    # save resource metrics to disk
    if resource_metrics_setup:
        save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)

    # save the other metrics to disk
    result_utils.save_results(cfg=cfg, metrics=result_metrics)