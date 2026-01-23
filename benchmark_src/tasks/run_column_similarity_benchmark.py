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
import faiss
import numpy as np
import glob

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
    import gc
    import torch
    
    results_file = os.path.join(cfg.cache_dir, "all_column_embeddings")
    if not os.path.exists(results_file):
        os.makedirs(results_file)

    metric_res = {}
    resource_metrics_setup = None

    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)

    test_cases = load_benchmark_data(cfg)
    logger.info(f"Done loading the test cases.")
    column_embedding_component = embedder._load_component("column_embedding_component", "ColumnEmbeddingComponent",
                                                              ColumnEmbeddingInterface)
    ## setup model
    _, resource_metrics_setup = component_utils.run_model_setup(component=column_embedding_component,
                                                                    input_table=None, dataset_information=None)

    dataset_config_path = Path(get_original_cwd()) / Path("benchmark_src/config/dataset") / f"{cfg.dataset_name}.yaml"

    if not dataset_config_path.exists():
        print(os.getcwd())
        print(f"Could not find dataset config path: {dataset_config_path}")
    try:
        dataset_cfg = OmegaConf.load(str(dataset_config_path))
    except Exception as e:
        print(f"Could not load dataset config: {e}")

    for testcase in test_cases:
        logger.info(f"Working on testcase {testcase}")
        table_paths, gt_data, dataset = test_cases[testcase]
        
        logger.info(f"Creating embeddings and building index in streaming fashion")
        
        # Build index incrementally without storing all embeddings
        all_indexes = {}  # Maps index position -> (table, column)
        index = None
        embedding_dim = None
        current_idx = 0
        
        # First pass: Build FAISS index incrementally
        for table_idx, table_path in enumerate(table_paths):
            file_format = table_paths[table_path]
            df = load_dataframe(table_path, file_format=file_format)
            
            t = os.path.basename(table_path).replace('.csv', '').replace('.df', '')
            print(f'at table *{t}* with shape {df.shape}, progress: {table_idx+1} / {len(table_paths)}')
            
            column_embeddings, column_names = column_embedding_component.create_column_embeddings_for_table(input_table=df)
            assert len(column_names) > 1, f"Parsing issue? Got {column_names} as column name."
            
            # Move embeddings to CPU and convert to numpy immediately
            if hasattr(column_embeddings, 'cpu'):
                column_embeddings = column_embeddings.cpu().numpy()
            elif isinstance(column_embeddings, list):
                column_embeddings = np.array([emb.cpu().numpy() if hasattr(emb, 'cpu') else emb for emb in column_embeddings])
            
            # Initialize FAISS index on first iteration
            if index is None:
                embedding_dim = column_embeddings[0].size
                index = faiss.IndexFlatL2(embedding_dim)
            
            # Add embeddings to index and track mappings
            for idx, c in enumerate(column_names):
                full_col_name = t + '.' + c
                all_indexes[current_idx] = (t, full_col_name)
                current_idx += 1
            
            # Add all column embeddings from this table to index
            index.add(column_embeddings)
            
            # Free memory immediately
            del df, column_embeddings, column_names
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        print(f"Built FAISS index with {len(all_indexes)} columns from {len(table_paths)} tables")
        
        # Prepare ground truth and search queries
        new_gt = {}
        top_k = 0
        search_queries = []  # List of (table, column) tuples to search
        
        if cfg.dataset_name == 'valentine':
            for match in gt_data['matches']:
                table = match['source_table']
                column = table + '.' + match['source_column']
                if column not in new_gt:
                    new_gt[column] = []
                l = new_gt[column]
                search_queries.append((table, column))
                column = match['target_table'] + '.' + match['target_column']
                l.append(column)
            for c in new_gt:
                if len(new_gt[c]) > top_k:
                    top_k = len(new_gt[c])
        else:
            for k in gt_data:
                table = k.split('.')[0]
                # Check if this column exists in our index
                column_exists = any(all_indexes[i][1] == k for i in all_indexes if all_indexes[i][0] == table)
                if column_exists:
                    search_queries.append((table, k))
                    new_gt[k] = gt_data[k]
                    if len(gt_data[k]) > top_k:
                        top_k = len(gt_data[k])
                # otherwise skip this column as it is not in our index
        
        print('new_gt', new_gt)
        print('search queries', search_queries)
        assert len(new_gt) == len(search_queries)
        
        # Second pass: Extract search embeddings for queries only
        search_sources = []
        searched_indexes = []
        
        for table_path in table_paths:
            file_format = table_paths[table_path]
            t = os.path.basename(table_path).replace('.csv', '').replace('.df', '')
            
            # Check if this table has any queries we need
            relevant_queries = [(tbl, col) for tbl, col in search_queries if tbl == t]
            if not relevant_queries:
                continue
            
            # Load table and get embeddings
            df = load_dataframe(table_path, file_format=file_format)
            column_embeddings, column_names = column_embedding_component.create_column_embeddings_for_table(input_table=df)
            
            # Move to CPU
            if hasattr(column_embeddings, 'cpu'):
                column_embeddings = column_embeddings.cpu().numpy()
            elif isinstance(column_embeddings, list):
                column_embeddings = np.array([emb.cpu().numpy() if hasattr(emb, 'cpu') else emb for emb in column_embeddings])
            
            # Extract only the embeddings we need for queries
            for tbl, full_col in relevant_queries:
                col_name = full_col.split('.', 1)[1]  # Remove table prefix
                if col_name in column_names:
                    idx = column_names.index(col_name)
                    search_sources.append(column_embeddings[idx])
                    searched_indexes.append((tbl, full_col))
            
            # Free memory
            del df, column_embeddings, column_names
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        search_sources = np.asarray(search_sources)
        print(f"Searching with {len(search_sources)} query embeddings")
        assert len(new_gt) == len(search_sources)
        
        # Perform search
        D, I = index.search(search_sources, top_k + 1)
        
        # Process results
        result = {}
        for x, i in enumerate(I):
            source_table, source_col = searched_indexes[x]
            joinable_list = []
            result[source_col] = joinable_list
            for y, neighbor in enumerate(i):
                target_table, target_col = all_indexes[neighbor]
                r = target_col
                joinable_list.append(r)
            if source_col in joinable_list:
                joinable_list.remove(source_col)

        missing_queries = set(new_gt.keys())-set(result.keys())
        logger.info(f"Result file is missing {len(missing_queries)} queries out of {len(new_gt)}")
        assert len(missing_queries) == 0, missing_queries
        logger.info(f'top k is set to:{top_k}')

        MRR = compute_mrr_from_list(new_gt, result, top_k)
        MAP = compute_map_from_list(new_gt, result, top_k)
        Precision, Recall = compute_precision_recall_at_k(new_gt, result, top_k)
        
        metric_res[dataset] = {'MRR': MRR, 'MAP': MAP, 'Precision':Precision, 'Recall': Recall}
        print('Expected', dataset, new_gt)
        print('Result', result, top_k)
        print(dataset, 'MRR', "MAP", "Precision", "Recall", MRR, MAP, Precision, Recall)
        
        # Clean up for next test case
        del index, all_indexes, search_sources, D, I
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {}
    for dataset in metric_res:
        for key in metric_res[dataset]:
            if key not in result:
                result[key] = []
            result[key].append(metric_res[dataset][key])

    summary_result = {}
    if cfg.dataset_name == 'valentine':
        for key in result:
            summary_result[key] = statistics.mean(result[key])
            summary_result[key + '_std'] = statistics.stdev(result[key])
    else:
        for key in result:
            summary_result[key] = result[key][0]

    return summary_result, resource_metrics_setup


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