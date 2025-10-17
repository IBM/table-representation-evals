from omegaconf import DictConfig
import multiprocessing
import logging
from hydra.utils import get_original_cwd
import json
import pickle
import os
import sys
# Add ContextAwareJoin to Python path
context_aware_join_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ContextAwareJoin')
if context_aware_join_path not in sys.path:
    sys.path.insert(0, context_aware_join_path)

from src.myutils.evaluation import compute_mrr_from_list, compute_map_from_list, compute_ndcg, compute_precision_recall_at_k
from src.myutils.utilities import convert_to_dict_of_list
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk
from benchmark_src.utils import gather_results, framework
from benchmark_src.tasks import component_utils
import faiss
import numpy as np
import glob
from tqdm import tqdm
from src.myutils.utilities import load_dataframe, convert_to_dict_of_list, get_groundtruth_with_scores
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


def load_benchmark_data(cfg):
    dataset_dir = str(Path(get_original_cwd()) / Path(cfg.benchmark_datasets_dir) / cfg.dataset_name)
    if cfg.dataset_name == 'opendata':
        file_format = '.df'
    else:
        file_format = '.csv'

    # Helper to find all lowest-level subfolders
    def get_leaf_dirs(root_dir, keep):
        leaf_dirs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if keep not in dirpath:
                continue
            # If a directory has no subdirectories, it is a leaf
            if not dirnames:
                leaf_dirs.append(dirpath)
        return leaf_dirs


    test_cases = {}
    if cfg.dataset_name.lower() == "valentine":
        leaf_dirs = get_leaf_dirs(dataset_dir, keep='Joinable')
    else:
        leaf_dirs = [dataset_dir]

    for dataset in leaf_dirs:
        datalake_tables = glob.glob(f"{dataset}/**/*{file_format}", recursive=True)
        # for column similarity tasks, groups of tables are in datalakes
        # maintain a mapping of datalake to tables
        table2dfs = {}
    #   TBD unclear why logger does not work
    #   for table in tqdm(datalake_tables, desc="Computing Embeddings", file=logger):
        for table in tqdm(datalake_tables, desc="Computing Embeddings"):
            try:
                print('loading table', table)
                df = load_dataframe(table, file_format=file_format)
                table2dfs[table] = df
            except:
                print("Cannot find table", table)

        if cfg.dataset_name.lower() == "valentine":
            gt = glob.glob(f"{dataset}/*mapping.json", recursive=True)
        else:
            gt = glob.glob(f"{dataset}/**/gt.*", recursive=True)
            gt = [x for x in gt if x.endswith('json') or x.endswith('jsonl') or x.endswith('pickle')]

        assert len(gt) == 1
        gt = gt[0]

        if gt.endswith('.json'):
            gt_data = json.load(open(gt, 'r'))
        elif gt.endswith('.jsonl'):
            gt_data = convert_to_dict_of_list(gt)
        elif gt.endswith('.pickle'):
            raise NotImplementedError
        else:
            raise NotImplementedError
        test_cases[dataset] = table2dfs, gt_data, dataset.replace('/', '_')


    return test_cases


@monitor_resources()
def run_inference_based_on_column_embeddings(cluster_ranges, cfg):
    # get column embeddings and assert they have the correct format and shape
    # instantiate the embedding approach clas
    results_file = os.path.join(cfg.cache_dir, "all_column_embeddings")
    if not os.path.exists(results_file):
        os.makedirs(results_file)

    metric_res = {}
    resource_metrics_setup = None

    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)

    test_cases = load_benchmark_data(cfg)
    column_embedding_component = embedder._load_component("column_embedding_component", "ColumnEmbeddingComponent",
                                                              ColumnEmbeddingInterface)
    ## setup model
    _, resource_metrics_setup = component_utils.run_model_setup(component=column_embedding_component,
                                                                    input_table=None, dataset_information=None)

    for testcase in test_cases:
        all_columns = {}
        table2dfs, gt_data, dataset = test_cases[testcase]

        if not os.path.exists(f'{results_file}/{dataset}.pkl'):
            for table in table2dfs:
                t = os.path.basename(table).replace('.csv', '')
                print('at table', t)
                all_columns[t] = {}
                column_embeddings, column_names = column_embedding_component.create_column_embeddings_for_table(input_table=table2dfs[table])
                for idx, c in enumerate(column_names):
                    c = t + '.' + c
                    all_columns[t][c] = column_embeddings[idx]
            with open(f'{results_file}/{dataset}.pkl', "wb") as file:
                pickle.dump(all_columns, file)
        else:
            with open(f'{results_file}/{dataset}.pkl', "rb") as file:
                all_columns = pickle.load(file)
                resource_metrics_setup = None

        all_cols = []
        all_indexes = {}
        i = 0
        for table in all_columns:
            for column in all_columns[table]:
                all_indexes[i] = table, column
                i += 1
                all_cols.append(all_columns[table][column])
        assert len(all_cols) == len(all_indexes)
        arr = np.asarray(all_cols)

        index = faiss.IndexFlatL2(arr[0].size)
        index.add(arr)
        search_sources = []
        searched_indexes = []

        new_gt = {}
        if cfg.dataset_name == 'valentine':
            for match in gt_data['matches']:
                table = match['source_table']
                column = table + '.' + match['source_column']
                if column not in new_gt:
                    new_gt[column] = []
                l = new_gt[column]
                searched_indexes.append((table, column))
                search_sources.append(all_columns[table][column])
                column = match['target_table'] + '.' + match['target_column']
                l.append(column)
        else:
            new_gt = gt_data
            for k in gt_data:
                table = k.split('.')[0]
                search_sources.append(all_columns[table][k])

        search_sources = np.asarray(search_sources)

        k = 2  # TBD how do we set k from config

        D, I = index.search(search_sources, k)

        result = {}
        for x, i in enumerate(I):
            source_table, source_col = searched_indexes[x]
            joinable_list = []
            result[source_col] = joinable_list
            for y, neighbor in enumerate(i):
                target_table, target_col = all_indexes[neighbor]
                r = target_col
                joinable_list.append(r)

        missing_queries = set(new_gt.keys())-set(result.keys())

        logger.debug(f"Result file is missing {len(missing_queries)} queries out of {len(new_gt)}")
        assert len(missing_queries) == 0, missing_queries

        MRR = compute_mrr_from_list(new_gt, result, 1)
        MAP = compute_map_from_list(new_gt, result, 1)
        Precision, Recall  = compute_precision_recall_at_k(new_gt, result, 1)
        metric_res[dataset] = {'MRR': MRR, 'MAP': MAP, 'Precision':Precision, 'Recall': Recall}

        print(dataset, 'MRR', "MAP", "Precision", "Recall", MRR, MAP, Precision, Recall)

    result = {}
    for dataset in metric_res:
        for key in metric_res[dataset]:
            if key not in result:
                result[key] = []
            result[key].append(metric_res[dataset][key])

    print('Result', result)
    summary_result = {}
    for key in result:
        summary_result[key + '_mean'] = statistics.mean(result[key])
        summary_result[key + '_std'] = statistics.stdev(result[key])

    return summary_result, resource_metrics_setup


def main(cfg: DictConfig):
    logger.debug(f"Started embedding")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True) 

    # run inference with model
    logger.info(f"Running column similarity based on column embeddings")
    cluster_ranges =[1000]

    result, resource_metrics_task = run_inference_based_on_column_embeddings(cluster_ranges=cluster_ranges, cfg=cfg)
    result_metrics, resource_metrics_setup = result
    
    # save resource metrics to disk
    if resource_metrics_setup:
        save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)

    # save the other metrics to disk
    gather_results.save_results(cfg=cfg, metrics=result_metrics)

    

