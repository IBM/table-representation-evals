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
        datalake_tables = glob.glob(f"{dataset}/**/*{file_format}", recursive=True)
        # for column similarity tasks, groups of tables are in datalakes
        # maintain a mapping of datalake to tables
        table2dfs = {}
    #   TBD unclear why logger does not work
    #   for table in tqdm(datalake_tables, desc="Computing Embeddings", file=logger):
        for table in tqdm(datalake_tables, desc="Computing Embeddings"):
            try:
                logger.debug(f'loading table: {table}')
                df = load_dataframe(table, file_format=file_format)
                print(df.columns)
                table2dfs[table] = df
            except:
                logger.error(f"Cannot find table: {table}")

        if cfg.dataset_name.lower() == "valentine":
            gt = glob.glob(f"{dataset}/*mapping.json", recursive=True)
        elif cfg.dataset_name.lower() == "wikijoin_small":
            gt = glob.glob(f"{dataset}/gt_small.*", recursive=True)
        elif cfg.dataset_name.lower() == "nextia":
            d = dataset.replace('/datalake', '')
            gt = glob.glob(f"{d}/**/gt.*", recursive=True)
        else:
            gt = glob.glob(f"{dataset}/**/gt.*", recursive=True)
            gt = [x for x in gt if x.endswith('json') or x.endswith('jsonl') or x.endswith('pickle')]

        assert len(gt) == 1, gt
        gt = gt[0]

        if gt.endswith('.json'):
            gt_data = json.load(open(gt, 'r'))
        elif gt.endswith('.jsonl'):
            gt_data = convert_to_dict_of_list(gt)
        elif gt.endswith('.pickle'):
            raise NotImplementedError
        else:
            raise NotImplementedError

        if cfg.dataset_name.lower() == "wikijoin_small":
            l = [x.split('.')[0] + '.csv' for x in gt_data.keys()]
            alx = []
            for x in gt_data.values():
                for y in x:
                    alx.append(y.split('.')[0] + '.csv')
            fls = l + alx
            new_table2dfs = {}
            for k in table2dfs:
                x = k.split('/')[-1]
                if x in fls:
                    new_table2dfs[k] = table2dfs[k]
            assert len(new_table2dfs) < len(table2dfs)
            assert len(new_table2dfs) > 0, new_table2dfs
            print('created a small version of wikijoin', len(new_table2dfs))
            table2dfs = new_table2dfs

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

    dataset_config_path = Path(get_original_cwd()) / Path("benchmark_src/config/dataset") / f"{cfg.dataset_name}.yaml"

    if not dataset_config_path.exists():
        print(os.getcwd())
        print(f"Could not find dataset config path: {dataset_config_path}")
    try:
        dataset_cfg = OmegaConf.load(str(dataset_config_path))
    except Exception as e:
        print(f"Could not load dataset config: {e}")

    for testcase in test_cases:
        all_columns = {}
        table2dfs, gt_data, dataset = test_cases[testcase]
        if not os.path.exists(f'{results_file}/{dataset}.pkl'):
            for table in table2dfs:
                t = os.path.basename(table).replace('.csv', '').replace('.df', '')
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

        for k in all_columns:
            if k.startswith('animal'):
                for v in all_columns[k]:
                    print("table column", k, v)
        new_gt = {}
        top_k = 0
        print(all_columns.keys())
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
            for c in new_gt:
                if len(new_gt[c]) > top_k:
                    top_k = len(new_gt[c])
        else:
            # set k to max of gt data
            new_gt = {}
        
            for k in gt_data:
                table = k.split('.')[0]
                if table in all_columns:
                    search_sources.append(all_columns[table][k])
                    searched_indexes.append((table, k))
                    new_gt[k] = gt_data[k]
                    if len(gt_data[k]) > top_k:
                        top_k = len(gt_data[k])

        search_sources = np.asarray(search_sources)
        print('new_gt', new_gt)
        print('search indexes', searched_indexes)
        assert len(new_gt) == len(search_sources)
            
        # check if there are column
        # need to add 1 to top_k so the search is adjusted for returning searched column
        D, I = index.search(search_sources, top_k + 1)

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
            #logger.debug('neighbors for ', x, i.tolist(), source_col, joinable_list)

        missing_queries = set(new_gt.keys())-set(result.keys())

        logger.info(f"Result file is missing {len(missing_queries)} queries out of {len(new_gt)}")
        assert len(missing_queries) == 0, missing_queries
        logger.info(f'top k is set to:{top_k}')

        MRR = compute_mrr_from_list(new_gt, result, top_k)
        MAP = compute_map_from_list(new_gt, result, top_k)

        Precision, Recall  = compute_precision_recall_at_k(new_gt, result, top_k)
        metric_res[dataset] = {'MRR': MRR, 'MAP': MAP, 'Precision':Precision, 'Recall': Recall}
        print('Expected', dataset, new_gt)
        print('Result', result, top_k)
        print(dataset, 'MRR', "MAP", "Precision", "Recall", MRR, MAP, Precision, Recall)

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
    cluster_ranges =[1000]

    result, resource_metrics_task = run_inference_based_on_column_embeddings(cluster_ranges=cluster_ranges, cfg=cfg)
    result_metrics, resource_metrics_setup = result
    
    # save resource metrics to disk
    if resource_metrics_setup:
        save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)

    # save the other metrics to disk
    result_utils.save_results(cfg=cfg, metrics=result_metrics)

    

