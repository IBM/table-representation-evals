from omegaconf import DictConfig
import multiprocessing
import logging
from hydra.utils import get_original_cwd
import json
import pickle
import os

from ContextAwareJoin.src.myutils.evaluation import compute_mrr_from_list, compute_map_from_list, compute_ndcg, \
    compute_precision_recall_at_k
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk
from benchmark_src.utils import gather_results, framework
from benchmark_src.tasks import component_utils
import faiss
import numpy as np
import glob
from tqdm import tqdm
from ContextAwareJoin.src.myutils.utilities import load_dataframe, convert_to_dict_of_list, get_groundtruth_with_scores
from pathlib import Path


logger = logging.getLogger(__name__)

def load_benchmark_data(cfg):
    dataset_dir = str(Path(get_original_cwd()) / Path(cfg.benchmark_datasets_dir) / cfg.dataset_name)
    if cfg.dataset_name == 'opendata':
        file_format = '.df'
    else:
        file_format = '.csv'

    datalake_tables = glob.glob(f"{dataset_dir}/**/*{file_format}", recursive=True)
    # for column similarity tasks, groups of tables are in datalakes
    # maintain a mapping of datalake to tables
    datalake2tables = {}
    for table in tqdm(datalake_tables, desc="Computing Embeddings", file=logger):
        try:
            df = load_dataframe(table, file_format=file_format)
            if table not in datalake2tables:
                datalake2tables[table] = []
            datalake2tables[table].append(df)

    gt = glob.glob(f"{dataset_dir}/**/gt.*", recursive=True)

    if gt.endswith('.json'):
        gt_data = json.load(open(gt, 'r'))
    elif gt.endswith('.jsonl'):
        gt_data = convert_to_dict_of_list(gt)
    elif gt.endswith('.pickle'):
        raise NotImplementedError
    else:
        raise NotImplementedError

    return datalake2tables, gt_data


@monitor_resources()
def run_inference_based_on_column_embeddings(cluster_ranges, cfg):
    # get column embeddings and assert they have the correct format and shape
    # instantiate the embedding approach class

    results_file = "all_column_embeddings"

    if not os.path.exists(results_file):
        embedding_approach_class = framework.get_approach_class(cfg)
        embedder = embedding_approach_class(cfg)

        datalake2tables, gt_data = load_benchmark_data(cfg)
        metric_res = {}
        for datalake in datalake2tables:
            all_columns = {}
            if not os.path.exists(f'{results_file}/{datalake}.pkl'):

                logger.info(f"Starting to get column embeddings for datalake {datalake} columns")
                ## load the needed component
                column_embedding_component = embedder._load_component("column_embedding_component", "ColumnEmbeddingComponent",
                                                                   ColumnEmbeddingInterface)
                ## setup model
                _, resource_metrics_setup = component_utils.run_model_setup(component=column_embedding_component,
                                                                            input_table=None, dataset_information=None)

                for table in datalake2tables[datalake]:
                    all_columns[table] = {}
                    column_embeddings = column_embedding_component.create_column_embeddings_for_table(input_table=table)
                    for c in column_embeddings:
                        all_columns[table][c] = column_embeddings

                with open(f'{results_file}/{datalake}.pkl', "wb") as file:
                    pickle.dump(all_columns, file)
            else:
                with open(f'{results_file}/{datalake}.pkl', "rb") as file:
                    all_columns = pickle.load(file)
                    resource_metrics_setup = None


            all_cols = []
            all_indexes = {}
            i = 0
            for table in all_columns:
                for column in all_columns[table]:
                    all_indexes[(table, column)] = i
                    i += 1
                    all_cols.append(all_columns[table][column])
            arr = np.asarray(all_cols)
            index = faiss.IndexFlatL2(arr[0].size())
            index.add(arr)

            all_gt = []
            for mapping in gt_data:
                source_table = mapping['source']['filename']
                source_column = mapping['source']['col']
                source = all_indexes[(source_table, source_column)]

                for target in mapping['joinable_list']:
                    target_table = target['filename']
                    target_column = target['col']
                    target = all_indexes[(target_table, target_column)]
                    all_gt.append((source, target))


            k = 4  # TBD how do we set k from config
            search_sources = []
            for source, _ in all_gt:
                search_sources.append(arr[source])
            D, I = index.search(search_sources, k)

            result = []
            for x, i in enumerate(I):
                source_t, source_col = all_gt[i]
                joinable_list = []
                for y, neighbor in enumerate(I[i]):
                    target_t, target_col = all_gt[neighbor]
                    r = {"filename": target_t,
                         "col": f"{target_t}.{target_col}",
                         "score": float(D[x][y])}
                    joinable_list.append(r)
                result.append({"source": {"filename": source_t, "col": f"{source_t}.{source_col}"},
                          "joinable_list": joinable_list})


            dict = {}
            for line in result:
                json_line = json.loads(line)
                dict[json_line['source']['col']] = {i['col']: 1 for i in json_line['joinable_list']}
            result = dict

            result = convert_to_dict_of_list(result)
            gt_with_score = get_groundtruth_with_scores(gt_data)
            missing_queries =  set(gt_data.keys())-set(result.keys())
            logger.log(f"Result file is missing {len(missing_queries)} queries out of {len(gt_data)}")
            MRR = compute_mrr_from_list(gt_data, result, k)
            MAP = compute_map_from_list(gt_data, result, k)
            NDCG = compute_ndcg(gt_with_score, result, k=k)
            Precision, Recall  = compute_precision_recall_at_k(gt_data, result, k)
            metric_res[datalake] = {'MRR': MRR, 'MAP': MAP, 'NDCG': NDCG, 'Precision':Precision, 'Recall': Recall}

    return (metric_res, resource_metrics_setup)


def main(cfg: DictConfig):
    logger.debug(f"Started clustering")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True) 

    # run inference with model
    logger.info(f"Running clustering based on row embeddings")
    cluster_ranges =[1000]

    result, resource_metrics_task = run_inference_based_on_column_embeddings(cluster_ranges=cluster_ranges, cfg=cfg)
    result_metrics, resource_metrics_setup = result
    
    # save resource metrics to disk
    if resource_metrics_setup:
        save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)

    # save the other metrics to disk
    gather_results.save_results(cfg=cfg, metrics=result_metrics)

    

