from omegaconf import DictConfig
import multiprocessing
import logging
from pathlib import Path
import sklearn
import pandas
from hydra.utils import get_original_cwd
import json
import pickle
import os

from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk
from benchmark_src.utils import gather_results, framework
from benchmark_src.tasks import component_utils
import hdbscan

logger = logging.getLogger(__name__)


def load_benchmark_data(cfg):
    dataset_name = str(Path(get_original_cwd()) / Path(cfg.benchmark_datasets_dir) / cfg.dataset_name) + '.csv'
    input_table = pandas.read_csv(dataset_name, engine='c', on_bad_lines='skip')
    return input_table


@monitor_resources()
def run_inference_based_on_row_embeddings(min_clusters, row_embedding_component, test_table):
    # get row embeddings and assert they have the correct format and shape
    logger.info(f"Starting to get row embeddings for the {len(test_table)} rows")
    row_file = "row_embeddings.pkl"
    if not os.path.exists(row_file):
        print('creating row embeddings')
        test_row_embeddings = row_embedding_component.create_row_embeddings_for_table(input_table=test_table)
        with open("row_embeddings.pkl", "wb") as file:
            pickle.dump(test_row_embeddings, file)
    else:
        print('re-using row embeddings')
        with open(row_file, "rb") as file:
            test_row_embeddings = pickle.load(file)

    component_utils.assert_row_embedding_format(row_embeddings=test_row_embeddings, input_table=test_table)

    X_test = test_row_embeddings
    logger.info(f"Starting the clustering")
    #hdb = sklearn.cluster.HDBSCAN(min_cluster_size=min_clusters)
    hdb = hdbscan.HDBSCAN()
    hdb.fit(X_test)

    logger.info(f"Computing the metrics")
    metric_res = {}
    metric_res['silhouette'] = float(sklearn.metrics.silhouette_score(X_test, hdb.labels_, metric='euclidean'))
    metric_res['calinski_harabasz'] = float(sklearn.metrics.calinski_harabasz_score(X_test, hdb.labels_))
    metric_res['davies_bouldin'] = float(sklearn.metrics.davies_bouldin_score(X_test, hdb.labels_))
    metric_res['pred_labels'] = hdb.labels_.tolist()
    return metric_res


def main(cfg: DictConfig):
    logger.debug(f"Started clustering")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True) 

    # instantiate the embedding approach class
    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)

    test_table = load_benchmark_data(cfg)
    logger.info(f"Loaded the clustering data")

    ## load the needed component
    row_embedding_component = embedder._load_component("row_embedding_component", "RowEmbeddingComponent", RowEmbeddingInterface)

    ## setup model
    _, resource_metrics_setup = component_utils.run_model_setup(component=row_embedding_component, input_table=test_table, dataset_information=None)

    # run inference with model
    logger.info(f"Running clustering based on row embeddings")
    result_metrics, resource_metrics_task = run_inference_based_on_row_embeddings(row_embedding_component=row_embedding_component, test_table=test_table, min_clusters=20)

    # save resource metrics to disk
    save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)

    # save results to disk
    # save prediction labels to a seperate file:
    with open("clustering_hdb_labels.json", "w") as file:
        json.dump(result_metrics["pred_labels"], file)
    # remove prediction labels from result metrics
    del result_metrics["pred_labels"]
    # save the other metrics to disk
    gather_results.save_results(cfg=cfg, metrics=result_metrics)

    

