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
from benchmark_src.utils import framework, result_utils
from benchmark_src.tasks import component_utils
import faiss
import numpy as np
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

def load_benchmark_data(cfg):
    dataset_name = str(Path(get_original_cwd()) / "ContextAwareJoin" / "datasets" / cfg.dataset_name) + '.csv'
    input_table = pandas.read_csv(dataset_name, engine='c', on_bad_lines='skip')
    return input_table


@monitor_resources()
def run_inference_based_on_row_embeddings(cluster_ranges, cfg):
    # get row embeddings and assert they have the correct format and shape
    # instantiate the embedding approach class

    row_file = "row_embeddings.pkl"
    if not os.path.exists(row_file):
        embedding_approach_class = framework.get_approach_class(cfg)
        embedder = embedding_approach_class(cfg)

        test_table = load_benchmark_data(cfg)
        logger.info(f"Starting to get row embeddings for the {len(test_table)} rows")

        logger.info(f"Loaded the clustering data")

        ## load the needed component
        row_embedding_component = embedder._load_component("row_embedding_component", "RowEmbeddingComponent",
                                                           RowEmbeddingInterface)

        ## setup model
        _, resource_metrics_setup = component_utils.run_model_setup(component=row_embedding_component,
                                                                    input_table=test_table, dataset_information=None)

        test_row_embeddings = row_embedding_component.create_row_embeddings_for_table(input_table=test_table)
        component_utils.assert_row_embedding_format(row_embeddings=test_row_embeddings, input_table=test_table)

        with open("row_embeddings.pkl", "wb") as file:
            pickle.dump(test_row_embeddings, file)
    else:
        print('re-using row embeddings')
        with open(row_file, "rb") as file:
            test_row_embeddings = pickle.load(file)
        resource_metrics_setup = None

    X_test = test_row_embeddings
    logger.info(f"Starting the clustering")

    niter = 20
    verbose = True
    d = len(X_test[0])

    scores = {}

    for k in cluster_ranges:
        print('starting with k=', k)
        kmeans = faiss.Kmeans(d, k, niter=niter, verbose=verbose)
        x = np.asarray(X_test)
        kmeans.train(x)
        _, I = kmeans.index.search(x, 1)
        cluster_labels = I.flatten().tolist()
        scores[k] = (silhouette_score(x, cluster_labels, metric='euclidean', sample_size=1000), cluster_labels)

    best_k_val = -1
    best_k = -1
    for k in scores:
        if scores[k][0] > best_k:
            best_k_val = scores[k][0]
            best_k = k


    logger.info(f"Computing the metrics")
    metric_res = {}
    metric_res['silhouette'] = best_k_val
    metric_res['calinski_harabasz'] = float(sklearn.metrics.calinski_harabasz_score(X_test, scores[best_k][1]))
    metric_res['davies_bouldin'] = float(sklearn.metrics.davies_bouldin_score(X_test, scores[best_k][1]))
    metric_res['pred_labels'] = scores[best_k][1]
    return (metric_res, resource_metrics_setup)


def main(cfg: DictConfig):
    logger.debug(f"Started clustering")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True) 

    # run inference with model
    logger.info(f"Running clustering based on row embeddings")
    cluster_ranges =[1000]
#    result_metrics, resource_metrics_task, resource_metrics_setup = run_inference_based_on_row_embeddings(cluster_ranges=cluster_ranges, cfg=cfg)

    result, resource_metrics_task = run_inference_based_on_row_embeddings(cluster_ranges=cluster_ranges, cfg=cfg)
    result_metrics, resource_metrics_setup = result
    
    # save resource metrics to disk
    if resource_metrics_setup:
        save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)


    # save results to disk
    # save prediction labels to a seperate file:
    with open("clustering_hdb_labels.json", "w") as file:
        json.dump(result_metrics["pred_labels"], file)
    # remove prediction labels from result metrics
    del result_metrics["pred_labels"]
    # save the other metrics to disk
    result_utils.save_results(cfg=cfg, metrics=result_metrics)

    

