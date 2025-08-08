from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sentence_transformers
import logging
import multiprocessing

from sklearn.metrics.pairwise import cosine_similarity

from benchmark_src.tasks import component_utils
from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface
from benchmark_src.utils import load_benchmark, benchmark_metrics, gather_results, framework
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk

logger = logging.getLogger(__name__)

def load_benchmark_data(cfg):
    dataset_folder = Path(get_original_cwd()) / Path(cfg.benchmark_datasets_dir) / cfg.dataset_name
    assert dataset_folder.exists(), f"Could not find path {dataset_folder}"

    with open(dataset_folder / "dataset_information.json", "r") as file:
        dataset_information = json.load(file)

    # load the input table of a datset and the testcase paths
    input_table = load_benchmark.get_input_table(dataset_folder)
    if cfg.test_case_limit:
        cfg.test_case_limit = int(cfg.test_case_limit)
    testcase_paths = load_benchmark.load_testcase_paths(dataset_path=dataset_folder, limit=cfg.test_case_limit)

    return input_table, testcase_paths, dataset_information

@monitor_resources()
def run_similarity_task_based_on_row_embeddings(row_embedding_component, input_table, testcase_paths, pk_column: str):
    logger.debug(f"Called run_similarity_task_based_on_row_embeddings")
    # get row embeddings and assert they have the correct format and shape
    row_embeddings = row_embedding_component.create_row_embeddings_for_table(input_table=input_table)
    component_utils.assert_row_embedding_format(row_embeddings=row_embeddings, input_table=input_table)

    all_results = []

    for testcase_path in tqdm(testcase_paths):
        similar_pair, dissimilar_pair = load_benchmark.load_testcase_more_similar_than(testcase_path)

        # find the necessary row embeddings of the three participating rows (a, b, c)
        row_a_index = input_table[input_table[pk_column] == similar_pair["a"]["qid"]].index[0]
        row_b_index = input_table[input_table[pk_column] == similar_pair["b"]["qid"]].index[0]
        row_c_index = input_table[input_table[pk_column] == dissimilar_pair["c"]["qid"]].index[0]

        row_a_embedding = row_embeddings[row_a_index]
        row_b_embedding = row_embeddings[row_b_index]
        row_c_embedding = row_embeddings[row_c_index]

        # compute cosine similarity between the similar pair and the dissimilar pair
        sim_ab = cosine_similarity(row_a_embedding.reshape(1, -1), row_b_embedding.reshape(1, -1))[0][0]
        sim_ac = cosine_similarity(row_a_embedding.reshape(1, -1), row_c_embedding.reshape(1, -1))[0][0]

        # result is 1 if the similarity is higher for the similar pair than for the dissimilar pair
        if sim_ab > sim_ac:
            all_results.append(True)
        else:
            all_results.append(False)
                
        if False:
            print("#"*100)
            print(f"Testcase ID: {testcase_id} - key column: {pk_column} - Input Row:")
            print(testcase_input_df)
            print("GT Output row:")
            print(testcase_gt_output_df)
            print("-"*100)
            print(result)

    return all_results


def main(cfg: DictConfig):
    logger.info(f"Started run_more_similar_than_benchmark")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True) 

    input_table, testcase_paths, dataset_information = load_benchmark_data(cfg)
    pk_column = dataset_information["primary_key_column"]

    # instantiate the embedding approach class
    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)

    ## load the row embedding component
    row_embedding_component = embedder._load_component("row_embedding_component", "RowEmbeddingComponent", RowEmbeddingInterface)

    ## setup model
    _, resource_metrics_setup = component_utils.run_model_setup(component=row_embedding_component, input_table=input_table, dataset_information=dataset_information)
    
    # run similarity search based on the row embeddings
    all_results, resource_metrics_task = run_similarity_task_based_on_row_embeddings(row_embedding_component=row_embedding_component, input_table=input_table, testcase_paths=testcase_paths, pk_column=pk_column)

    # save resource metrics to disk
    save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)


    # compute and save results
    accuracy = sum(all_results) / len(all_results)
    result_metrics = {"accuracy": accuracy}
    logger.info(f"Accuracy: {accuracy}")
    gather_results.save_results(cfg=cfg, metrics=result_metrics)
