from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sentence_transformers
import logging
import multiprocessing

from benchmark_src.tasks import component_utils
from benchmark_src.approach_interfaces.row_similarity_task_interface import RowSimilaritySearchInterface
from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface
from benchmark_src.utils import load_benchmark, benchmark_metrics, framework, result_utils
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk

logger = logging.getLogger(__name__)

def load_benchmark_data(cfg):
    dataset_folder = Path(get_original_cwd()) / Path(cfg.cache_dir) / "row_similarity_data_full" / cfg.dataset_name
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
def run_similarity_search_based_on_row_embeddings(row_embedding_component, input_table, testcase_paths, pk_column: str, k: int):
    logger.debug(f"Called run_similarity_search_based_on_row_embeddings")
    # get row embeddings and assert they have the correct format and shape
    row_embeddings = row_embedding_component.create_row_embeddings_for_table(input_table=input_table)
    component_utils.assert_row_embedding_format(row_embeddings=row_embeddings, input_table=input_table)

    all_positions = []
    for testcase_path in tqdm(testcase_paths):
        testcase_id, testcase_input_df, testcase_gt_output_df = load_benchmark.load_testcase(testcase_path)

        # get embedding of input row by finding the input row in the original table
        testcase_row_pk_value = testcase_input_df[pk_column].item()
        testcase_row_idx_in_table = input_table[input_table[pk_column] == testcase_row_pk_value].index[0]
        input_row_embedding = row_embeddings[testcase_row_idx_in_table]
        
        # run similarity search
        result = sentence_transformers.util.semantic_search(query_embeddings=input_row_embedding, corpus_embeddings=row_embeddings, top_k=k+1) # +1 because most similar is row itself
        result_tuples = result[0] # one list for every query, just have one query
        
        if False:
            print("#"*100)
            print(f"Testcase ID: {testcase_id} - key column: {pk_column} - Input Row:")
            print(testcase_input_df)
            print("GT Output row:")
            print(testcase_gt_output_df)
            print("-"*100)
            print(result)

        # get correct entry id from input dataframe at position of the row
        predicted_row_ids = []
        for result_dict in result_tuples:
            corpus_id = result_dict["corpus_id"]
            if corpus_id != testcase_row_idx_in_table:
                orig_row_id = input_table[pk_column].iloc[corpus_id]
                predicted_row_ids.append(orig_row_id)

        position_list = benchmark_metrics.get_position_of_gt(gt_row_ids=list(testcase_gt_output_df[pk_column]), predicted_row_ids=predicted_row_ids)
        all_positions += position_list

    return all_positions

@monitor_resources()
def run_similarity_search_custom_approach(similarity_search_component, testcase_paths: list, input_table: pd.DataFrame, pk_column, k: int):
    logger.debug(f"Called run_similarity_search_component")
    all_positions = []
    for testcase_path in tqdm(testcase_paths):
        testcase_id, testcase_input_df, testcase_gt_output_df = load_benchmark.load_testcase(testcase_path)

        # get ranked list of top-k most similar rows
        try:
            ranked_list = similarity_search_component.custom_row_similarity_search(input_table=input_table, input_row=testcase_input_df, k=k)
            # TODO: assert that ranked_list has correct format and length
        except Exception as e:
            print(f"Exception occured when running row similarity search of approach: {e}")
            ranked_list = []

        position_list = benchmark_metrics.get_position_of_gt(gt_row_ids=list(testcase_gt_output_df[pk_column]), predicted_row_ids=ranked_list)
        all_positions += position_list

    return all_positions

def main(cfg: DictConfig):
    logger.info(f"Started run_row_similarity_benchmark")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True) 

    input_table, testcase_paths, dataset_information = load_benchmark_data(cfg)
    pk_column = dataset_information["primary_key_column"]

    # instantiate the embedding approach class
    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)

    run_row_similarity_search_based_on = cfg.benchmark_tasks.row_similarity_search.task_parameters.run_similarity_search_based_on
    if run_row_similarity_search_based_on == "row_embeddings":
        ## load the needed component
        row_embedding_component = embedder._load_component("row_embedding_component", "RowEmbeddingComponent", RowEmbeddingInterface)

        ## setup model
        _, resource_metrics_setup = component_utils.run_model_setup(component=row_embedding_component, input_table=input_table, dataset_information=dataset_information)
        
        # run similarity search based on the row embeddings
        all_positions, resource_metrics_task = run_similarity_search_based_on_row_embeddings(row_embedding_component=row_embedding_component, input_table=input_table, testcase_paths=testcase_paths, pk_column=pk_column, k=cfg.task.top_k)
        
    elif run_row_similarity_search_based_on == "custom_function":
        ## load the needed component
        similarity_search_component = embedder._load_component("row_similarity_search_component", "RowSimilaritySearchComponent", RowSimilaritySearchInterface)

        ## setup model for task
        _, resource_metrics_setup = component_utils.run_model_setup(component=similarity_search_component, input_table=input_table, dataset_information=dataset_information)

        ## run the task
        all_positions, resource_metrics_task = run_similarity_search_custom_approach(similarity_search_component=similarity_search_component, testcase_paths=testcase_paths, input_table=input_table,pk_column=pk_column, k=cfg.task.top_k)
    else:
        logger.error(f"Got unsupported value for 'run_row_similarity_search_based_on' parameter: {run_row_similarity_search_based_on}")
        raise ValueError


    # save resource metrics to disk
    save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)

    # compute and save results
    result_metrics = benchmark_metrics.compute_all_metrics(all_positions)
    result_utils.save_results(cfg=cfg, metrics=result_metrics)
