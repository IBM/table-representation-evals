from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import logging
import multiprocessing

from sklearn.metrics.pairwise import cosine_similarity

from benchmark_src.tasks import component_utils
from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface
from benchmark_src.utils import load_benchmark, framework, result_utils
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk
from benchmark_src.dataset_creation.wikidata_books import create_books_dataset
from benchmark_src.dataset_creation.create_variations import create_dataset_variations

logger = logging.getLogger(__name__)

def load_benchmark_data(cfg):
    # TODO: check if data is already in cache, if not, generate it
    
    ################### get dataset name and information on variations #####################
    if "@@" in cfg.dataset_name: 
        dataset_info = str(cfg.dataset_name).split("@@")
        dataset_name = dataset_info[0]
        variation_string = dataset_info[1]
        logger.info(f"Variation is: {variation_string}")
        variation_info = variation_string.split("---")
        logger.info(f"Have {len(variation_info)} variation parameters: {variation_info}")
        variation_parameters = {s[0]: s[1] for x in variation_info for s in [x.split("::")]}
    else:
        dataset_name = cfg.dataset_name

    base_dataset_folder = Path(get_original_cwd()) / Path(cfg.cache_dir) / "datasets" / "more_similar_than" / dataset_name
    if dataset_name == "wikidata_books":
        ################### check if the base dataset exists #####################
        if not base_dataset_folder.exists():
            create_books_dataset.create_books_dataset(cfg=cfg, dataset_save_dir=base_dataset_folder)
            logger.info('Done creating wikidata books dataset')
        else:
            logger.info('Wikidata books dataset found in cache')
        ################### check if selected variation exists #####################
        if variation_info == ['column_naming::original']:
            logger.info(f"Running with base dataset")
            variation_dataset_folder = base_dataset_folder / "original"
        else:
            variation_dataset_folder = base_dataset_folder / variation_string
            if not variation_dataset_folder.exists():
                logger.info(f"Have to create variation data")
                # TODO: load base dataset input table and dataset information and hand it over
                with open(base_dataset_folder /"original" / "dataset_information.json", "r") as file:
                    base_dataset_information = json.load(file)
                base_input_table = load_benchmark.get_input_table(base_dataset_folder / "original")    
                print(variation_info, type(variation_info))
                # TODO: create variations
                create_dataset_variations(variation_parameters=variation_parameters,
                                           dataset_df=base_input_table, 
                                           dataset_info=base_dataset_information, 
                                           save_dir=variation_dataset_folder,
                                           variation_name=variation_string)


    assert variation_dataset_folder.exists(), f"Could not find path {variation_dataset_folder}"

    # get folder of the exact variation

    with open(variation_dataset_folder / "dataset_information.json", "r") as file:
        dataset_information = json.load(file)

    # load the input table of a datset and the testcase paths
    input_table = load_benchmark.get_input_table(variation_dataset_folder, verbose=True)
    if cfg.test_case_limit:
        cfg.test_case_limit = int(cfg.test_case_limit)
    testcase_paths = load_benchmark.load_testcase_paths(dataset_path=base_dataset_folder, limit=cfg.test_case_limit)

    return input_table, testcase_paths, dataset_information

@monitor_resources()
def run_similarity_task_based_on_row_embeddings(row_embedding_component, input_table, testcase_paths, pk_column: str):
    logger.debug(f"Called run_similarity_task_based_on_row_embeddings")
    # get row embeddings and assert they have the correct format and shape
    row_embeddings = row_embedding_component.create_row_embeddings_for_table(input_table=input_table)
    component_utils.assert_row_embedding_format(row_embeddings=row_embeddings, input_table=input_table)

    all_results = []

    for testcase_path in tqdm(testcase_paths):
        testcase_dict = load_benchmark.load_testcase_more_similar_than(testcase_path)

        similar_pair = testcase_dict["similar_pair"]
        dissimilar_pair = testcase_dict["dissimilar_pair"]

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
        results_info = {'id': int(testcase_path.stem), 'difficulty': testcase_dict['difficulty'], 'category': testcase_dict['category']}
        if sim_ab > sim_ac:
            results_info['solved'] = True
        else:
            results_info['solved'] = False
        all_results.append(results_info)

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
    results_df = pd.DataFrame(all_results).set_index('id')
    results_df.sort_index(inplace=True)
    results_df.to_csv('results_per_datapoint.csv')
    overall_accuracy = float(results_df['solved'].mean())
    accuracy_by_difficulty = {
        f"accuracy_{diff}": acc 
        for diff, acc in results_df.groupby('difficulty')['solved'].mean().to_dict().items()
    }
    result_metrics = {"accuracy": overall_accuracy}
    result_metrics.update(accuracy_by_difficulty)
    logger.info(result_metrics)
    result_utils.save_results(cfg=cfg, metrics=result_metrics)
