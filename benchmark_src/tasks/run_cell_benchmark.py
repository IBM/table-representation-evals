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
from benchmark_src.dataset_creation.cell_datasets import cell_dataset_creation

logger = logging.getLogger(__name__)

def load_benchmark_data(cfg):
    # check if the dataset is in the cache
    dataset_folder = Path(get_original_cwd()) / Path(cfg.cache_dir) / "cell_level_data" / cfg.dataset_name

    # if it isn't, create the dataset
    #if not dataset_folder.exists():
    if True:
        logger.info(f"Need to create the dataset")
        cell_dataset_creation.create_cell_dataset(cfg)

    assert dataset_folder.exists(), f"Could not find path {dataset_folder}"

    # load the input tables and the testcase paths

    return


def main(cfg: DictConfig):
    logger.info(f"Started run_row_similarity_benchmark")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True) 

    # load the data
    # TODO
    load_benchmark_data(cfg)

    # # instantiate the embedding approach class
    # embedding_approach_class = framework.get_approach_class(cfg)
    # embedder = embedding_approach_class(cfg)

    # ## load the needed component
    # row_embedding_component = embedder._load_component("cell_embedding_component", "CellEmbeddingComponent", CellEmbeddingInterface)

    # ## setup model
    # _, resource_metrics_setup = component_utils.run_model_setup(component=row_embedding_component, input_table=input_table, dataset_information=dataset_information)
    
    # # run similarity search based on the row embeddings
    # all_positions, resource_metrics_task = run_similarity_search_based_on_cell_embeddings(row_embedding_component=row_embedding_component, input_table=input_table, testcase_paths=testcase_paths, pk_column=pk_column, k=cfg.task.top_k)
    