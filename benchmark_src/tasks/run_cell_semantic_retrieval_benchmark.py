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
import numpy as np

from benchmark_src.tasks import component_utils
from benchmark_src.dataset_creation.cell_datasets import cell_dataset_creation

from benchmark_src.approach_interfaces.cell_embedding_interface import CellEmbeddingInterface
from benchmark_src.utils import load_benchmark, benchmark_metrics, framework, result_utils
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk


logger = logging.getLogger(__name__)


def table_embeddings_to_dict(df, embeddings, paper_id, table_id):
    """
    Converts a table embedding matrix into a dict keyed by cell_uid
    """
    num_rows, num_cols = df.shape

    # ----------------- Validation -----------------
    if embeddings.ndim != 3:
        raise ValueError(f"Embeddings must be a 3D array, got ndim={embeddings.ndim}")

    if embeddings.shape[0] != num_rows + 1:
        raise ValueError(
            f"Embeddings row dimension mismatch: expected {num_rows + 1} (including header), "
            f"got {embeddings.shape[0]}"
        )

    if embeddings.shape[1] != num_cols:
        raise ValueError(
            f"Embeddings column dimension mismatch: expected {num_cols}, got {embeddings.shape[1]}"
        )

    # ----------------- Build dict -----------------
    cell_emb_dict = {}

    # --- Header row (row = 0) ---
    for c in range(num_cols):
        uid = f"{paper_id}/{table_id}/0/{c}"
        cell_emb_dict[uid] = embeddings[0, c]

    # --- Data rows (row >= 1) ---
    for r in range(num_rows):
        for c in range(num_cols):
            uid = f"{paper_id}/{table_id}/{r+1}/{c}"
            cell_emb_dict[uid] = embeddings[r + 1, c]

    return cell_emb_dict


def load_benchmark_data(cfg):
    # check if the dataset is in the cache
    dataset_folder = Path(get_original_cwd()) / Path(cfg.cache_dir) / "cell_level_data" / cfg.dataset_name

    # if it isn't, create the dataset
    needs_download = False
    if not dataset_folder.exists():
        logger.info(f"Need to create the dataset")
        needs_download = True

    # check if the testcases exist
    testcases_folder = dataset_folder / "testcases_retrieval_consistency"
    if not testcases_folder.exists():
        logger.info(f"Need to create the testcases")
        cell_dataset_creation.create_cell_dataset(cfg, needs_download)

    assert testcases_folder.exists(), f"Could not find path {testcases_folder}"

    # load the input tables and the testcase paths
    tables_folder = dataset_folder / "tables_csv"
    testcases = list(testcases_folder.glob("*.json"))   
    logger.info(f"Found {len(list(tables_folder.glob('*.csv')))} tables in {tables_folder} and {len(testcases)} testcases in {testcases_folder}")

    testcases = sorted(testcases)

    return tables_folder, testcases

@monitor_resources()
def run_cell_benchmark(cfg, cell_embedding_component, tables_folder: Path, testcases: list[Path]):
    
    embeddings_cache = {} # paperID / tableID as keys

    results = []
    
    # loop over all testcases
    for testcase_path in tqdm(testcases, desc="Evaluating testcases"):
        with open(testcase_path, "r") as f:
            testcase = json.load(f)
        
        input_tables_to_embed = [] # list of the dataframes
        input_table_infos = [] # list of the metadata

        ## load the input tables if they are not yet cached
        for table_file_name in testcase["tables"]:
            embeddings_in_cache = False
            if table_file_name in embeddings_cache:
                embeddings_in_cache = True

            if not embeddings_in_cache:
                table_path = tables_folder / table_file_name
                assert table_path.exists(), f"Could not find table at {table_path}"
                table_df = pd.read_csv(table_path)
                #print(table_df.head())
                input_tables_to_embed.append(table_df)
                input_table_infos.append({
                    "table_file_name": table_file_name,
                    "paper_id": table_file_name.split("_")[0],
                    # take everything after the first underscore
                    "table_id": "_".join(table_file_name.split("_")[1:])
                })

        ## get cell embeddings for all cells in the input tables (as 2d matrix, same shape as the table header+rows x columns)
        for idx, table_to_embed in enumerate(input_tables_to_embed):
            cell_embeddings_matrix = cell_embedding_component.create_cell_embeddings_for_table(input_table=table_to_embed)
            metadata = input_table_infos[idx]
            cell_embeddings_dict = table_embeddings_to_dict(table_to_embed, cell_embeddings_matrix, metadata["paper_id"], metadata["table_id"])
            embeddings_cache[metadata["table_file_name"]] = cell_embeddings_dict

        ## get get query cell from testcase and query cell embedding (create local cache for paper_id+table_id embeddings to re-use?)

        # -------------------- Get query cell embedding --------------------
        query_info = testcase["query"]
        query_table_file = f"{query_info['paper_id']}_{query_info['table_id']}"
        query_uid = f"{query_info['paper_id']}/{query_info['table_id']}/{query_info['row']}/{query_info['col']}"
        try:
            query_embedding = embeddings_cache[query_table_file][query_uid].reshape(1, -1)  # shape [1, dim]
        except KeyError as e:
            logger.error(f"Error in testcase {testcase_path} , looking for {query_table_file} in embeddings, need query cell {query_uid}")
            if query_table_file in embeddings_cache:
                logger.error(f"Have embeddings for the table")
                logger.error(embeddings_cache[query_table_file].keys())
            else:
                logger.error(f"Embeddings for the table complemetly missing..")
            
            logger.error(f"Newly embedded tables were {input_table_infos}")
            raise e


        ## get most similar embeddings to query cell embedding (how to get them from 2d matrix??)

        ## check that the most similar embeddings are the same as in the gt (calculate top_k accuracy = x out of the k nearest neighbors have the same type)

         # -------------------- Prepare candidate embeddings --------------------
        # Flatten all embeddings from all involved tables
        candidate_uids = []
        candidate_embeddings = []

        for table_file in testcase["tables"]:
            for uid, emb in embeddings_cache[table_file].items():
                # exclude the query cell itself from candidates
                if uid != query_uid:
                    candidate_uids.append(uid)
                    candidate_embeddings.append(emb)

        candidate_embeddings = np.stack(candidate_embeddings, axis=0)  # shape [num_candidates, dim]

        # -------------------- Compute cosine similarity --------------------
        sims = cosine_similarity(query_embedding, candidate_embeddings)[0]  # shape [num_candidates]

        # -------------------- Get top-k nearest neighbors --------------------
        top_k = testcase["top_k"]
        topk_indices = np.argsort(sims)[::-1][:top_k]
        topk_uids = [candidate_uids[i] for i in topk_indices]

        # -------------------- Compute accuracy against ground truth --------------------
        ground_truth_uids = set(
            f"{cell['paper_id']}/{cell['table_id']}/{cell['row']}/{cell['col']}"
            for cell in testcase["ground_truth"]
        )

        correct = sum([1 for uid in topk_uids if uid in ground_truth_uids])
        accuracy = correct / top_k if top_k > 0 else 0.0

        results.append({
            "testcase_file": str(testcase_path),
            "top_k": top_k,
            "correct": correct,
            "accuracy": accuracy
        })

    return results

def main(cfg: DictConfig):
    logger.info(f"Started run_row_similarity_benchmark")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True) 

    # load the data
    tables_folder, testcases = load_benchmark_data(cfg)

    # instantiate the embedding approach class
    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)

    # ## load the needed component
    cell_embedding_component = embedder._load_component("cell_embedding_component", "CellEmbeddingComponent", CellEmbeddingInterface)

    # ## setup model
    _, resource_metrics_setup = component_utils.run_model_setup(component=cell_embedding_component, dataset_information=None)
    
    # # run similarity search based on the row embeddings
    # all_positions, resource_metrics_task = run_similarity_search_based_on_cell_embeddings(row_embedding_component=row_embedding_component, input_table=input_table, testcase_paths=testcase_paths, pk_column=pk_column, k=cfg.task.top_k)
    all_results, resource_metrics_task = run_cell_benchmark(cfg, cell_embedding_component, tables_folder, testcases)

    # save resource metrics to disk
    save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)

    # -------------------- Summary --------------------
    overall_acc = np.mean([r["accuracy"] for r in all_results])
    logger.info(f"accuracy = (# of correct cells in top-k) / k")
    logger.info(f"Mean top-k accuracy over {len(all_results)} testcases: {overall_acc:.3f}")


    # compute and save results
    result_metrics = {
        "accuracy": overall_acc
    }

    result_utils.save_results(cfg=cfg, metrics=result_metrics)

    # save all_results also as JSON
    with open("results_per_testcase.json", "w") as file:
        json.dump(all_results, file, indent=2)
