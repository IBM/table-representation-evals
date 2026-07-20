import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sentence_transformers
import logging
import multiprocessing
import numpy as np
import torch
from omegaconf import DictConfig

from benchmark_src.tasks import component_utils
from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface
from benchmark_src.utils import load_benchmark, benchmark_metrics, framework, result_utils
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk

logger = logging.getLogger(__name__)

def load_datalake_infos(cfg):
    dataset_folder = Path(cfg.cache_dir) / "datasets" / "table_retrieval" / "gitTables"
    datalake_folders = sorted(x for x in dataset_folder.iterdir() if x.is_dir())

    test_case_limit = getattr(cfg.task, "test_case_limit", None)
    if test_case_limit:
        test_case_limit = int(test_case_limit)
        datalake_folders = datalake_folders[:test_case_limit]
        logger.info(f"test_case_limit={test_case_limit}: using {len(datalake_folders)} datalake folders")

    return datalake_folders

@monitor_resources()
def run_task(datalake_folders, table_embedding_component):
    recall_scores = []
    mrr_scores = []
    map_scores = []
    results_per_datalake = {}

    # get table embeddings and assert they have the correct format and shape
    for datalake_folder in datalake_folders:
        logger.info(f"Processing: {datalake_folder}")

        # load table paths and testcases
        datalake_tables = [x for x in (datalake_folder / "tables").iterdir() if x.suffix == ".csv"]
        testcase_paths = [x for x in (datalake_folder / "testcases").iterdir() if x.suffix == ".json"]

        # get embeddings for each table in the datalake
        all_table_embeddings = {} # map table filename to embedding

        logger.info(f"Creating embeddings for {len(datalake_tables)} tables.")
        for table_path in tqdm(datalake_tables, desc=f"Embedding tables for {datalake_folder.name}"):
            # read the pandas table
            input_table = pd.read_csv(table_path)

            # parameter whether to use just the schema?
            

            assert len(input_table) > 0
            assert len(input_table.columns) > 1

            # get table embedding
            table_embedding = table_embedding_component.create_table_embedding(input_table)

            # verify the shape (should be a single embedding)
            component_utils.assert_table_embedding_format(table_embedding)

            # save embedding in mapping
            all_table_embeddings[table_path.name] = table_embedding 

        # stack embeddings into matrix for semantic_search
        table_names = sorted(list(all_table_embeddings.keys()))

        # ensure all embeddings are 1D before stacking
        embedding_list = []
        for name in table_names:
            emb = all_table_embeddings[name]

            # if shape is (1, dim) -> squeeze to (dim,)
            if emb.ndim == 2:
                emb = emb.squeeze(0)

            embedding_list.append(emb)

        datalake_embeddings = np.vstack(embedding_list)  # shape: (num_tables, dim)
        datalake_embeddings = sentence_transformers.util.normalize_embeddings(torch.tensor(datalake_embeddings))

        # optional safety check
        assert datalake_embeddings.ndim == 2
        assert datalake_embeddings.shape[0] == len(table_names)

        logger.info(
            f"Datalake embedding matrix shape: {datalake_embeddings.shape}"
        )

        # Initialize metrics storage
        recall_at_k_list = []
        mrr_list = []
        average_precision_list = []

        for testcase_path in testcase_paths:
            # load testcase json
            with open(testcase_path, "r") as file:
                testcase_data = json.load(file)

            # get query table embedding
            query_table = testcase_data["query"]
            assert query_table in all_table_embeddings, f"{query_table} missing"
            query_table_embedding = all_table_embeddings[query_table]
            if query_table_embedding.ndim == 1:
                query_table_embedding = query_table_embedding.reshape(1, -1)
            query_table_embedding = sentence_transformers.util.normalize_embeddings(torch.tensor(query_table_embedding))

            # get gt similar tables
            gt_tables = testcase_data["positives"] # is a list of filenames
            k = len(gt_tables)

            # run similarity search
            hits = sentence_transformers.util.semantic_search(
                query_embeddings=query_table_embedding,
                corpus_embeddings=datalake_embeddings,
                top_k=k + 1  # +1 because most similar is itself
            )[0]

            # map indices back to filenames
            retrieved_tables = [
                table_names[hit["corpus_id"]]
                for hit in hits
                if table_names[hit["corpus_id"]] != query_table
            ][:k]

            # logger.debug(f"Query: {query_table}")
            # logger.debug(f"Retrieved: {retrieved_tables}")
            # logger.debug(f"Ground Truth: {gt_tables}")


            # compute Recall@k
            num_correct = len(set(retrieved_tables) & set(gt_tables))
            recall_at_k = num_correct / len(gt_tables)
            recall_at_k_list.append(recall_at_k)

            # compute MRR
            mrr = 0.0
            for rank, table_name in enumerate(retrieved_tables, start=1):
                if table_name in gt_tables:
                    mrr = 1.0 / rank
                    break
            mrr_list.append(mrr)

            # ---- Average Precision (AP) ----
            num_hits = 0
            ap = 0.0
            for rank, table_name in enumerate(retrieved_tables, start=1):
                if table_name in gt_tables:
                    num_hits += 1
                    precision_at_rank = num_hits / rank
                    ap += precision_at_rank
            ap /= len(gt_tables)  # average precision for this query

            average_precision_list.append(ap)


        # aggregate metrics for this datalake
        mean_recall = sum(recall_at_k_list) / len(recall_at_k_list) if recall_at_k_list else 0.0
        mean_mrr = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0
        map_score = sum(average_precision_list) / len(average_precision_list) if average_precision_list else 0.0

        logger.info(f"Datalake {datalake_folder.name} evaluation:")
        logger.info(f"  Mean Recall@k: {mean_recall:.4f}")
        logger.info(f"  Mean MRR: {mean_mrr:.4f}")
        logger.info(f"  MAP: {map_score:.4f}")

        recall_scores.append(mean_recall)
        mrr_scores.append(mean_mrr) 
        map_scores.append(map_score)

        results_per_datalake[datalake_folder.name] = {
            "Mean Recall@k": mean_recall,
            "Mean MRR": mean_mrr,
            "MAP": map_score
        }

    # aggregate overall metrics across datalakes
    overall_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    overall_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    overall_map = sum(map_scores) / len(map_scores) if map_scores else 0.0

    results = {
        "Recall": overall_recall,
        "MRR": overall_mrr,
        "MAP": overall_map
    }

    return results, results_per_datalake

def main(cfg: DictConfig):
    logger.info(f"Started run_table_retrieval_gittables")

    # instantiate the embedding approach class
    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)
    table_embedding_component = embedder._load_component("table_embedding_component", "TableEmbeddingComponent", TableEmbeddingInterface)

    # get the data (first get a list of all datalakes, then per datalake embed each table)
    datalake_folders = load_datalake_infos(cfg)
    logger.info(f"Found {len(datalake_folders)} datalake folders")

    ## setup model
    _, resource_metrics_setup = component_utils.run_model_setup(component=table_embedding_component)

    # run task
    results, resource_metrics_task = run_task(datalake_folders=datalake_folders, table_embedding_component=table_embedding_component)
    result_metrics, results_per_datalake = results

    # save resource metrics
    save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)

    # save performance metrics to disk
    result_utils.save_results(cfg=cfg, metrics=result_metrics)

    with open(Path(cfg.output_dir) / "results_per_datalake.json", "w") as file:
            json.dump(results_per_datalake, file, indent=2)

    print(f"Finished run_table_retrieval_gittables with results: {result_metrics}")