import logging
import sys
import traceback
from pathlib import Path

from benchmark_src.utils.framework import StreamToLogger
from benchmark_src.utils.cfg_utils import guard_cfg_no_none

logger = logging.getLogger(__name__)


def run_single(cfg):
    """
    Dispatch a single (approach, task, dataset) run.
    cfg is an OmegaConf DictConfig with fields:
      task, approach, dataset_name, output_dir, cache_dir, project_root, test_case_limit
    Called by run_experiments.py for each job.
    """
    sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)

    logger.info("#" * 80)
    logger.info(
        f"Task: {cfg.task.task_name}  |  Approach: {cfg.approach.approach_name}"
        f"  |  Dataset: {cfg.dataset_name}"
    )

    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "datasets").mkdir(exist_ok=True)
    (cache_dir / "models").mkdir(exist_ok=True)

    guard_cfg_no_none(cfg)

    output_dir = Path(cfg.output_dir)

    if (output_dir / "results.json").is_file():
        logger.info(f"Already have results.json at {output_dir} — skipping")
        return

    try:
        if cfg.task.task_name == "row_similarity_search":
            from benchmark_src.tasks import run_row_similarity_benchmark
            run_row_similarity_benchmark.main(cfg)
        elif cfg.task.task_name == "predictive_ml":
            from benchmark_src.tasks import run_predictive_ml_benchmark
            run_predictive_ml_benchmark.main(cfg)
        elif cfg.task.task_name == "more_similar_than":
            from benchmark_src.tasks import run_more_similar_than_benchmark
            run_more_similar_than_benchmark.main(cfg)
        elif cfg.task.task_name == "clustering":
            from benchmark_src.tasks import run_clustering_benchmark
            run_clustering_benchmark.main(cfg)
        elif cfg.task.task_name == "column_similarity_search":
            from benchmark_src.tasks import run_column_similarity_benchmark
            run_column_similarity_benchmark.main(cfg)
        elif cfg.task.task_name == "column_type_annotation":
            from benchmark_src.tasks import run_column_type_annotation_benchmark
            run_column_type_annotation_benchmark.main(cfg)
        elif cfg.task.task_name == "table_retrieval":
            if cfg.dataset_name == "gitTables":
                from benchmark_src.tasks import run_table_retrieval_gittables
                run_table_retrieval_gittables.main(cfg)
            else:
                from benchmark_src.tasks import run_table_retrieval_benchmark
                run_table_retrieval_benchmark.main(cfg)
        elif cfg.task.task_name == "table_shuffling":
            from benchmark_src.tasks import run_table_shuffling_benchmark
            run_table_shuffling_benchmark.main(cfg)
        elif cfg.task.task_name == "table_type_detection":
            from benchmark_src.tasks import run_table_type_detection_benchmark
            run_table_type_detection_benchmark.main(cfg)
        elif cfg.task.task_name == "cell_task":
            from benchmark_src.tasks import run_cell_semantic_retrieval_benchmark
            run_cell_semantic_retrieval_benchmark.main(cfg)
        elif cfg.task.task_name == "nl2column_mapping":
            from benchmark_src.tasks import run_NL2column_mapping_benchmark
            run_NL2column_mapping_benchmark.main(cfg)
        elif cfg.task.task_name in ("cell_to_column_mapping", "nl2cell2column_mapping"):
            from benchmark_src.tasks import run_NL2cell2column_mapping_benchmark
            run_NL2cell2column_mapping_benchmark.main(cfg)
        else:
            logger.error(f"Unknown task: {cfg.task.task_name}")
    except Exception as e:
        traceback.print_exc()
        logger.error("Error during benchmark run, please check traceback above.")
        raise
