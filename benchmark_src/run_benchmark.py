import hydra 
from omegaconf import DictConfig
from pathlib import Path
import logging
import sys
import traceback
from hydra.utils import get_original_cwd

from benchmark_src.utils.framework import register_resolvers

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig):
    print("#"*100)
    print("#"*100)
    print("#"*100)

    print("Got config: ")
    print(cfg)

    # convert paths in config to proper pathlib paths:
    cfg.benchmark_datasets_dir = Path(cfg.benchmark_datasets_dir)

    # create cache folder
    cfg.cache_dir = Path(get_original_cwd()) / Path(cfg.cache_dir)
    cfg.cache_dir.mkdir(exist_ok=True)
    dataset_cache_dir = cfg.cache_dir / "datasets"
    dataset_cache_dir.mkdir(exist_ok=True)
    models_cache_dir = cfg.cache_dir / "models"
    models_cache_dir.mkdir(exist_ok=True)
    logger.info(f"Cache directory at {cfg.cache_dir}")    


    # check if current directory already contains a result file, if yes do not need to run again
    current_directory = Path.cwd()
    if (current_directory / "results.json").is_file():
        logger.info(f'Already have a result file for this configuration in the given output directory.{current_directory / "results.json"}')
    else:
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
            else:
                logger.error("Unknown task:", cfg.task)
        except Exception as e:
            traceback.print_exc()
            logger.error("Error during benchmark run, please check traceback above.")
            sys.exit(1)

    
if __name__ == "__main__":
    print('Called run_benchmark.py')
    register_resolvers()
    main()