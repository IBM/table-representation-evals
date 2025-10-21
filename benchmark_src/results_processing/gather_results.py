import hydra
from omegaconf import DictConfig
from pathlib import Path
import logging
import attrs
from hydra.core.config_store import ConfigStore

from benchmark_src.results_processing import aggregate, resources

logger = logging.getLogger(__name__)

@attrs.define
class ResultsConfig:
    results_folder_name: str = "results"

ConfigStore.instance().store(name="results_config", node=ResultsConfig)

@hydra.main(version_base=None, config_name="results_config")
def main(cfg: DictConfig) -> None:
    assert cfg.results_folder_name != "", f"Error in gathering results: Please enter the foldername of the results folder"
    results_folder = Path(cfg.results_folder_name)
    print(f"Gathering based on results folder: *{results_folder}*")
    assert results_folder.exists(), f"Could not find results folder at {results_folder}"

    detailed_results_folder = results_folder / "results_per_task"
    detailed_results_folder = Path(detailed_results_folder)
    detailed_results_folder.mkdir(parents=True, exist_ok=True)    

    print(f"Calling gather_results")
    aggregate.gather_results(results_folder, detailed_results_folder)

    print(f"Calling gather_resources")
    resources.gather_resources(results_folder, detailed_results_folder)


if __name__ == "__main__":
    main()