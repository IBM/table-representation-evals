import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from datasets import Dataset
from datasets import load_dataset
from huggingface_hub import snapshot_download
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    name: str
    corpus: Dataset
    queries: Dataset


def load_with_dataset(hf_dataset_path: str, split: str) -> Dataset:
    dataset = load_dataset(path=hf_dataset_path, split=split)
    return dataset


# Convert each table cells' list content to strings to avoid type conflicts
def convert_dict_to_dataset(corpus_dict: Dict[str, Any]) -> Dataset:
    fixed_corpus_dict = {}
    for key, values in corpus_dict.items():
        if isinstance(values, list):
            fixed_corpus_dict[key] = [str(v) if v is not None else None for v in values]
        else:
            fixed_corpus_dict[key] = values

    return Dataset.from_dict(fixed_corpus_dict)


def load_with_snapshot(corpus_hf_path: str, json_path: str) -> Dataset:
    path_to_data_dir = snapshot_download(repo_id=corpus_hf_path, repo_type="dataset")
    logger.info(f"Loading corpus from {path_to_data_dir}")
    path_to_dataset = Path(path_to_data_dir, json_path)

    with open(path_to_dataset, "r") as file:
        corpus_dict = json.load(file)

    return convert_dict_to_dataset(corpus_dict)


def load_single_dataset(dataset_name: str, dataset_config: DictConfig) -> DatasetBundle:
    logger.info(f"Loading queries for {dataset_name} using datasets library...")
    queries_data = load_with_dataset(dataset_config.hf_queries_dataset_path, dataset_config.split)

    if dataset_config.use_snapshot:
        if not hasattr(dataset_config, 'repo_json_path'):
            raise ValueError(f"Missing 'repo_json_path' for snapshot dataset {dataset_name}")
        json_path = dataset_config.repo_json_path
        logger.info(f"Loading corpus for {dataset_name} using snapshot_download...")
        corpus_data = load_with_snapshot(dataset_config.hf_corpus_dataset_path, json_path)
    else:
        logger.info(f"Loading corpus for {dataset_name} using datasets library...")
        corpus_data = load_with_dataset(dataset_config.hf_corpus_dataset_path, dataset_config.split)

    logger.info(f"Successfully loaded dataset: {dataset_name}")
    return DatasetBundle(
        name=dataset_name,
        corpus=corpus_data,
        queries=queries_data,
    )


def collect_all_target_datasets(config: DictConfig) -> Dict[str, DatasetBundle]:
    logger.info(f"Collecting target datasets: {config.sub_datasets.keys()}")

    datasets = {
        name: load_single_dataset(name, d_config)
        for name, d_config in config.sub_datasets.items()
    }

    logger.info(f"Successfully loaded {len(datasets)} datasets")
    return datasets

def load_config() -> DictConfig:
    """Load the target dataset configuration from the default path."""
    config_path = Path(get_original_cwd()) / "benchmark_src" / "config" / "dataset" / "target.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Target config file not found at {config_path}")
    return OmegaConf.load(config_path)

def get_target_dataset_by_name(dataset_name: str) -> DatasetBundle:
    """
    Load a specific target dataset by name using the provided configuration path.
    """
    config = load_config()

    if not hasattr(config, "sub_datasets") or dataset_name not in config.sub_datasets:
        raise KeyError(f"Dataset '{dataset_name}' not found in sub_datasets of the config.")

    dataset_config: DictConfig = config.sub_datasets[dataset_name]
    return load_single_dataset(dataset_name, dataset_config)


def main():
    logger.info("Starting to collect all target datasets...")

    datasets = collect_all_target_datasets(config=load_config())

    logger.info("Dataset collection completed successfully!")
    logger.info(f"Loaded datasets: {list(datasets.keys())}")

    for dataset_name, dataset_data in datasets.items():
        logger.info(f"  - {dataset_name}:")
        logger.info(f"    Corpus: {len(dataset_data.corpus)} rows (Dataset)")
        logger.info(f"    Queries: {len(dataset_data.queries)} rows (Dataset)")

    return datasets


if __name__ == "__main__":
    try :
        main()
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}", exc_info=True)
