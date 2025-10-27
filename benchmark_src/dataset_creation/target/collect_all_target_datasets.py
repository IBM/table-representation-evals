import json
import logging
from pathlib import Path
from typing import Dict, Any, Union
import yaml
from datasets import Dataset
from datasets import load_dataset
from huggingface_hub import snapshot_download


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_with_dataset(hf_dataset_path: str, split: str) -> Dataset:
    dataset = load_dataset(path=hf_dataset_path, split=split)
    return dataset


def load_with_snapshot(corpus_hf_path: str, json_path: str) -> dict:
    path_to_data_dir = snapshot_download(repo_id=corpus_hf_path, repo_type="dataset")
    logger.info(f"Loading corpus from {path_to_data_dir}")
    path_to_dataset = Path(path_to_data_dir, json_path)
    with open(path_to_dataset, "r") as file:
        corpus = json.load(file)
    return corpus


def load_target_config(config_path: Path) -> Dict[str, dict]:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded config from {config_path}")
    return config


def load_single_dataset(dataset_name: str, dataset_config: Dict[str, str]) -> Dict[str, Union[Dataset, Dict, Any]]:
    required_fields = ['hf_corpus_dataset_path', 'hf_queries_dataset_path', 'split', 'use_snapshot']
    for field in required_fields:
        if field not in dataset_config:
            raise ValueError(f"Missing required field '{field}' for dataset {dataset_name}")

    split = dataset_config['split']
    corpus_path = dataset_config['hf_corpus_dataset_path']

    logger.info(f"Loading queries for {dataset_name} using datasets library...")
    queries_data = load_with_dataset(dataset_config['hf_queries_dataset_path'], split)

    if dataset_config['use_snapshot']:
        if 'repo_json_path' not in dataset_config:
            raise ValueError(f"Missing 'repo_json_path' for snapshot dataset {dataset_name}")
        json_path = dataset_config['repo_json_path']
        logger.info(f"Loading corpus for {dataset_name} using snapshot_download...")
        corpus_data = load_with_snapshot(corpus_path, json_path)
    else:
        logger.info(f"Loading corpus for {dataset_name} using datasets library...")
        corpus_data = load_with_dataset(corpus_path, split)

    logger.info(f"Successfully loaded dataset: {dataset_name}")
    return {
        'corpus': corpus_data,
        'queries': queries_data,
        'config': dataset_config
    }


def collect_all_target_datasets(config: Dict[str, dict]) -> Dict[str, Dict]:
    sub_datasets = config['sub_datasets']
    logger.info(f"Found {len(sub_datasets)} datasets in configuration")

    datasets = {
        name: load_single_dataset(name, d_config)
        for name, d_config in sub_datasets.items()
    }

    logger.info(f"Successfully loaded {len(datasets)} datasets")
    return datasets


def main():
    logger.info("Starting to collect all target datasets...")
    config_path = Path.cwd() / "benchmark_src" / "config" / "dataset" / "target.yaml"

    try:
        config_data = load_target_config(config_path)
    except FileNotFoundError as e:
        logger.error(f"Error: Target config file not found at {config_path}")
        raise e

    datasets = collect_all_target_datasets(config=config_data)

    logger.info("Dataset collection completed successfully!")
    logger.info(f"Loaded datasets: {list(datasets.keys())}")

    for dataset_name, dataset_data in datasets.items():
        config = dataset_data['config']
        query_type = config.get('query_type', 'Unknown')
        logger.info(f"  - {dataset_name}: {query_type}")

        corpus = dataset_data['corpus']
        if isinstance(corpus, Dataset):
            logger.info(f"    Corpus: {len(corpus)} rows (Dataset)")
        elif isinstance(corpus, dict):
            logger.info(f"    Corpus: {len(corpus)} keys (dict)")

        queries = dataset_data['queries']
        logger.info(f"    Queries: {len(queries)} rows (Dataset)")

        return datasets

    return None


if __name__ == "__main__":
    try :
        main()
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}", exc_info=True)
