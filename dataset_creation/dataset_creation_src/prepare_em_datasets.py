import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path

import utils
from em_datasets import musicbrainz, geological_settlements, deepmatcher_datasets

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def main(cfg:DictConfig):
    utils.logger_init()
    logger.info(f"Got config: {cfg}")
    cfg.output_dir = Path(cfg.output_dir)

    assert Path(cfg['raw_datasets_dir']).exists(), f"Couldn't find raw_datasets_dir {cfg['raw_datasets_dir']}"

    Path.mkdir(cfg.output_dir, exist_ok=True) # Several datasets are saved in the same folder
    
    dataset_name = cfg["dataset"]["dataset_name"]

    print(f"#"*200)
    logger.info(f"Dataset: {dataset_name}")

    if dataset_name == "musicbrainz":
        dataset = musicbrainz.MusicBrainzDataset(cfg)
        dataset.prepare_data()
    elif dataset_name == "deepmatcher":
        deepmatcher_datasets.create_datasets(cfg)
    elif dataset_name == "geological-settlements":
        dataset = geological_settlements.GeologicalSettlementsDataset(cfg)
        dataset.prepare_data()
    else:
        raise ValueError(f"Unkown dataset name: {dataset_name}")

if __name__ == "__main__":
    main()