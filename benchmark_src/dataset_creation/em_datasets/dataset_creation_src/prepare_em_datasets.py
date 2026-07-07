import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from omegaconf import OmegaConf

import utils
from em_datasets import musicbrainz, geological_settlements, deepmatcher_datasets

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def main(
    dataset: Annotated[str, typer.Option(help="Dataset name (deepmatcher | musicbrainz | geological-settlements)")],
    output_dir: Annotated[str, typer.Option(help="Directory to save created datasets")],
    raw_datasets_dir: Annotated[str, typer.Option(help="Directory with raw downloaded datasets")],
    table_row_limit: Annotated[Optional[int], typer.Option(help="Limit input tables to this many rows (omit for no limit)")] = None,
):
    utils.logger_init()

    raw_datasets_dir = Path(raw_datasets_dir)
    assert raw_datasets_dir.exists(), f"Couldn't find raw_datasets_dir {raw_datasets_dir}, please make sure that the folder exists"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    dataset_cfg_path = _CONFIG_DIR / "dataset" / f"{dataset}.yaml"
    if not dataset_cfg_path.exists():
        raise typer.BadParameter(f"No dataset config found at {dataset_cfg_path}")

    cfg = OmegaConf.merge(
        OmegaConf.create({"raw_datasets_dir": str(raw_datasets_dir), "output_dir": str(output_path), "table_row_limit": table_row_limit}),
        OmegaConf.load(dataset_cfg_path),
    )

    dataset_name = cfg.dataset_name
    logger.debug(f"Got config: {cfg}")
    logger.info("#" * 200)
    logger.info(f"Dataset: {dataset_name}")

    if dataset_name == "musicbrainz":
        ds = musicbrainz.MusicBrainzDataset(cfg)
        ds.prepare_data()
    elif dataset_name == "deepmatcher":
        deepmatcher_datasets.create_datasets(cfg)
    elif dataset_name == "geological-settlements":
        ds = geological_settlements.GeologicalSettlementsDataset(cfg)
        ds.prepare_data()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


if __name__ == "__main__":
    typer.run(main)