import logging
from pathlib import Path

import pandas as pd
from hydra.utils import get_original_cwd

logger = logging.getLogger(__name__)

LABEL_DICT = {
    "Product": 0,
    "Person": 1,
    "LocalBusiness": 2,
    "CreativeWork": 3,
    "Event": 4,
    "Place": 5,
    "Restaurant": 6,
    "Recipe": 7,
    "JobPosting": 8,
    "Hotel": 9,
}

TTD_DATA_SUBDIR = Path("cache/dataset_creation_resources/ttd")
SUPPORTED_SPLITS = {"train", "dev", "test"}


def _ttd_root() -> Path:
    return Path(get_original_cwd()) / TTD_DATA_SUBDIR


def _with_empty_header(df: pd.DataFrame) -> pd.DataFrame:
    """Match HyTrel TTD preprocessing: empty header, body from df.values."""
    table = pd.DataFrame(df.values)
    table.columns = ["" for _ in table.columns]
    return table


def load_ttd_split(split: str) -> tuple[list[pd.DataFrame], list[int]]:
    """
    Load one WDC Schema.org TTD split.

    This mirrors HyTrel's evaluate_ttd.py preprocessing:
    table headers are replaced by empty strings, and the label is the file
    prefix before the first underscore.
    """
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported TTD split '{split}'. Expected one of {sorted(SUPPORTED_SPLITS)}.")

    split_dir = _ttd_root() / split
    if not split_dir.exists():
        raise FileNotFoundError(f"TTD split directory does not exist: {split_dir}")

    tables: list[pd.DataFrame] = []
    labels: list[int] = []
    skipped = 0

    for path in sorted(split_dir.glob("*.json.gz")):
        label = path.name.split("_", 1)[0]
        if label not in LABEL_DICT:
            skipped += 1
            logger.warning(f"Skipping TTD file with unknown label prefix: {path.name}")
            continue

        try:
            df = pd.read_json(path, compression="gzip", lines=True)
        except ValueError:
            skipped += 1
            logger.exception(f"Skipping unreadable TTD file: {path}")
            continue

        if df.empty:
            skipped += 1
            logger.warning(f"Skipping empty TTD table: {path.name}")
            continue

        tables.append(_with_empty_header(df))
        labels.append(LABEL_DICT[label])

    logger.info(f"Loaded {len(tables)} TTD {split} tables from {split_dir}; skipped {skipped} files.")
    return tables, labels
