import logging
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from hydra.utils import get_original_cwd
from tqdm import tqdm

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

WDC_DATA_SUBDIR = Path("cache/dataset_creation_resources/ttd")
SUPPORTED_SPLITS = {"train", "dev", "test"}


def _wdc_root() -> Path:
    return Path(get_original_cwd()) / WDC_DATA_SUBDIR


def _load_single(path: Path) -> pd.DataFrame | None:
    """Load one TTD file, returning a table with empty column headers or None on failure."""
    try:
        df = pd.read_json(path, compression="gzip", lines=True)
    except ValueError:
        return None
    if df.empty:
        return None
    table = pd.DataFrame(df.values)
    table.columns = [""] * len(table.columns)
    return table


def _filter_paths(all_paths: list[Path]) -> tuple[list[tuple[Path, int]], int]:
    """Filter to paths with known label prefixes. Returns (valid_paths, unknown_count)."""
    valid_paths = []
    unknown_count = 0
    for path in all_paths:
        label = path.name.split("_", 1)[0]
        if label not in LABEL_DICT:
            unknown_count += 1
            logger.warning(f"Skipping WDC file with unknown label prefix: {path.name}")
        else:
            valid_paths.append((path, LABEL_DICT[label]))
    return valid_paths, unknown_count


def _apply_limit(valid_paths: list[tuple[Path, int]], limit: int) -> list[tuple[Path, int]]:
    """Subsample to at most `limit` entries, balanced across classes."""
    by_label: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for path, label_id in valid_paths:
        by_label[label_id].append((path, label_id))
    per_class = math.ceil(limit / len(by_label))
    limited = [entry for entries in by_label.values() for entry in entries[:per_class]]
    return limited[:limit]


def load_wdc_split(split: str, limit: int | None = None) -> tuple[list[pd.DataFrame], list[int]]:
    """
    Load one WDC Schema.org split.

    This mirrors HyTrel's evaluate_ttd.py preprocessing:
    table headers are replaced by empty strings, and the label is the file
    prefix before the first underscore.
    """
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported WDC split '{split}'. Expected one of {sorted(SUPPORTED_SPLITS)}.")

    split_dir = _wdc_root() / split
    if not split_dir.exists():
        raise FileNotFoundError(f"WDC split directory does not exist: {split_dir}")

    valid_paths, skipped = _filter_paths(sorted(split_dir.glob("*.json.gz")))

    if limit is not None:
        valid_paths = _apply_limit(valid_paths, limit)

    ordered: dict[int, tuple[pd.DataFrame, int]] = {}

    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_load_single, path): (i, label_id) for i, (path, label_id) in enumerate(valid_paths)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading WDC {split}", mininterval=10):
            i, label_id = futures[future]
            table = future.result()
            if table is None:
                skipped += 1
            else:
                ordered[i] = (table, label_id)

    tables = [ordered[i][0] for i in sorted(ordered)]
    labels = [ordered[i][1] for i in sorted(ordered)]

    logger.info(f"Loaded {len(tables)} WDC {split} tables from {split_dir}; skipped {skipped} files.")
    return tables, labels
