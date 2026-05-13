import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_ecb(
    data_dir: str,
    max_rows_per_table: int,
    min_rows: int = 2,
    min_cols: int = 3,
    max_cols: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load first max_rows_per_table rows from each ECB .csv.gz SDMX table.

    No row-counting pass — reads each file with pd.read_csv(nrows=...)
    and skips tables that don't meet min_rows/min_cols/max_cols.

    Returns a list of dicts with keys:
        database_id: "ecb"
        table_id: "{filename_stem}"
        table: [[header], [row1], [row2], ...]

    Args:
        data_dir: Path to the directory containing .csv.gz files.
        max_rows_per_table: Maximum data rows to read per table.
        min_rows: Minimum data rows to accept (excluding header).
        min_cols: Minimum columns to accept.
        max_cols: Maximum columns to accept, or None for unbounded.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"ECB data directory not found: {data_dir}")

    gz_files = sorted(data_path.glob("*.csv.gz"))
    logger.info(
        f"Loading up to {max_rows_per_table} rows from each of "
        f"{len(gz_files)} ECB .csv.gz files in {data_dir}"
    )

    tables: List[Dict[str, Any]] = []
    skipped = 0

    for filepath in gz_files:
        stem = filepath.stem.replace(".csv", "")

        try:
            df = pd.read_csv(filepath, compression="gzip", nrows=max_rows_per_table)
        except Exception:
            logger.warning(f"Failed to read {filepath.name}, skipping")
            skipped += 1
            continue

        num_rows = len(df)
        num_cols = len(df.columns)

        if num_rows < min_rows or num_cols < min_cols:
            skipped += 1
            continue
        if max_cols is not None and num_cols > max_cols:
            skipped += 1
            continue

        rows = [list(df.columns)] + df.values.tolist()
        tables.append(
            {
                "database_id": "ecb",
                "table_id": stem,
                "table": rows,
            }
        )

    logger.info(f"Loaded {len(tables)} ECB tables ({skipped} skipped)")
    return tables
