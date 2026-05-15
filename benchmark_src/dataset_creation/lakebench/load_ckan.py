import bz2
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)


def load_ckan(
    data_dir: str,
    min_rows: int = 2,
    max_rows: Optional[int] = None,
    min_cols: int = 3,
    max_cols: Optional[int] = None,
    max_tables: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load CKAN-subset .partN.csv.bz2 files as anchor tables, filtering in-stream.

    Reads part files (skipping .neg.*), applies row/column filters immediately,
    and caps at max_tables via reservoir sampling using the module-level RNG.

    Args:
        data_dir: Path to the directory containing .csv.bz2 files.
        min_rows: Minimum data rows (excluding header).
        max_rows: Maximum data rows, or None for unbounded.
        min_cols: Minimum columns.
        max_cols: Maximum columns, or None for unbounded.
        max_tables: Maximum tables to return, or None for unbounded.

    Returns:
        List of dicts with database_id, table_id, table keys.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"CKAN data directory not found: {data_dir}")

    part_files = sorted(
        p for p in data_path.glob("*.csv.bz2") if ".neg." not in p.name
    )

    logger.info(f"Found {len(part_files)} part files in {data_dir}")
    logger.info(
        f"Filter: {min_rows} <= rows <= {max_rows or 'inf'}, "
        f"{min_cols} <= cols <= {max_cols or 'inf'}, "
        f"max_tables={max_tables or 'inf'}"
    )

    tables: List[Dict[str, Any]] = []
    files_seen = 0
    files_loaded = 0

    for filepath in part_files:
        files_seen += 1

        try:
            with bz2.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = [row for row in reader]
        except Exception as e:
            logger.warning(f"Failed to read {filepath.name}, skipping,reason:\n{e}")
            continue

        if not rows or not rows[0]:
            continue

        num_rows = len(rows) - 1  # exclude header
        num_cols = len(rows[0])

        if num_rows < min_rows:
            continue
        if max_rows is not None and num_rows > max_rows:
            continue
        if num_cols < min_cols:
            continue
        if max_cols is not None and num_cols > max_cols:
            continue

        table_id = filepath.stem.replace(".csv", "")
        record = {
            "database_id": "ckan-subset",
            "table_id": table_id,
            "table": rows,
        }

        if max_tables is None:
            tables.append(record)
        elif len(tables) < max_tables:
            tables.append(record)
        else:
            # Reservoir sampling: replace with decreasing probability
            j = random.randint(0, files_loaded)
            if j < max_tables:
                tables[j] = record

        files_loaded += 1

    logger.info(
        f"Loaded {len(tables)} CKAN tables ({files_loaded} passed filters "
        f"out of {files_seen} files)"
    )
    return tables
