"""
Download and process the SOTAB Round2-CTA-SCH dataset (SemTab 2023 repackaging of WDC SOTAB-V2,
https://zenodo.org/records/8422037) into the framework's column_type_annotation cache format.

The relative paths below were confirmed by downloading and extracting the archive; its top-level
folder is "SOTAB V2 for SemTab 2023/".

Usage:
    python download_sotab.py --raw_datasets_dir cache/raw_datasets/sotab \
                              --output_dir cache/datasets/column_type_annotation/sotab
"""
import argparse
import gzip
import json
import logging
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ZENODO_ZIP_URL = "https://zenodo.org/api/records/8422037/files/SOTAB%20V2%20for%20SemTab%202023.zip/content"
ZENODO_ZIP_FILENAME = "SOTAB_V2_for_SemTab_2023.zip"

ROUND2_CTA_SCH_DIR = "SOTAB V2 for SemTab 2023/Round2-SOTAB-CTA-SCH-DatasetsAndGroundTruth"
LABEL_VOCAB_RELATIVE_PATH = f"{ROUND2_CTA_SCH_DIR}/cta_labels_round2.txt"
GT_RELATIVE_PATHS = {
    "train": f"{ROUND2_CTA_SCH_DIR}/sotab_cta_train_round2.csv",
    "test": f"{ROUND2_CTA_SCH_DIR}/gt/sotab_cta_test_round2.csv",
}
MIN_COLS_PER_CLASS = 50


def download_zip(url: str, dest_path: Path):
    if dest_path.exists():
        logger.info(f"Zip already downloaded at {dest_path}, skipping download.")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    logger.info(f"Downloading {url} -> {dest_path} (this is ~4GB, may take a while)")
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_path.rename(dest_path)
    logger.info("Download complete.")


def extract_zip(zip_path: Path, extract_to: Path, marker_name: str = ".extracted"):
    marker = extract_to / marker_name
    if marker.exists():
        logger.info(f"Already extracted at {extract_to}, skipping extraction.")
        return
    extract_to.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    # the archive may itself contain nested zips per round/task - extract those too
    for nested_zip in extract_to.rglob("*.zip"):
        nested_dir = nested_zip.with_suffix("")
        if nested_dir.exists():
            continue
        logger.info(f"Extracting nested zip {nested_zip} -> {nested_dir}")
        try:
            with zipfile.ZipFile(nested_zip, "r") as zf:
                zf.extractall(nested_dir)
        except zipfile.BadZipFile:
            logger.warning(f"Could not extract nested zip {nested_zip}, skipping.")

    marker.write_text("done")


def build_table_index(root: Path) -> dict:
    """Maps table filename (as it appears in the ground truth CSVs) -> path on disk."""
    index = {}
    for path in root.rglob("*"):
        if path.is_file() and (path.name.endswith(".json.gz") or path.name.endswith(".json")):
            index[path.name] = path
    logger.info(f"Indexed {len(index)} candidate table files under {root}")
    return index


def load_ground_truth(gt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(gt_path, header=None, names=["table_name", "column_index", "label"], dtype=str)
    # if the file actually had a header row, drop it
    if str(df.iloc[0]["table_name"]).strip().lower() == "table_name":
        df = df.iloc[1:].reset_index(drop=True)
    df["column_index"] = df["column_index"].astype(int)
    return df


def load_sotab_table(path: Path) -> pd.DataFrame:
    """Loads one SOTAB table file (gzip JSON-lines, no header) as a DataFrame with positional columns."""
    compression = "gzip" if path.name.endswith(".gz") else None
    rows = []
    opener = gzip.open if compression == "gzip" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if len(rows) == 0:
        return pd.DataFrame()

    if isinstance(rows[0], dict):
        df = pd.DataFrame(rows)
    else:
        # rows are JSON arrays -> positional columns
        df = pd.DataFrame(rows)
    return df


def select_train_tables(gt_df: pd.DataFrame, min_cols_per_class: int) -> set:
    """
    Greedily selects the minimal set of tables (alphabetical order for determinism) such that
    every class in gt_df has at least min_cols_per_class labeled columns in the selection.
    """
    class_counts = Counter()
    selected = set()
    for table_name, group in gt_df.groupby("table_name"):
        labels = group["label"].tolist()
        if any(class_counts[l] < min_cols_per_class for l in labels):
            selected.add(table_name)
            class_counts.update(labels)
    logger.info(
        f"Stratified train selection: {len(selected)} tables, "
        f"{sum(class_counts.values())} labeled cols, "
        f"{len(class_counts)} classes each with >= {min_cols_per_class} cols"
    )
    return selected


def convert_split(split_name: str, gt_df: pd.DataFrame, table_index: dict, tables_out_dir: Path, selected_tables: set = None) -> tuple[dict, dict]:
    """
    Converts tables referenced in gt_df to plain CSVs under tables_out_dir.

    If selected_tables is given, only those table names are converted.

    Returns:
        table_paths: {absolute_path: ".csv"}
        column_labels: {csv_filename: {column_index_str: label}}
    """
    tables_out_dir.mkdir(parents=True, exist_ok=True)
    table_paths = {}
    column_labels = {}

    for table_name, group in gt_df.groupby("table_name"):
        if selected_tables is not None and table_name not in selected_tables:
            continue
        if table_name not in table_index:
            logger.warning(f"[{split_name}] Ground truth references table '{table_name}' which was not found on disk, skipping.")
            continue

        csv_name = Path(table_name).name
        for suffix in (".json.gz", ".json"):
            if csv_name.endswith(suffix):
                csv_name = csv_name[: -len(suffix)]
                break
        csv_name = f"{csv_name}.csv"
        out_path = tables_out_dir / csv_name

        if not out_path.exists():
            try:
                df = load_sotab_table(table_index[table_name])
            except Exception as e:
                logger.error(f"[{split_name}] Failed to load table {table_name}: {e}, skipping.")
                continue
            df.to_csv(out_path, index=False)

        table_paths[str(out_path)] = ".csv"
        column_labels[csv_name] = {
            str(row["column_index"]): row["label"] for _, row in group.iterrows()
        }

    logger.info(f"[{split_name}] Converted {len(table_paths)} tables, {sum(len(v) for v in column_labels.values())} labeled columns.")
    return table_paths, column_labels


def ensure_dataset(raw_datasets_dir: Path, output_dir: Path):
    """
    Makes sure the processed column_type_annotation/sotab dataset exists under output_dir,
    downloading and processing the raw SOTAB archive if it doesn't.

    Called both from the CLI entrypoint below and directly from the task runner on first use
    (no separate manual setup step needed).
    """
    raw_datasets_dir = Path(raw_datasets_dir)
    output_dir = Path(output_dir)

    if (output_dir / "valid_data.json").exists():
        logger.info(f"{output_dir / 'valid_data.json'} already exists, skipping SOTAB dataset creation.")
        return

    zip_path = raw_datasets_dir / ZENODO_ZIP_FILENAME
    extracted_dir = raw_datasets_dir / "extracted"

    download_zip(ZENODO_ZIP_URL, zip_path)
    extract_zip(zip_path, extracted_dir)

    label_vocab_path = extracted_dir / LABEL_VOCAB_RELATIVE_PATH
    assert label_vocab_path.exists(), f"Expected label vocab file at {label_vocab_path}"
    label_vocab = [line.strip() for line in label_vocab_path.read_text().splitlines() if line.strip()]
    logger.info(f"Loaded {len(label_vocab)} labels from {label_vocab_path}")

    table_index = build_table_index(extracted_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    splits_out = {}
    for split_name in ["train", "test"]:
        gt_path = extracted_dir / GT_RELATIVE_PATHS[split_name]
        assert gt_path.exists(), f"Expected {split_name} ground truth file at {gt_path}"
        gt_df = load_ground_truth(gt_path)

        selected_tables = select_train_tables(gt_df, MIN_COLS_PER_CLASS) if split_name == "train" else None
        table_paths, column_labels = convert_split(
            split_name=split_name,
            gt_df=gt_df,
            table_index=table_index,
            tables_out_dir=output_dir / "tables" / split_name,
            selected_tables=selected_tables,
        )
        splits_out[split_name] = {"table_paths": table_paths, "column_labels": column_labels}

    valid_data = {"label_vocab": label_vocab, "splits": splits_out}
    with open(output_dir / "valid_data.json", "w") as f:
        json.dump(valid_data, f, indent=2)

    logger.info(f"Wrote {output_dir / 'valid_data.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_datasets_dir", type=str, default="cache/raw_datasets/sotab")
    parser.add_argument("--output_dir", type=str, default="cache/datasets/column_type_annotation/sotab")
    args = parser.parse_args()

    logging.basicConfig(format="{levelname} - {name} - {asctime} {message}", level=logging.INFO, style="{", datefmt="%Y-%m-%d %H:%M")

    ensure_dataset(raw_datasets_dir=args.raw_datasets_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
