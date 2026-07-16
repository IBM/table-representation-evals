"""
Download and process the GitTables column type detection (CTA) benchmark
(Zenodo record 5706316, https://zenodo.org/records/5706316) into the framework's
column_type_annotation cache format, using the DBpedia-typed annotation set
(2533 labeled columns across 122 classes).

Usage:
    python download_gittables_cta.py --raw_datasets_dir cache/raw_datasets/gittables_cta \
                                      --output_dir cache/datasets/column_type_annotation/gittables_cta
"""
import argparse
import json
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ZENODO_FILES = {
    "tables.zip": "https://zenodo.org/api/records/5706316/files/tables.zip/content",
    "dbpedia_gt.csv": "https://zenodo.org/api/records/5706316/files/dbpedia_gt.csv/content",
    "dbpedia_labels.csv": "https://zenodo.org/api/records/5706316/files/dbpedia_labels.csv/content",
}
TRAIN_FRACTION = 0.8


def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        logger.info(f"{dest_path} already downloaded, skipping download.")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    logger.info(f"Downloading {url} -> {dest_path}")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_path.rename(dest_path)


def extract_zip(zip_path: Path, extract_to: Path, marker_name: str = ".extracted"):
    marker = extract_to / marker_name
    if marker.exists():
        logger.info(f"Already extracted at {extract_to}, skipping extraction.")
        return
    extract_to.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # the archive also carries __MACOSX/ resource-fork junk alongside tables/*.csv
            if member.filename.startswith("tables/") and member.filename.endswith(".csv"):
                zf.extract(member, extract_to)
    marker.write_text("done")


def load_ground_truth(gt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(gt_path, usecols=["table_id", "target_column", "annotation_label"])
    # table_id carries a "_dbpedia" suffix not present in the table filenames inside tables.zip
    df["table_name"] = df["table_id"].str.removesuffix("_dbpedia") + ".csv"
    return df


def split_table_names(table_names: list, train_fraction: float) -> tuple[set, set]:
    """
    Deterministic split by sorted table name. No per-class minimum (unlike SOTAB's
    select_train_tables): with only 122 classes across 2533 labeled columns, most classes
    are too sparse for a per-split minimum to be achievable at all.
    """
    ordered = sorted(table_names)
    split_point = int(len(ordered) * train_fraction)
    return set(ordered[:split_point]), set(ordered[split_point:])


def convert_tables(gt_df: pd.DataFrame, raw_tables_dir: Path, selected_names: set, tables_out_dir: Path) -> tuple[dict, dict]:
    """
    Converts the tables in selected_names to plain CSVs under tables_out_dir.

    Returns:
        table_paths: {absolute_path: ".csv"}
        column_labels: {csv_filename: {column_index_str: label}}
    """
    tables_out_dir.mkdir(parents=True, exist_ok=True)
    table_paths = {}
    column_labels = {}

    for table_name, group in gt_df[gt_df["table_name"].isin(selected_names)].groupby("table_name"):
        raw_path = raw_tables_dir / table_name
        if not raw_path.exists():
            logger.warning(f"Ground truth references table '{table_name}' which was not found in the archive, skipping.")
            continue

        out_path = tables_out_dir / table_name
        if not out_path.exists():
            # the raw CSV's first column is an unlabeled row index; drop it so ground-truth
            # column indices (0-based over col0..colN) line up with the columns written here
            df = pd.read_csv(raw_path, index_col=0)
            df.to_csv(out_path, index=False)

        table_paths[str(out_path)] = ".csv"
        column_labels[table_name] = {
            str(row["target_column"]): row["annotation_label"] for _, row in group.iterrows()
        }

    return table_paths, column_labels


def ensure_dataset(raw_datasets_dir: Path, output_dir: Path):
    """
    Makes sure the processed column_type_annotation/gittables_cta dataset exists under
    output_dir, downloading and processing the raw GitTables CTA benchmark if it doesn't.

    Called both from the CLI entrypoint below and directly from the task runner on first use
    (no separate manual setup step needed).
    """
    raw_datasets_dir = Path(raw_datasets_dir)
    output_dir = Path(output_dir)

    if (output_dir / "valid_data.json").exists():
        logger.info(f"{output_dir / 'valid_data.json'} already exists, skipping GitTables CTA dataset creation.")
        return

    for filename, url in ZENODO_FILES.items():
        download_file(url, raw_datasets_dir / filename)

    extracted_dir = raw_datasets_dir / "extracted"
    extract_zip(raw_datasets_dir / "tables.zip", extracted_dir)
    raw_tables_dir = extracted_dir / "tables"

    label_vocab = pd.read_csv(raw_datasets_dir / "dbpedia_labels.csv")["annotation_label"].tolist()
    logger.info(f"Loaded {len(label_vocab)} labels from dbpedia_labels.csv")

    gt_df = load_ground_truth(raw_datasets_dir / "dbpedia_gt.csv")
    train_names, test_names = split_table_names(gt_df["table_name"].unique().tolist(), TRAIN_FRACTION)

    output_dir.mkdir(parents=True, exist_ok=True)
    splits_out = {}
    for split_name, selected_names in [("train", train_names), ("test", test_names)]:
        table_paths, column_labels = convert_tables(
            gt_df=gt_df,
            raw_tables_dir=raw_tables_dir,
            selected_names=selected_names,
            tables_out_dir=output_dir / "tables" / split_name,
        )
        splits_out[split_name] = {"table_paths": table_paths, "column_labels": column_labels}
        logger.info(
            f"[{split_name}] Converted {len(table_paths)} tables, "
            f"{sum(len(v) for v in column_labels.values())} labeled columns."
        )

    valid_data = {"label_vocab": label_vocab, "splits": splits_out}
    with open(output_dir / "valid_data.json", "w") as f:
        json.dump(valid_data, f, indent=2)

    logger.info(f"Wrote {output_dir / 'valid_data.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_datasets_dir", type=str, default="cache/raw_datasets/gittables_cta")
    parser.add_argument("--output_dir", type=str, default="cache/datasets/column_type_annotation/gittables_cta")
    args = parser.parse_args()

    logging.basicConfig(format="{levelname} - {name} - {asctime} {message}", level=logging.INFO, style="{", datefmt="%Y-%m-%d %H:%M")

    ensure_dataset(raw_datasets_dir=Path(args.raw_datasets_dir), output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
