import requests
import os
from tqdm import tqdm
from pathlib import Path
import zipfile
import pandas as pd
from collections import defaultdict, Counter
import json
from pyarrow import parquet as pq
import csv
import random


datalakes = {
    "datalake_01": "metabolic_rate_tables_licensed.zip",
    "datalake_02": "incubation_period_tables_licensed.zip",
    "datalake_03": "beats_per_minute_tables_licensed.zip",
    "datalake_04": "crime_rate_tables_licensed.zip",
    "datalake_05": "orbit_period_tables_licensed.zip",
    "datalake_06": "radial_velocity_tables_licensed.zip",
    "datalake_07": "kilometers_per_hour_tables_licensed.zip",
    "datalake_08": "miles_per_hour_tables_licensed.zip",
    "datalake_09": "neonatal_mortality_tables_licensed.zip",
    "datalake_10": "menstrual_cycle_tables_licensed.zip",
    "datalake_11": "fertile_period_tables_licensed.zip",
    "datalake_12": "cardiac_output_tables_licensed.zip",
    "datalake_13": "inflation_rate_tables_licensed.zip",
    "datalake_14": "return_on_invested_capital_tables_licensed.zip",
    "datalake_15": "sampling_rate_tables_licensed.zip",
    "datalake_16": "rotational_latency_tables_licensed.zip"
}

# Domains: biomedical, physics/astronomy, social/economic, engineering/signal processing

# Zenodo record for GitTables 1M (version 0.0.6)
ZENODO_BASE_URL = "https://zenodo.org/record/6517052/files/"

CACHE_PATH_RESOURCES = Path("cache/dataset_creation_resources/gitTables_table_retrieval")
CACHE_PATH_FINAL_DATA = Path("cache/datasets/table_retrieval/gitTables")

# Minimum tables per repo to be eligible as query table
MIN_TABLES_FOR_QUERY = 3

def download_file(filename):
    out_path = os.path.join(CACHE_PATH_RESOURCES, filename)
    if os.path.exists(out_path):
        print(f"Already exists, skipping: {filename}")
        return

    url = ZENODO_BASE_URL + filename
    print(f"Downloading: {filename} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(out_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=filename
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print(f"Downloaded: {filename}")

    # unzip the file into a folder with the same name (without .zip)
    with zipfile.ZipFile(out_path, "r") as zip_ref:
        extract_path = os.path.join(CACHE_PATH_RESOURCES, filename.replace(".zip", ""))
        if not os.path.exists(extract_path):
            print(f"Extracting: {filename} ...")
            zip_ref.extractall(extract_path)
            print(f"Extracted: {filename}")
        else:
            print(f"Already extracted, skipping: {filename}")


def create_dataset(cfg):
    pass

def process_data():
    # download the data
    for f in datalakes.values():
        download_file(f)

    # extract information about repositories the tables come from for each datalake
    datalake_repo_to_tables = {}  # datalake -> {repo: [table_ids]}

    for datalake_idx, f in datalakes.items():
        repo_to_tables = defaultdict(list)
        datalake_name = f.replace(".zip", "")
        datalake_extract_dir = CACHE_PATH_RESOURCES / datalake_name
        # Process all Parquet files
        parquet_files = list(datalake_extract_dir.glob("*.parquet"))

        print(f"Processing datalake: {datalake_name} with {len(parquet_files)} parquet files...")

        for pfile in parquet_files:
            filename = pfile.stem
            table = pq.read_table(pfile)
            metadata = table.schema.metadata
        
            # Extract repository from the csv url
            repo = None  # default fallback
            if b'gittables' in metadata:
                gittables_meta = json.loads(metadata[b'gittables'].decode("utf-8"))
                csv_url = gittables_meta.get("csv_url")
                if csv_url and "github.com" in csv_url:
                    repo = "/".join(csv_url.split("github.com/")[1].split("/")[:2])
                else:
                    # fallback to asset_id or ZIP file
                    repo = gittables_meta.get("asset_id", "<fallback>")

            repo_to_tables[repo].append(filename)

        # get number of tables per repo, use Counter to get distribution of table counts across repos
        repo_table_counts = {repo: len(tables) for repo, tables in repo_to_tables.items()}
        table_count_distribution = Counter(repo_table_counts.values())
        print(f"Repo table count distribution for {datalake_name}: {table_count_distribution}")
        datalake_repo_to_tables[datalake_idx] = repo_to_tables

    return datalake_repo_to_tables

def create_dataset(datalake_repo_to_tables):
    for datalake_idx, repo_to_tables in datalake_repo_to_tables.items():
        datalake_name = datalakes[datalake_idx].replace(".zip", "")
        datalake_extract_dir = CACHE_PATH_RESOURCES / datalake_name

        # Create output folders
        datalake_final_dir = CACHE_PATH_FINAL_DATA / datalake_name
        tables_dir = datalake_final_dir / "tables"
        testcases_dir = datalake_final_dir / "testcases"
        tables_dir.mkdir(parents=True, exist_ok=True)
        testcases_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating CSV files for {datalake_name}...")
        table_id_to_filename = {}

        # Convert all Parquet tables to CSV
        parquet_files = list(datalake_extract_dir.glob("*.parquet"))
        for pfile in parquet_files:
            table = pq.read_table(pfile)
            csv_filename = pfile.stem + ".csv"
            table_id_to_filename[pfile.stem] = csv_filename
            table.to_pandas().to_csv(tables_dir / csv_filename, index=False)

        print(f"Creating testcases for {datalake_name}...")
        testcase_counter = 1
        for repo, tables in repo_to_tables.items():
            if len(tables) >= MIN_TABLES_FOR_QUERY:
                # pick one table as query, use seeded random to ensure reproducibility
                random.seed(42 + testcase_counter)  # different seed for each testcase
                query_table = random.choice(tables)
                positives = [t for t in tables if t != query_table]

                # map table IDs to CSV filenames
                query_csv = table_id_to_filename[query_table]
                positives_csv = [table_id_to_filename[t] for t in positives]

                # save testcase as JSON
                testcase_file = testcases_dir / f"testcase_{testcase_counter:03d}.json"
                with open(testcase_file, "w") as f:
                    json.dump({"query": query_csv, "positives": positives_csv}, f, indent=2)

                testcase_counter += 1

        print(f"{datalake_name}: {testcase_counter-1} testcases created.")

if __name__ == "__main__":
    # create cache paths if they don't exist
    CACHE_PATH_RESOURCES.mkdir(parents=True, exist_ok=True)
    CACHE_PATH_FINAL_DATA.mkdir(parents=True, exist_ok=True)

    datalake_repo_to_tables = process_data()
    #
    create_dataset(datalake_repo_to_tables)