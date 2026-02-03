from hydra.utils import get_original_cwd
from pathlib import Path
import numpy as np
import glob
import json
import statistics
from tqdm import tqdm
import os
import sys


# Add ContextAwareJoin to Python path
# Path to src folder in ContextAwareJoin submodule
context_aware_join_src = Path("ContextAwareJoin") / "src"

# Make sure it exists
if not context_aware_join_src.exists():
    raise FileNotFoundError(f"{context_aware_join_src} does not exist!")

# Add to sys.path if not already present
if str(context_aware_join_src) not in sys.path:
    sys.path.insert(0, str(context_aware_join_src))
    print(f"Added {context_aware_join_src} to sys.path")

from myutils.utilities import load_dataframe, convert_to_dict_of_list, get_groundtruth_with_scores#
from benchmark_src.utils import load_benchmark


def get_leaf_dirs(root_dir, keep=None):
    """
    Find all lowest-level subdirectories (leaves in directory tree).
    
    Args:
        root_dir: Root directory to search
        keep: Optional substring that must be in path
    """
    leaf_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if keep and keep not in dirpath:
            continue
        if not dirnames:  # No subdirectories = leaf
            leaf_dirs.append(dirpath)
    return leaf_dirs


def load_benchmark_data(dataset_name):
    """
    Load benchmark datasets with lazy loading (paths only, not data).
    
    Returns:
        dict: {testcase_path: (table_paths, gt_data, dataset_name)}
        
    Memory-efficient: Only stores file paths, not actual dataframes.
    """
    # Determine dataset directory
    if dataset_name == 'wikijoin_small':
        dataset_dir = "ContextAwareJoin/datasets/wikijoin"
    else:
        dataset_dir = f"ContextAwareJoin/datasets/{dataset_name}"
    
    print(f"Looking for datasets in dir: {dataset_dir}")
    assert Path(dataset_dir).exists(), f"Could not find dataset dir: {dataset_dir}"

    if dataset_name == 'opendata':
        file_format = '.df'
    else:
        file_format = '.csv'

    print(f"Dataset name: {dataset_name}, file_format is {file_format}")

    # Determine leaf directories based on dataset type
    
    if dataset_name.lower() == "valentine":
        leaf_dirs = get_leaf_dirs(dataset_dir, keep='Joinable')
    elif dataset_name.lower() == "nextia":
        leaf_dirs = get_leaf_dirs(dataset_dir)
    else:
        leaf_dirs = [dataset_dir]

    if len(leaf_dirs) == 0:
        raise ValueError(f"Did not find the dataset. leaf_dirs={leaf_dirs}")

    print(f'LEAF_DIRS: {leaf_dirs}')
    
    test_cases = {}
    num_gt_test_cases = []
    num_tables = 0
    tables_num_rows = []
    tables_num_cols = []
    for dataset in tqdm(leaf_dirs):
        ############################
        # Load GT first
        ############################
        if dataset_name.lower() == "valentine":
            gt = glob.glob(f"{dataset}/*mapping.json", recursive=True)
        elif dataset_name.lower() == "wikijoin_small":
            gt = glob.glob(f"{dataset}/gt_small.*", recursive=True)
        elif dataset_name.lower() == "nextia":
            d = dataset.replace('/datalake', '')
            print(d)
            gt = glob.glob(f"{d}/**/gt.*", recursive=True)
        else:
            gt = glob.glob(f"{dataset}/**/gt.*", recursive=True)
            gt = [x for x in gt if x.endswith('json') or x.endswith('jsonl') or x.endswith('pickle')]
            print(f"else case. Found gt: {gt}")

        assert len(gt) == 1, f"Error: gt is {gt}"
        gt = gt[0]

        # Load ground truth data
        if gt.endswith('.json'):
            gt_data = json.load(open(gt, 'r'))
        elif gt.endswith('.jsonl'):
            gt_data = convert_to_dict_of_list(gt)
        elif gt.endswith('.pickle'):
            raise NotImplementedError
        else:
            raise NotImplementedError

         # Find all datalake tables
        datalake_tables = glob.glob(f"{dataset}/**/*{file_format}", recursive=True)
        # also get all .CSV files (some datasets have uppercase extensions)
        if file_format == '.csv':
            datalake_tables += glob.glob(f"{dataset}/**/*{file_format.upper()}", recursive=True)
        
        # Filter tables for wikijoin_small upfront (only include tables referenced in GT)
        if dataset_name.lower() == "wikijoin_small":
            l = [x.split('.')[0] + '.csv' for x in gt_data.keys()]
            alx = []
            for x in gt_data.values():
                for y in x:
                    alx.append(y.split('.')[0] + '.csv')
            fls = set(l + alx)  # Use set for faster lookup
            
            filtered_tables = []
            for table in datalake_tables:
                x = table.split('/')[-1]
                if x in fls:
                    filtered_tables.append(table)
            
            assert len(filtered_tables) < len(datalake_tables)
            assert len(filtered_tables) > 0, filtered_tables
            print(f'created a small version of wikijoin with {len(filtered_tables)} tables')
            datalake_tables = filtered_tables

        # Validate tables can be loaded (optional check)
        use_tqdm = len(datalake_tables) > 20
        iterator = tqdm(datalake_tables, desc="Validating Datasets") if use_tqdm else datalake_tables

        valid_tables = []
        for table in iterator:
            try:
                #logger.debug(f'validating table: {table}')
                df = load_benchmark.load_dataframe(table, file_format=file_format)
                # make sure there is at least one row
                if len(df) == 0:
                    print("empty df")
                    continue
                if len(df.columns) == 1:
                    raise Exception(f"Table {table} loaded with {len(df.columns)} columns: {df.columns}")
                tables_num_rows.append(len(df))
                tables_num_cols.append(len(df.columns))
                valid_tables.append(table)
                num_tables += 1
                del df  # Free memory immediately
            except Exception as e:
                print(f"---->")
                print(e)
        num_gt_test_cases.append(len(gt_data))
        # Store table paths instead of loaded dataframes
        table_paths = {table: file_format for table in valid_tables}
        test_cases[dataset] = table_paths, gt_data, dataset.replace('/', '_')

    print(f"Loaded benchmark: {num_tables} tables, num test cases (gt): {sum(num_gt_test_cases)}.")
    print(f"Tables stats - Avg rows: {statistics.mean(tables_num_rows) if tables_num_rows else 0}, Avg cols: {statistics.mean(tables_num_cols) if tables_num_cols else 0}")

    print(f"Total test cases loaded: {len(test_cases)} in cfg dataset {dataset_name}")

    return test_cases

def remove_table_from_gt(table_name, gt_data):  
    """
    Skip key-value pairs in ground truth where either key or value belongs to table_name.
    
    :param table_name: Description
    :param gt_data: Description
    """
    new_gt = {}
    for key, values in gt_data.items():
        # TODO: we need everything before the last . (might include other dots before) as one string
        key_table = key.rsplit(".", 1)[0]
        if key_table == table_name:
            continue  # skip this key entirely
        not_in_values = True
        for value in values:
            value_table = value.rsplit(".", 1)[0]
            if value_table == table_name:
                not_in_values = False
                continue  # skip this value
        if not_in_values:
            new_gt[key] = values
    return new_gt

def run_validation(dataset_name: str):

    valid_data = {}

    # first try to load gt
    test_cases = load_benchmark_data(dataset_name)
    #print(f"Have sub_datasets: {test_cases.keys()}")
    for testcase in test_cases:
        table_paths, gt_data, sub_dataset_name = test_cases[testcase]
        print("Subdataset")
        print(sub_dataset_name)

        print(f"Have {len(table_paths)} valid tables with {len(gt_data)} ground truth entries")


        # next load all tables from the dataload (using robust dataloding method)

        all_tables_in_gt = set()
        gt_key_tables = set()
        gt_value_tables = set()
        for key, values in gt_data.items():
            # column name within wikijoin data that includes "e.g."
            if "e.g." in key:
                key = value.replace("e.g.", "")
            key_table = key.rsplit(".", 1)[0]
            all_tables_in_gt.add(key_table)
            gt_key_tables.add(key_table)
            for value in values:
                if "e.g." in value:
                    value = value.replace("e.g.", "")
                value_table = value.rsplit(".", 1)[0]
                all_tables_in_gt.add(value_table)
                gt_value_tables.add(value_table)

        all_tables_valid_loaded = set()
        for table_path in table_paths.keys():
            table_name = Path(table_path)
            all_tables_valid_loaded.add(table_name.stem)

        # get intersection of sets:
        new_gt_needed = False
        for table in all_tables_in_gt:
            if table not in all_tables_valid_loaded:

                print(f"GT Table {table} not found!!")

                if table in gt_key_tables:
                    print("is key table")
                else:
                    print("is value table") 

                gt_data = remove_table_from_gt(table, gt_data)
                new_gt_needed = True
        
        if not new_gt_needed:
            print("All good, have all GT tables among the loaded tables")
        
        else:
            print(f"After removal, new GT has {len(gt_data)} keys")

        valid_data[testcase] = {"table_paths": table_paths, "gt_data": gt_data, "sub_dataset_name": sub_dataset_name}
        print(f"Saved to cache: {len(table_paths)} valid paths, {len(gt_data)} valid gt testcases")

    # save valid data to cache(load them instead of validating again in the task script)
    cache_folder = Path("cache/datasets/column_similarity_search") / dataset_name
    cache_folder.mkdir(exist_ok=True, parents=True)
    with open(cache_folder / "valid_data.json", "w") as file:
        json.dump(valid_data, file, indent=2)

    

if __name__ == "__main__":
    # valentine has other dataformat for GT, so skipping it here
    dataset_names = ["opendata", "autojoin", "nextia", "wikijoin_small"]
    #dataset_names = ["wikijoin_small"]
    for dataset_name in dataset_names:
        run_validation(dataset_name=dataset_name)

    # wikijoin has no loading issues, everything is fine (run again with full GT?)

    # opendata has a lot of .df files that have just one column, but all gt data is present in the valid tables

    # nextia has some dtype warnings due to mixed types and has a few (3) tables that could not be loaded robustly but that appear in the GT

    # autojoin has two tables that could not be loaded which were in the GT

    # valentine has other file format for GT, but all tables load fine so no issues..