import json
import pandas as pd
from pathlib import Path
import logging
from io import StringIO
import os

DATA_PATH = Path("data/MusicBrainz")

logger = logging.getLogger(__name__)

def robust_csv_loader(dataset_path: "Path | str") -> pd.DataFrame:
    """
    Loads a CSV file robustly.
    - Detects encoding using chardet
    - Handles different delimiters
    - Automatically tries fallback encodings (utf-8, latin1, windows-1252)
    - Adds unicode_escape only if file contains escape sequences
    - Reads only first N rows for overview
    """
    # normalize dataset_path to Path
    dataset_path = Path(dataset_path)

    # --- Check if file is completely empty (size 0) ---
    if dataset_path.stat().st_size == 0:
        logger.error(f"File {dataset_path} is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    # --- Read a sample of raw bytes to detect encoding and inspect content ---
    with open(dataset_path, "rb") as f:
        rawdata = f.read(10000)

    # If the first chunk is empty or only whitespace bytes, conservatively check the whole file
    if not rawdata.strip():
        # read entire file size is non-zero but content may be whitespace; read safely with binary check
        with open(dataset_path, "rb") as f:
            all_bytes = f.read()
        if not all_bytes.strip():
            logger.error(f"File {dataset_path} contains only whitespace. Returning empty DataFrame.")
            return pd.DataFrame()

    encodings_to_try = [
        "utf-8",
        "latin1",
        "unicode_escape"
        "windows-1252",
    ]

    # --- Try different encodings and delimiters ---
    delimiters = [",", ";", "\t", "|"]
    for enc in encodings_to_try:
        for delim in delimiters:
            try:
                df = pd.read_csv(dataset_path, delimiter=delim, encoding=enc)
                if len(df.columns) > 1:  # sanity check
                    logger.debug(f"Loaded successfully with encoding='{enc}', delimiter='{delim}'")
                    return df
            except Exception as e:
                logger.debug(f"Failed with encoding={enc}, delimiter={delim} ({type(e).__name__})")

    raise ValueError(f"Could not robustly load CSV file: {dataset_path}")


def load_dataframe(file_path, file_format=".csv"):
    assert os.path.isfile(file_path), f"Could not find {file_path} on disk"
    if file_format == '.parquet':
        df = pd.read_parquet(file_path, engine='pyarrow')
    elif file_format == '.csv':
        df = robust_csv_loader(Path(file_path))
    elif file_format == '.df':
        df = pd.read_pickle(file_path)
    return df

def get_input_table(dataset_path: Path, verbose=False):
    input_table = pd.read_csv(dataset_path / "input_table.csv")

    if verbose:
        logger.info(f"Loaded dataset input table with {len(input_table)} rows and {len(input_table.columns)} attributes: {input_table.columns}")
    return input_table

def load_testcase(testcase_path: Path):
    with open(testcase_path, "r") as file:
        testcase_dict = json.load(file)

    testcase_id = testcase_dict["testcase_id"]
    testcase_input_df = pd.read_json(StringIO(testcase_dict["input_row"]), orient="table")
    testcase_gt_output_df = pd.read_json(StringIO(testcase_dict["ground_truth"]), orient="table")

    return testcase_id, testcase_input_df, testcase_gt_output_df

def load_testcase_more_similar_than(testcase_path: Path):
    with open(testcase_path, "r") as file:
        testcase_dict = json.load(file)

    similar_pair = testcase_dict["similar_pair"]
    dissimilar_pair = testcase_dict["dissimilar_pair"]

    assert similar_pair["a"]["qid"] == dissimilar_pair["a"]["qid"]

    return testcase_dict

def load_testcase_paths(dataset_path: Path, limit: int=None):
    """
    Load the paths of all test case JSON files of a given dataset

    Params:
        dataset_path: the pathlib Path to the dataset folder
        limit: int, maximum number of test cases to return (optional)

    Returns:
        list of paths to the JSON test case files
    """
    # make sure that testcases folder exists
    testcases_folder = dataset_path / "test_cases"
    assert testcases_folder.exists()
    testcases_paths = list(testcases_folder.glob("*.json"))
    print(f"Found {len(testcases_paths)} testcases in {dataset_path}, limit is {limit}")

    if limit:
        limit = min(limit, len(testcases_paths))
        testcases_paths = testcases_paths[:limit]

    return testcases_paths

if __name__ == "__main__":
    logging.basicConfig(format='{levelname} - {name} - {asctime} {message}', 
                        level=logging.INFO,
                        style="{",
                        datefmt="%Y-%m-%d %H:%M")
    
    assert DATA_PATH.exists()

    # load input data
    input_table = get_input_table(DATA_PATH)

    testcases_folder = DATA_PATH / "test_cases"
    testcases_paths = list(testcases_folder.glob("*.json"))

    # load a test case
    for testcase_path in testcases_paths[:10]:
        print("-------------------")
        testcase_id, testcase_input_df, testcase_gt_output_df = load_testcase(testcase_path)

        print(testcase_id)
        print(testcase_input_df)
        print("--")
        print(testcase_gt_output_df)
