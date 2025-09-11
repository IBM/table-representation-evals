import json
import pandas as pd
from pathlib import Path
import logging
from io import StringIO

DATA_PATH = Path("data/MusicBrainz")

logger = logging.getLogger(__name__)


def get_input_table(dataset_path: Path):
    input_table = pd.read_csv(dataset_path / "input_table.csv")

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
