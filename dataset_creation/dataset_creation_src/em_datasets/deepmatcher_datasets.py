import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
import logging
import json
import shutil

from em_datasets.em_dataset import EM_Dataset
import utils

logger = logging.getLogger(__name__)

creation_random = np.random.RandomState(48573948)

class DeepmatcherDataset(EM_Dataset):

    def __init__(self, cfg, dataset_name):
        self.cfg = cfg
        self.output_path = Path(cfg.output_dir) / dataset_name
        self.dataset_name = dataset_name
        super().__init__(cfg=cfg)

    def new_id_A(self, row):
        id_value = "A_" + str(row["id"])
        return id_value

    def new_id_B(self, row):
        id_value = "B_" + str(row["id"])
        return id_value

    def merge_tables(self, table_A_df: pd.DataFrame, table_B_df: pd.DataFrame):
        """
        Merges table A and table B into one joined table

        IDs of table B are appended to table A, original ids are saved with prefix A_ or B_ in "original_id" column
        """
        # add original id column
        table_A_df["original_id"] = table_A_df.apply(self.new_id_A, axis=1)
        table_B_df["original_id"] = table_B_df.apply(self.new_id_B, axis=1)

        table_A_df.drop("id", axis=1, inplace=True)
        table_B_df.drop("id", axis=1, inplace=True)

        merged_df = pd.concat([table_A_df, table_B_df], ignore_index=True, sort=False)
        merged_df["id"] = merged_df.index

        # bring id column back to the front
        column_list = list(merged_df.columns)
        column_list.insert(0, column_list.pop(-1))
        merged_df = merged_df[column_list]

        return merged_df
    
    def load_data(self):
        """
        Check if the dataset folder exists in raw datasets folder, if not download the dataset
        Load the input table data into a pandas Dataframe

        returns:
            pd.Dataframe:   the input table of the dataset
        """
        raw_datasets_path = Path(self.cfg['raw_datasets_dir'])

        dataset_path = raw_datasets_path / "Structured" / self.dataset_name
        self.dataset_path = dataset_path

        if dataset_path.exists():
            logger.info("Dataset was already downloaded")
        else:
            logger.info("Now downloading dataset")
            utils.download_url(url=self.cfg["dataset"]["url"], path=raw_datasets_path, unzip=True)

        Path.mkdir(self.output_path, exist_ok=False)
        table_A_df = pd.read_csv(dataset_path / "tableA.csv")
        table_B_df = pd.read_csv(dataset_path / "tableB.csv")

        # merge tables to perpare input table
        merged_table_df = self.merge_tables(table_A_df, table_B_df)

        return merged_table_df
    
    def remove_percent(self, text):
        text = text.replace("%", "").strip()
        if text == "-":
            text = np.nan
        return float(text)
    
    def remove_dollar(self, text):
        text = text.replace("$", "").strip()
        if text in ["Album Only", "Pre-Order Only"]:
            text = np.nan
        if text == "FREE":
            text = 0
        try:
            text = float(text) 
        except:
            print(text)
            text = float(np.nan)
        return text

    def custom_modifications(self, input_table: pd.DataFrame):
        """
        Adapt the input table for the benchmark.

        returns:
            pd.Dataframe:   the modified input table
        """
        if self.dataset_name == "Beer":
            # remove " %" from ABV values and convert them to floats
            input_table["ABV"] = input_table["ABV"].apply(self.remove_percent)
        elif self.dataset_name == "iTunes-Amazon":
            # remove $ from price values and convert them to floats
            input_table["Price"] = input_table["Price"].apply(self.remove_dollar)
        return input_table

    def get_possible_testcases(self, input_table: pd.DataFrame):
        """
        Load all testcases, thereby make sure the row ids really exist in the input table

        returns: 
            list: the testcases
            list: the ids of rows involved in testcases
        """
        gt_matches_df = pd.read_csv(self.dataset_path / "test.csv")

        # take only matches (where label column is 1)
        gt_matches_df = gt_matches_df[gt_matches_df["label"] == 1]

        # collect ids and testcases and make sure all ids are present in the input table
        testcases = []
        testcase_row_ids = []
        for match in gt_matches_df.itertuples():
            table_A_id = "A_" + str(match.ltable_id)
            table_B_id = "B_" + str(match.rtable_id)

            if table_A_id in list(input_table["original_id"]) and table_B_id in list(input_table["original_id"]):
                testcases.append((table_A_id, table_B_id))
                testcase_row_ids.append(table_A_id)
                testcase_row_ids.append(table_B_id)

        return testcases, testcase_row_ids

    def limit_data(self, input_table, testcase_row_ids, testcases):
        """
        Limits the input table and testcases to a given row limit.
        Check if length of input table exceeds the row limit.
        """
        print(self.cfg.table_row_limit)
        if self.cfg.table_row_limit and len(input_table) > self.cfg.table_row_limit:
            logger.info(f"Need to limit table rows from {len(input_table)} to {self.cfg.table_row_limit}")
            limited_testcase_row_ids = []
            limited_testcases = []
            if len(testcase_row_ids) > self.cfg.table_row_limit:
                logger.info(f"Need to also limit the number of testcases, currently have {len(testcases)} with {len(testcase_row_ids)} rows involved")
                for id_list in testcases:
                    if len(limited_testcase_row_ids) + len(id_list) < self.cfg.table_row_limit:
                        limited_testcase_row_ids.extend(id_list)
                        limited_testcases.append(id_list)
            else:
                logger.info(f"Do not need to limit the number of testcases, have {len(testcases)} with {len(testcase_row_ids)} rows involved")
                limited_testcase_row_ids = testcase_row_ids
                limited_testcases = testcases
            logger.info(f"Going to use {len(limited_testcases)} testcases with {len(limited_testcase_row_ids)} involved rows")
            
            rows_from_testcases = input_table[input_table["original_id"].isin(limited_testcase_row_ids)]
            rest_of_dataframe = input_table[~input_table["original_id"].isin(limited_testcase_row_ids)]
            additional_rows = rest_of_dataframe.sample(self.cfg.table_row_limit-len(rows_from_testcases), random_state=creation_random)

            input_table = pd.concat([rows_from_testcases, additional_rows])

            testcases = limited_testcases
        else:
            logger.info(f"Limiting not necessary")

        self.input_table = input_table
        return input_table, testcases

    def save_input_table(self, input_table_df: pd.DataFrame):
        """
        Save the input table to disk, delete all information that should not be given to the models

        Deletes the original_id column from the table.
        """
        input_data_df = input_table_df.drop("original_id", axis=1)

        input_data_df.to_csv(self.output_path / "input_table.csv", index=False)
        logger.info(f"Created input table in {self.output_path} with {len(input_data_df)} rows")

        return input_data_df

    def save_testcases(self, testcases):
        """
        Save the testcases to dict
        # TODO: create class for testcases
        """     
        testcase_dir = self.output_path / "test_cases"
        Path.mkdir(testcase_dir, exist_ok=False)
        test_case_id = 0


        # create one test case for every match
        for match in testcases:
            test_case = {"testcase_id": test_case_id, 
                    "dataset": self.cfg["dataset"]["dataset_name"], 
                    }
            
            input_row = self.input_table[self.input_table["original_id"]==match[0]]
            ground_truth_row = self.input_table[self.input_table["original_id"]==match[1]]

            assert len(input_row) == 1, f"Could not find input row {match[0]}"
            assert len(ground_truth_row) == 1, f"Could not find gt row {match[1]}"

            input_row = input_row.drop("original_id", axis=1)
            ground_truth_row = ground_truth_row.drop("original_id", axis=1)

            test_case["input_row"] = input_row.to_json(orient="table", index=False)
            test_case["ground_truth"] = ground_truth_row.to_json(orient="table", index=False)

            with open(self.output_path / "test_cases" / f"{test_case_id}.json", "w", encoding="utf8") as file:
                json.dump(test_case, file, indent=2)

            test_case_id += 1

        return test_case_id


def create_datasets(cfg: DictConfig):
    for dataset_name in cfg["dataset"]["sub_datasets"]:
        logger.info(f"Sub-Dataset: {dataset_name}")
        dataset = DeepmatcherDataset(cfg=cfg, dataset_name=dataset_name)
        dataset.prepare_data()

