import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
import json 
import logging
import shutil

from em_datasets.em_dataset import EM_Dataset

import utils

logger = logging.getLogger(__name__)

DATASET_FOLDERNAME = "geographicalSettelments" # Typo already in original foldername
SAVE_PATH = Path("geological-settlements")

creation_random = np.random.RandomState(48573948)

class GeologicalSettlementsDataset(EM_Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        self.output_path = cfg.output_dir / SAVE_PATH
        self.DATASET_FOLDERNAME = DATASET_FOLDERNAME

        super().__init__(cfg=cfg)

    def load_data(self):
        """
        Check if the dataset folder exists in raw datasets folder, if not download the dataset
        Load the input table data into a pandas Dataframe

        returns:
            pd.Dataframe:   the input table of the dataset
        """
        # check if dataset folder exists, otherwise download
        raw_datasets_path = Path(self.cfg['raw_datasets_dir'])
        self.dataset_path = raw_datasets_path / self.DATASET_FOLDERNAME

        if self.dataset_path.exists():
            logging.info("Dataset was already downloaded")
        else:
            logging.info("Now downloading dataset")
            utils.download_url(url=self.cfg["dataset"]["url"], path=raw_datasets_path, unzip=True)

        Path.mkdir(self.output_path, exist_ok=True)

        input_data_path = self.dataset_path / "settlements.json"

        input_data_dict = {}
        with open(input_data_path, "r") as file:
            for line in file:
                datapoint = json.loads(line)
                assert datapoint["id"] not in input_data_dict.keys()
                input_data_dict[datapoint["id"]] = datapoint["data"]
        
        # prepare input table
        # Transform the json data into a dataframe and save it as csv
        all_datapoints = []
        for idx, datapoint in input_data_dict.items():
            processed_datapoint = {"id": idx}
            for col_name, value in datapoint.items():
                if isinstance(value, list):
                    selected_value = str(creation_random.choice(value))
                else:
                    selected_value = str(value)
                processed_datapoint[col_name] = selected_value
            all_datapoints.append(processed_datapoint)

        input_data_df = pd.DataFrame(all_datapoints)



        return input_data_df

    def to_float(self, text):
        return float(text)

    def custom_modifications(self, input_table: pd.DataFrame):
        """
        Adapt the input table for the benchmark.

        Converting lat, lon and ele columns to floats instead of strings

        returns:
            pd.Dataframe:   the modified input table

        """
        input_table["lat"] = input_table["lat"].apply(self.to_float)
        input_table["lon"] = input_table["lon"].apply(self.to_float)
        input_table["ele"] = input_table["ele"].apply(self.to_float)
        return input_table
    
    def get_possible_testcases(self, input_table):
        """
        Load all testcases, thereby make sure the row ids really exist in the input table

        returns: 
            list: the testcases
            list: the ids of rows involved in testcases
        """
        gt_data_pth = self.dataset_path / "combinedSettlements(PerfectMatch).json"

        gt_data_list = []
        with open(gt_data_pth, "r") as file:
            for line in file:
                gt_data_list.append(json.loads(line))

        involved_row_ids = []
        testcases = []
        for cluster_info in gt_data_list:
            # is a list of IDS
            datapoints_of_cluster = cluster_info["data"]["clusteredVertices"]
            # make sure that datapoints of cluster are included in the original table
            for row_id in datapoints_of_cluster:
                if not row_id in list(input_table["id"]):
                    datapoints_of_cluster.remove(row_id)
            if len(datapoints_of_cluster) > 1:
                testcases.append(datapoints_of_cluster)
                involved_row_ids.extend(datapoints_of_cluster)

        return testcases, involved_row_ids

    def limit_data(self, input_table, testcase_row_ids, testcases):
        """
        Limits the input table and testcases to a given row limit.
        Check if length of input table exceeds the row limit.
        """
        if self.cfg.table_row_limit and len(input_table) > self.cfg.table_row_limit:
            print(f"Need to limit table rows from {len(input_table)} to {self.cfg.table_row_limit}")
            limited_testcase_row_ids = []
            limited_testcases = []
            if len(testcase_row_ids) > self.cfg.table_row_limit:
                print(f"Need to also limit the number of testcases, currently have {len(testcases)} with {len(testcase_row_ids)} rows involved")
                for id_list in testcases:
                    if len(limited_testcase_row_ids) + len(id_list) < self.cfg.table_row_limit:
                        limited_testcase_row_ids.extend(id_list)
                        limited_testcases.append(id_list)
            else:
                print(f"Do not need to limit the number of testcases, have {len(testcases)} with {len(testcase_row_ids)} rows involved")
                limited_testcase_row_ids = testcase_row_ids
                limited_testcases = testcases
            print(f"Going to use {len(limited_testcases)} testcases with {len(limited_testcase_row_ids)} involved rows")
            
            rows_from_testcases = input_table[input_table["id"].isin(limited_testcase_row_ids)]
            #print(f"Found the {len(rows_from_testcases)} rows")
            rest_of_dataframe = input_table[~input_table["id"].isin(limited_testcase_row_ids)]
            #print(f"Rest of dataframe has {len(rest_of_dataframe)} rows")
            additional_rows = rest_of_dataframe.sample(self.cfg.table_row_limit-len(rows_from_testcases), random_state=creation_random)

            input_table = pd.concat([rows_from_testcases, additional_rows])
            print(f"Length of input table after limiting:", len(input_table))

            testcases = limited_testcases
        else:
            logger.info(f"Limiting not necessary")

        self.input_table = input_table
        return input_table, testcases

    def save_input_table(self, input_table_df: pd.DataFrame):
        """
        Save the input table to disk, delete all information that should not be given to the models
        """
        input_table_df.to_csv(Path(self.output_path) / "input_table.csv", index=False)
        logger.info(f"Created input table in {self.output_path}")

        return input_table_df

    def save_testcases(self, testcases):
        """
        Save the testcases to dict
        # TODO: create class for testcases
        """
        testcase_dir = self.output_path / "test_cases"

        Path.mkdir(testcase_dir, exist_ok=True)
        test_case_id = 0

        for row_ids in testcases:
            test_case = {"testcase_id": test_case_id, 
                    "dataset": self.cfg["dataset"]["dataset_name"], 
                    }

            cluster_df = self.input_table.loc[self.input_table["id"].isin(row_ids)]
            assert len(cluster_df) == len(row_ids)

            # randomly pick one row of the cluster to be the input row
            random_index = creation_random.choice(cluster_df.index)
            input_row = cluster_df.loc[random_index].to_frame().transpose()

            # serialize the remaining rows as the GT solution
            remaining_gt_rows = cluster_df.drop(random_index)

            test_case["input_row"] = input_row.to_json(orient="table", index=False)
            test_case["ground_truth"] = remaining_gt_rows.to_json(orient="table", index=False)


            with open(testcase_dir / f"{test_case_id}.json", "w", encoding="utf8") as file:
                json.dump(test_case, file, indent=2)

            test_case_id += 1

        return test_case_id


    