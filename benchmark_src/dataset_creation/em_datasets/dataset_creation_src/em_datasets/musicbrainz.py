import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import json 
from omegaconf import DictConfig
import logging
import shutil

from em_datasets.em_dataset import EM_Dataset

import utils

logger = logging.getLogger(__name__)

DATASET_FILENAME = "musicbrainz-20-A01.csv.dapo"
SAVE_PATH = Path("MusicBrainz")

creation_random = np.random.RandomState(48573948)

class MusicBrainzDataset(EM_Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        self.output_path = cfg.output_dir / SAVE_PATH
        self.DATASET_FILENAME = DATASET_FILENAME
        
        super().__init__(cfg=cfg)

    def save_input_table(self, original_data_df: pd.DataFrame):
        """
        Delete information about clustering from the table 
        CID: Cluster ID
        CTID: ID of the entity within the cluster
        """
        input_data_df = original_data_df.copy()
        input_data_df = input_data_df.drop("CID", axis=1)
        input_data_df = input_data_df.drop("CTID", axis=1)

        # Quickfix: ai-db cannot deal with invisible spaces
        input_data_df = input_data_df.replace(r'\u3000', ' ', regex=True)

        input_data_df.to_csv(self.output_path / "input_table.csv", index=False)
        logger.info(f"Saved input table with {len(input_data_df)} rows in {self.output_path}")

        return input_data_df

    def check_for_different_track_numbers_in_clusters(self, clusters_df):
        clusters_num_trackids = []

        for _, frame in clusters_df:
            # only create test cases for clusters that have more than one element:
            if len(frame) > 1:
                trackid_num = len(set(frame['number']))
                clusters_num_trackids.append(trackid_num)

        print("Number of different track numbers in clusters:")
        print(Counter(clusters_num_trackids))

    def load_data(self):
        """
        Check if the dataset folder exists in raw datasets folder, if not download the dataset
        Load the input table data into a pandas Dataframe

        returns:
            pd.Dataframe:   the input table of the dataset
        """
        raw_datasets_path = Path(self.cfg['raw_datasets_dir'])
        dataset_path = raw_datasets_path / DATASET_FILENAME

        if dataset_path.exists():
            logging.info("Dataset was already downloaded")
        else:
            logging.info("Now downloading dataset")
            utils.download_url(url=self.cfg["dataset"]["url"], path=raw_datasets_path)

        Path.mkdir(self.output_path, exist_ok=True)
        
        original_data_df = pd.read_csv(dataset_path)

        return original_data_df
        
        
    def custom_modifications(self, input_table: pd.DataFrame):
        """
        Adapt the input table for the benchmark.

        TODO: write custom modifications here

        returns:
            pd.Dataframe:   the modified input table

        """
        #TODO: add custom modifications
        return input_table

    def get_possible_testcases(self, input_table):
        """
        Load all testcases, thereby make sure the row ids really exist in the input table

        returns: 
            list: the testcases
            list: the ids of rows involved in testcases
        """
        clusters_df = input_table.groupby("CID")

        # get number of rows
        involved_row_ids = []
        testcases = []
        for cluster_id, frame in clusters_df:
            if len(frame) > 1:
                involved_row_ids.extend(list(frame["TID"]))
                testcases.append(frame)

        return testcases, involved_row_ids

    def limit_data(self, input_table, testcase_row_ids, testcases):
        if self.cfg.table_row_limit and len(input_table) > self.cfg.table_row_limit:
            print(f"Need to limit table rows from {len(input_table)} to {self.cfg.table_row_limit}")
            limited_testcase_row_ids = []
            limited_testcases = []
            if len(testcase_row_ids) > self.cfg.table_row_limit:
                print(f"Need to also limit the number of testcases, currently have {len(testcases)} with {len(testcase_row_ids)} rows involved")
                for frame in testcases:
                    if len(limited_testcase_row_ids) + len(frame) < self.cfg.table_row_limit:
                        limited_testcase_row_ids.extend(list(frame["TID"]))
                        limited_testcases.append(frame)
            else:
                limited_testcase_row_ids = testcase_row_ids
                limited_testcases = testcases
            print(f"Going to use {len(limited_testcases)} testcases with {len(limited_testcase_row_ids)} involved rows")
            
            rows_from_testcases = input_table[input_table["TID"].isin(limited_testcase_row_ids)]
            print(f"Found the {len(rows_from_testcases)} rows")
            rest_of_dataframe = input_table[~input_table["TID"].isin(limited_testcase_row_ids)]
            #print(f"Rest of dataframe has {len(rest_of_dataframe)} rows")
            additional_rows = rest_of_dataframe.sample(self.cfg.table_row_limit-len(rows_from_testcases), random_state=creation_random)

            input_table = pd.concat([rows_from_testcases, additional_rows])
            print(len(input_table), input_table.columns)

            for id in limited_testcase_row_ids:
                if id not in list(input_table["TID"]):
                    print(f"Could not find {id}")

            return input_table, limited_testcases
        else:
            return input_table, testcases

    def save_testcases(self, testcases):
        testcase_dir = self.output_path / "test_cases"

        Path.mkdir(testcase_dir, exist_ok=True)

        logger.info("Creating test cases")

        test_case_id = 0
        for frame in testcases:
            # only create test cases for clusters that have more than one element:
            if len(frame) > 1:
                test_case = {"testcase_id": test_case_id, 
                            "dataset": self.cfg["dataset"]["dataset_name"], 
                            "CID": int(frame["CID"].iloc[0])}

                frame = frame.drop("CID", axis=1)
                frame = frame.drop("CTID", axis=1)

                # randomly pick one row of the cluster to be the input row
                random_index = creation_random.choice(frame.index)
                input_row = frame.loc[random_index].to_frame().transpose()

                # serialize the remaining rows as the GT solution
                remaining_gt_rows = frame.drop(random_index)

                test_case["input_row"] = input_row.to_json(orient="table", index=False)
                test_case["ground_truth"] = remaining_gt_rows.to_json(orient="table", index=False)

                with open(testcase_dir / f"{test_case_id}.json", "w", encoding="utf8") as file:
                    json.dump(test_case, file, indent=2)

                test_case_id += 1
                #print(test_case)
        return test_case_id
                



                

        