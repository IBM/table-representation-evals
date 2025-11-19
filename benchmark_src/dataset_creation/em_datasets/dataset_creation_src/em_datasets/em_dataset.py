from omegaconf import DictConfig
import pandas as pd
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("__name__")

import utils

#### Interface class for all Entity Matching Datasets

class EM_Dataset():

    def __init__(self, cfg):
        self.cfg = cfg
        pass

    def load_data(self):
        """
        Check if the dataset folder exists in raw datasets folder, if not download the dataset
        Load the input table data into a pandas Dataframe

        returns:
            pd.Dataframe:   the input table of the dataset
        """
        pass

    def custom_modifications(self, input_table: pd.DataFrame):
        """
        Adapt the input table for the benchmark.

        returns:
            pd.Dataframe:   the modified input table
        """
        pass

    def get_possible_testcases(self, input_table: pd.DataFrame):
        """
        Load all testcases, thereby make sure the row ids really exist in the input table

        returns: 
            list: the testcases
            list: the ids of rows involved in testcases
        """
        pass

    def limit_data(self, input_table, testcase_row_ids, testcases):
        """
        Limits the input table and testcases to a given row limit.
        Check if length of input table exceeds the row limit.
        """
        pass

    def save_input_table(self, input_table_df: pd.DataFrame):
        """
        Save the input table to disk, delete all information that should not be given to the models
        """
        pass

    def save_testcases(self, testcases):
        """
        Save the testcases to dict
        # TODO: create class for testcases
        """
        pass

    def prepare_data(self):
        logger.info(f"Preparing {self.cfg.dataset.dataset_name} dataset")
        
        # Download the dataset if necessary and load the input table
        original_data_df = self.load_data()

        # Add custom modifications for benchmark (maybe do this after limit...)
        input_table = self.custom_modifications(original_data_df)

        # Get testcases
        testcases, testcase_row_ids = self.get_possible_testcases(original_data_df)
        
        # Limit data if necessary (Decide how to go about limiting and modifications...)
        input_table, testcases = self.limit_data(input_table=original_data_df, testcase_row_ids=testcase_row_ids, testcases=testcases)
        
        # Save input table to disk
        input_data_df = self.save_input_table(input_table)

        # Save testcases to disk
        num_testcases = self.save_testcases(testcases=testcases)

        utils.create_statistics(dataset_name=self.cfg.dataset.dataset_name,
                                        input_table_df=input_data_df,
                                        num_testcases=num_testcases,
                                        save_path=self.output_path,
                                        primary_key_column=self.cfg.dataset.primary_key_column)

        logging.info(f"Created {num_testcases} test cases")
