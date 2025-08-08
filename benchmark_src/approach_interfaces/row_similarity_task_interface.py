import pandas as pd
from abc import ABC, abstractmethod

class RowSimilaritySearchInterface(ABC):

    def __init__(self, approach_instance):
        pass

    @abstractmethod
    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        """
        Please implement any steps you need to train/setup/load your model in order to later produce row embeddings of the given table.

            Args:
                input_table: pd.DataFrame   The table to work with
                dataset_information: dict   Additional information about the dataset, look into dataset for details

        """
        pass

    @abstractmethod
    def custom_row_similarity_search(self, input_table: pd.DataFrame, input_row, k: int):
        """
        Optional, for approaches that come with their own implemented similarity search approach to find the most similar rows given the table.
        
            Args:
                input_table: pd.DataFrame of the table to embed
                input_row: pd.Series, the given row to search the most similar row(s) for
                k: the number of rows to return (top-k) that are found to be  most similar to the input_row

            Returns:
                a ranked list of k row ids, the first one is the most similar row (decreasing positions)
                (the primary key column can be obtained using the dataset_information provided during preprocessing)
        """
        # Call your custom row_similarity_serach method from here
        ranked_list = []
        return ranked_list