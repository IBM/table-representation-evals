import pandas as pd
from abc import ABC, abstractmethod

class RowEmbeddingInterface(ABC):

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
    def create_row_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Create an embedding for each row of the given table and return it as a numpy array.

            Args:
                input_table: pd.DataFrame   The table to work with

            Returns: 
                np.ndarray: the matrix of the row embeddings with shape [#rows, embedding_dimension]
        """
        # Implement to embed every row of the given dataframe
        row_embeddings = None 
        # If your approach does not create row embeddings, just remove this method.
        return row_embeddings
        
