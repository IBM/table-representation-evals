import pandas as pd
from abc import ABC, abstractmethod

class CellEmbeddingInterface(ABC):

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
    def create_cell_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Create an embedding for each cell of the given table and return it as a numpy array.

            Args:
                input_table: pd.DataFrame   The table to work with

            Returns: 
                np.ndarray: the matrix of the row embeddings with shape [#row, #column, embedding_dimension]
        """
        # Implement to embed every cell of the given dataframe
        cell_embeddings = None
        return cell_embeddings
        
