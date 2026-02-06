import pandas as pd
import logging

from benchmark_src.approach_interfaces.cell_embedding_interface import CellEmbeddingInterface

logger = logging.getLogger(__name__)

### Implement this component if your approach is able to provide cell embeddings for a given table in a self-supervised way (no labels).
### Otherwise, just delete this file.



class CellEmbeddingComponent(CellEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance  # with the approach instance you can call functions implemented in your CustomTabularEmbeddingApproach class 

        
    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        """
        Please implement any steps you need to train/setup/load your model in order to later produce cell embeddings of the given table.

            Args:
                input_table: pd.DataFrame   The table to work with
                dataset_information: dict   Additional information about the dataset, look into dataset for details
        """
        pass

    def create_cell_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Create an embedding for each cell of the given table and return it as a numpy array.

            Args:
                input_table: pd.DataFrame   The table to work with

            Returns: 
                np.ndarray: the matrix of the row embeddings with shape [#row+1 (header), #column, embedding_dimension]
        """
        logger.info(f"Embedding the {input_table.size} cells + {len(input_table.column)} headers of the given table")
        # Implement to embed every cell of the given dataframe
        cell_embeddings = None
        # return a numpy array with shape [#row+1 (header), #column, embedding_dimension]
        return cell_embeddings
        
