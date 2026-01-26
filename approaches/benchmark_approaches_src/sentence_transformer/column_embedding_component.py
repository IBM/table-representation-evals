import pandas as pd
import logging
import torch
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface

### Implement this component if your approach is able to provide row embeddings for a given table in a self-supervised way (no labels).
### Otherwise, just delete this file.

logger = logging.getLogger(__name__)
class ColumnEmbeddingComponent(ColumnEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance # with the approach instance you can call functions implemented in your CustomTabularEmbeddingApproach class

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        """
        Please implement any steps you need to train/setup/load your model in order to later produce row embeddings of the given table.

            Args:
                input_table: pd.DataFrame   The table to work with
                dataset_information: dict   Additional information about the dataset, look into dataset for details
        """
        self.approach_instance.load_trained_model()

    def create_column_embeddings_for_table(self, input_table: str):
        """
        Create an embedding for each column of the given table and return it as a numpy array.

            Args:
                input_table: pd.DataFrame   The table to work with

            Returns: 
                np.ndarray: the matrix of the row embeddings with shape [#rows, embedding_dimension]
        """
         # limit table size to first 1000 rows for efficiency
        if len(input_table) > 1000:
            # if there are more than 500 columns, take only the first 100 rows:
            if len(input_table.columns) > 500:
                input_table = input_table.head(100)
            else:
                input_table = input_table.head(1000)
                
        logger.debug("starting preprocessing for column embedding generation")
        all_columns = self.approach_instance.preprocessing(input_table=input_table, component=self)
        logger.debug(f"finished preprocessing for column embedding generation, found {len(all_columns)} columns, starting encoding now")
        # encode the columns
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"PyTorch sees device: {device}")
        logger.debug(f"Model is on device: {next(self.approach_instance.model.parameters()).device}")
        column_embeddings = self.approach_instance.model.encode(list(all_columns.values()), show_progress_bar=True)
        logger.debug("finished encoding")
        return column_embeddings, list(all_columns.keys())