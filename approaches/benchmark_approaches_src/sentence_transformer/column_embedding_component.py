import pandas as pd
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface

### Implement this component if your approach is able to provide row embeddings for a given table in a self-supervised way (no labels).
### Otherwise, just delete this file.


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

    def create_column_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Create an embedding for each row of the given table and return it as a numpy array.

            Args:
                input_table: pd.DataFrame   The table to work with

            Returns: 
                np.ndarray: the matrix of the row embeddings with shape [#rows, embedding_dimension]
        """
        all_columns = self.approach_instance.preprocessing(input_table=input_table, component=self)
        # encode the rows
        column_embeddings = self.approach_instance.model.encode(all_columns, show_progress_bar=True)
        return column_embeddings