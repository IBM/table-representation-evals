import pandas as pd
from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface

class RowEmbeddingComponent(RowEmbeddingInterface):
    """
    Row embedding component for TabPFN approach.
    Delegates row embedding tasks to the approach instance.
    """
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        """
        Setup the TabPFN model for row embedding tasks.
        """
        # Load the model (this will be done when get_row_embeddings is called)
        pass

    def create_row_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Create embeddings for each row of the given table.
        """
        # Delegate to the approach instance
        return self.approach_instance.get_row_embeddings(input_table) 