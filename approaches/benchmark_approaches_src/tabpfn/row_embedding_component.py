import pandas as pd
import numpy as np
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
        # Load the model if not already loaded
        #self.approach_instance.load_trained_model()
        pass

    def create_row_embeddings_for_table(self, input_table: pd.DataFrame, train_size: int = None, train_labels: np.ndarray = None):
        """
        Create embeddings for each row of the given table.
        
        Args:
            input_table (pd.DataFrame): Input table to extract embeddings from
            train_size (int, optional): Number of rows to use for training
            train_labels (np.ndarray, optional): Labels for training rows
            
        Returns:
            np.ndarray: Row embeddings
        """
        # Delegate to the approach instance with train_size and train_labels
        return self.approach_instance.get_row_embeddings(input_table, train_size=train_size, train_labels=train_labels)