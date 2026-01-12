import pandas as pd
import numpy as np
from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface

class RowEmbeddingComponent(RowEmbeddingInterface):
    """
    Row embedding component for TabuLA-8B approach.
    
    This component handles the creation of row embeddings for tabular data
    using the TabuLA-8B model.
    """

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        """
        Setup the TabuLA-8B model for the row embedding task.
        
        Args:
            input_table: pd.DataFrame - The table to work with
            dataset_information: dict - Additional information about the dataset
        """
        # Load the model if not already loaded
        print(f"Setting up TabuLA-8B model for row embedding task...")
        if self.approach_instance.model is None:
            self.approach_instance.load_trained_model()

    def create_row_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Create embeddings for each row of the given table using TabuLA-8B.
        
        Args:
            input_table: pd.DataFrame - The table to work with
            
        Returns:
            np.ndarray: Matrix of row embeddings with shape [#rows, embedding_dimension]
        """
        # Preprocess the table rows
        print(f"Starting preprocessing for row embeddings for input table with shape {input_table.shape}...")
        preprocessed_rows = self.approach_instance.preprocessing(input_table)
        
        # Generate embeddings using TabuLA-8B
        print(f"Done with the preprocessing. Generating row embeddings for {len(preprocessed_rows)} rows...")
        row_embeddings = self.approach_instance.get_embeddings(preprocessed_rows)
        
        return row_embeddings 