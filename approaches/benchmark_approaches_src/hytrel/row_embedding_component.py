import pandas as pd
import numpy as np
from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface

class RowEmbeddingComponent(RowEmbeddingInterface):
    """
    Row embedding component for HyTrel approach.
    Uses the HyTrel model to generate a single embedding per row.
    """
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        # Load the model if not already loaded
        self.approach_instance.load_trained_model()

    def create_row_embeddings_for_table(self, input_table: pd.DataFrame, train_size=None, train_labels=None):
        # Get single row embeddings using the approach instance
        # Pass train_size to enable test-only embedding extraction when provided
        return self.approach_instance.get_row_embeddings(input_table, train_size=train_size)
