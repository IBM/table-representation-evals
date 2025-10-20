import pandas as pd
import numpy as np
from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface

class RowEmbeddingComponent(RowEmbeddingInterface):
    """
    Row embedding component for TabICL approach.
    Uses the TabICL model to generate a single embedding per row.
    """
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        # Load the model if not already loaded
        self.approach_instance.load_trained_model()

    def create_row_embeddings_for_table(self, input_table: pd.DataFrame):
        # Get single row embeddings using the approach instance
        return self.approach_instance.get_row_embeddings(input_table) 