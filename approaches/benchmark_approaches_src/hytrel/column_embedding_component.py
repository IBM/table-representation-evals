import pandas as pd
import numpy as np
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface

class ColumnEmbeddingComponent(ColumnEmbeddingInterface):
    """
    Column embedding component for HyTrel approach.
    Uses HyTrel's hypergraph structure to generate embeddings for each column.
    """
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        # Load the model if not already loaded
        self.approach_instance.load_trained_model()

    def create_column_embeddings_for_table(self, input_table: pd.DataFrame):
        # Get column embeddings using the approach instance
        return self.approach_instance.get_column_embeddings(input_table)

