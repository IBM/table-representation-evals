import pandas as pd
import numpy as np
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface


class ColumnEmbeddingComponent(ColumnEmbeddingInterface):
    """
    Column embedding component for TaBERT.

    Delegates to TaBertEmbedder.get_column_embeddings, which encodes the
    full table once and returns TaBERT's per-column representations.
    """

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        self.approach_instance.load_trained_model()

    def create_column_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Returns:
            Tuple (column_embeddings, column_names) where column_embeddings has
            shape (num_columns, embedding_dim).
        """
        # Limit rows fed to the model for efficiency
        if len(input_table) > 1000:
            input_table = input_table.head(
                100 if len(input_table.columns) > 500 else 1000
            )

        return self.approach_instance.get_column_embeddings(input_table)
