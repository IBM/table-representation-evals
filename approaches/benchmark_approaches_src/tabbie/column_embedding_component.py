import numpy as np
import pandas as pd
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface


class ColumnEmbeddingComponent(ColumnEmbeddingInterface):
    """
    Column embedding component for TABBIE.

    Delegates to TABBIEEmbedder.get_column_embeddings(), which extracts the
    column-CLS token (position col_embs[0, j+1, :]) from the TABBIE column
    transformer output.
    """

    def __init__(self, approach_instance) -> None:
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        self.approach_instance.load_trained_model()

    def create_column_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Returns:
            Tuple (column_embeddings, column_names)
            column_embeddings: np.ndarray of shape (num_cols, 768)
        """
        # For very wide tables, limit rows passed to control memory
        if len(input_table) > 1000:
            input_table = input_table.head(
                100 if len(input_table.columns) > 500 else 1000
            )
        return self.approach_instance.get_column_embeddings(input_table)
