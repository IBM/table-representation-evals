import numpy as np
import pandas as pd
from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface


class RowEmbeddingComponent(RowEmbeddingInterface):
    """
    Row embedding component for TABBIE.

    Delegates to TABBIEEmbedder.get_row_embeddings(), which extracts the
    per-row CLS token from the TABBIE row transformer output.
    """

    def __init__(self, approach_instance) -> None:
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        self.approach_instance.load_trained_model()

    def create_row_embeddings_for_table(
        self,
        input_table: pd.DataFrame,
        train_size=None,
        train_labels=None,
    ) -> np.ndarray:
        """
        Args:
            input_table: DataFrame to embed.
            train_size:  If provided, only test-row embeddings (after index
                         train_size) are returned.

        Returns:
            np.ndarray of shape (num_rows, 768) or (num_test_rows, 768).
        """
        embeddings = self.approach_instance.get_row_embeddings(input_table)
        if train_size is not None:
            embeddings = embeddings[train_size:]
        return embeddings
