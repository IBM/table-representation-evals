import pandas as pd
import numpy as np
from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface


class RowEmbeddingComponent(RowEmbeddingInterface):
    """
    Row embedding component for TaBERT.

    Delegates to TaBertEmbedder.get_row_embeddings, which encodes each row
    independently using TaBERT and returns the mean of the per-column encodings.
    """

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        self.approach_instance.load_trained_model()

    def create_row_embeddings_for_table(
        self, input_table: pd.DataFrame, train_size=None, train_labels=None
    ) -> np.ndarray:
        """
        Args:
            input_table: DataFrame whose rows are to be embedded.
            train_size: If provided, *input_table* contains train + test rows
                        and only test-row embeddings (after index train_size) are returned.

        Returns:
            np.ndarray of shape (num_rows, embedding_dim) or
            (num_test_rows, embedding_dim) when train_size is given.
        """
        return self.approach_instance.get_row_embeddings(input_table, train_size=train_size)
