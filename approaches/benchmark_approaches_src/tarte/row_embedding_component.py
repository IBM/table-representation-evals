from typing import Optional

import numpy as np
import pandas as pd

from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface


class RowEmbeddingComponent(RowEmbeddingInterface):
    """Row embedding component for TARTE."""

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        self.approach_instance.load_trained_model()

    def create_row_embeddings_for_table(
        self,
        input_table: pd.DataFrame,
        train_size: Optional[int] = None,
        train_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.approach_instance.get_row_embeddings(
            input_table, train_size, train_labels
        )
