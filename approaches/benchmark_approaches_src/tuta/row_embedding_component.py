"""
Row embedding component for TUTA.

Row embeddings are derived by mean-pooling the cell embeddings that belong
to each data row.  The header row (index 0 of the cell embedding array) is
excluded — only data rows are returned.

Output shape: (n_data_rows, 768)

Data rows beyond cfg.approach.max_rows receive zero row embeddings because
their cell embeddings are zero (they were not included in the forward pass).
"""

import logging

import numpy as np
import pandas as pd

from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface

logger = logging.getLogger(__name__)


class RowEmbeddingComponent(RowEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame = None, dataset_information: dict = None):
        self.approach_instance.load_model()
        logger.info("TUTA model loaded for row embedding task.")

    def create_row_embeddings_for_table(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Args:
            input_table: DataFrame to embed.

        Returns:
            np.ndarray of shape (n_rows, 768).
        """
        # cell_embs: (n_data_rows + 1, n_cols, 768)
        # Row 0 = header; rows 1+ = data rows
        cell_embs = self.approach_instance.get_cell_embeddings(input_table)

        # Data rows only (skip header row 0), mean across columns
        data_cell_embs = cell_embs[1:]          # (n_data_rows, n_cols, 768)
        row_embs = data_cell_embs.mean(axis=1)  # (n_data_rows, 768)

        logger.info(f"TUTA row embeddings: {row_embs.shape}")
        return row_embs
