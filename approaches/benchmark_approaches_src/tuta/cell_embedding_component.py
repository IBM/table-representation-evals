"""
Cell embedding component for TUTA.

Output shape: (n_data_rows + 1, n_cols, 768)
  Row 0:   column-header embeddings (from the header row tokens)
  Rows 1+: per-data-row cell embeddings

Each cell embedding is the mean of its body-token hidden states.
Empty cells (no body tokens) fall back to the SEP separator hidden state.

Data rows beyond cfg.approach.max_rows receive zero embeddings.
"""

import logging

import numpy as np
import pandas as pd

from benchmark_src.approach_interfaces.cell_embedding_interface import CellEmbeddingInterface

logger = logging.getLogger(__name__)


class CellEmbeddingComponent(CellEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame = None, dataset_information: dict = None):
        self.approach_instance.load_model()
        logger.info("TUTA model loaded for cell embedding task.")

    def create_cell_embeddings_for_table(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Args:
            input_table: DataFrame to embed.

        Returns:
            np.ndarray of shape (n_data_rows + 1, n_cols, 768).
        """
        cell_embs = self.approach_instance.get_cell_embeddings(input_table)
        logger.info(f"TUTA cell embeddings: {cell_embs.shape}")
        return cell_embs
