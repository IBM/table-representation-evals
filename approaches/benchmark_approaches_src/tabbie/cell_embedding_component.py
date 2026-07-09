import logging

import numpy as np
import pandas as pd
from benchmark_src.approach_interfaces.cell_embedding_interface import CellEmbeddingInterface

logger = logging.getLogger(__name__)


class CellEmbeddingComponent(CellEmbeddingInterface):
    """
    Cell embedding component for TABBIE.

    Concatenates the row- and column-transformer outputs for each cell
    position, producing 1536-dim embeddings (768 from each transformer).

    Output shape: (num_rows + 1, num_cols, 1536)
      - Row 0:   column/header-level embeddings
      - Rows 1+: per-data-row cell embeddings (row i+1 = data row i)
    """

    def __init__(self, approach_instance) -> None:
        self.approach_instance = approach_instance

    def setup_model_for_task(self, dataset_information: dict = None):
        self.approach_instance.load_trained_model()
        logger.info("TABBIE model loaded for cell embedding task.")

    def create_cell_embeddings_for_table(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Args:
            input_table: DataFrame to embed.

        Returns:
            np.ndarray of shape (num_rows + 1, num_cols, 1536).
        """
        return self.approach_instance.get_cell_embeddings(input_table)
