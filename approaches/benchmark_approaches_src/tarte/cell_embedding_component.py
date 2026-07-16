import pandas as pd
import numpy as np
import logging

from benchmark_src.approach_interfaces.cell_embedding_interface import CellEmbeddingInterface

logger = logging.getLogger(__name__)


class CellEmbeddingComponent(CellEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance
        super().__init__(approach_instance=approach_instance)

    def setup_model_for_task(self, input_table: pd.DataFrame = None, dataset_information: dict = None):
        logger.info("Setting up TARTE model for cell embedding task")
        self.approach_instance.load_trained_model()

    def create_cell_embeddings_for_table(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Returns cell embeddings of shape (n_rows, n_cols, 768).

        Each cell embedding is the transformer hidden state at the position
        corresponding to that column in the row's TARTE sequence.
        Missing / NaN cells get a zero vector.
        """
        return self.approach_instance.get_cell_embeddings(input_table)
