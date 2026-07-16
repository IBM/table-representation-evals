import pandas as pd
import numpy as np
import logging

from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface

logger = logging.getLogger(__name__)


class ColumnEmbeddingComponent(ColumnEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance
        super().__init__(approach_instance=approach_instance)

    def setup_model_for_task(self, input_table: pd.DataFrame = None, dataset_information: dict = None):
        logger.info("Setting up TARTE model for column embedding task")
        self.approach_instance.load_trained_model()

    def create_column_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Returns (column_embeddings, column_names) where:
          column_embeddings — np.ndarray of shape (n_cols, 768)
          column_names      — list of column name strings

        Each column's embedding is the mean of TARTE transformer outputs
        at that column's sequence position across up to 50 sampled rows.
        Columns absent from all sampled rows get a zero vector.
        """
        embeddings = self.approach_instance.get_column_embeddings(input_table)
        return embeddings, list(input_table.columns)
