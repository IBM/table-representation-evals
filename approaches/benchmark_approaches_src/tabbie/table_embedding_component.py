import logging

import numpy as np
import pandas as pd
from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface

logger = logging.getLogger(__name__)


class TableEmbeddingComponent(TableEmbeddingInterface):
    """
    Table embedding component for TABBIE.

    Produces a single L2-normalised vector for an entire table by
    mean-pooling the row-CLS embeddings across all data rows.
    """

    def __init__(self, approach_instance) -> None:
        self.approach_instance = approach_instance
        super().__init__()

    def setup_model_for_task(self):
        self.approach_instance.load_trained_model()

    def create_table_embedding(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Args:
            input_table: DataFrame representing the table.

        Returns:
            np.ndarray of shape (768,), L2-normalised.
        """
        if len(input_table) == 0:
            raise ValueError("Input table is empty.")
        return self.approach_instance.get_table_embedding(input_table)

    def create_query_embedding(self, query: str) -> np.ndarray:
        """
        Encode a natural-language query as a table embedding.

        Wraps the query in a single-column single-row DataFrame so it
        passes through the same TABBIE pipeline as any other table.

        Returns:
            np.ndarray of shape (768,), L2-normalised.
        """
        query_table = pd.DataFrame({"query": [query]})
        return self.approach_instance.get_table_embedding(query_table)
