import pandas as pd
import numpy as np
import logging

from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface

logger = logging.getLogger(__name__)


class TableEmbeddingComponent(TableEmbeddingInterface):
    """
    Table embedding component for TaBERT.

    Produces a single fixed-size vector for an entire table by mean-pooling
    the column-level encodings returned by TaBERT.
    """

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance
        super().__init__()

    def setup_model_for_task(self):
        self.approach_instance.load_trained_model()

    def create_table_embedding(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Args:
            input_table: DataFrame representing the table to embed.

        Returns:
            np.ndarray of shape (embedding_dim,), L2-normalised.
        """
        if len(input_table) == 0:
            raise ValueError("Input table is empty.")

        # Apply optional row limit
        if self.approach_instance.table_row_limit != -1:
            input_table = input_table.head(self.approach_instance.table_row_limit)

        table_embedding = self.approach_instance.get_table_embedding(input_table)

        # L2 normalise for cosine similarity
        norm = np.linalg.norm(table_embedding)
        if norm > 0:
            table_embedding = table_embedding / norm

        return table_embedding

    def create_query_embedding(self, query: str) -> np.ndarray:
        """
        Encode a natural-language query as a table embedding.

        The query is wrapped in a single-column, single-row DataFrame so it
        passes through the same TaBERT pipeline as any other table.

        Args:
            query: Natural-language query string.

        Returns:
            np.ndarray of shape (embedding_dim,), L2-normalised.
        """
        query_table = pd.DataFrame({"query": [query]})
        query_embedding = self.approach_instance.get_table_embedding(query_table)

        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        return query_embedding
