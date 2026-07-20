"""
Column embedding component for TUTA.

Column embeddings are derived by mean-pooling the cell embeddings that
belong to each column, including the header cell — so the column name
contributes to its representation, matching the convention other approaches
in this benchmark use (e.g. sentence_transformer's column embedding
explicitly prepends the column name to its serialized text). This mirrors
row_embedding_component.py's strategy but pools across rows (axis=0)
instead of across columns (axis=1).

This is not a native TUTA output — TUTA's own pretraining objectives and
heads only produce per-cell and per-table representations (see approach.py's
module docstring) — it's a pooling strategy adopted for this benchmark's
column-level tasks.

Output shape: (n_cols, 768)
"""

import logging

import numpy as np
import pandas as pd

from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface

logger = logging.getLogger(__name__)


class ColumnEmbeddingComponent(ColumnEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame = None, dataset_information: dict = None):
        self.approach_instance.load_model()
        logger.info("TUTA model loaded for column embedding task.")

    def create_column_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Args:
            input_table: DataFrame to embed.

        Returns:
            Tuple of:
              np.ndarray of shape (n_cols, 768).
              list of column names, in the same order as the embedding rows.
        """
        row_limit = self.approach_instance.column_embedding_row_limit
        sample_table = input_table.drop_duplicates()
        if len(sample_table) > row_limit:
            logger.info(
                f"TUTA: table has {len(input_table)} rows ({len(sample_table)} distinct) - "
                f"sampling {row_limit} rows for column embedding computation."
            )
            sample_table = sample_table.head(row_limit)

        # cell_embs: (n_sample_rows + 1, n_cols, 768)
        # Row 0 = header; rows 1+ = data rows
        cell_embs = self.approach_instance.get_cell_embeddings(sample_table)

        # Mean across rows, including the header row, per column
        column_embs = cell_embs.mean(axis=0)  # (n_cols, 768)

        logger.info(f"TUTA column embeddings: {column_embs.shape}")
        return column_embs, list(input_table.columns)
