import logging

import numpy as np
import pandas as pd
import torch

from benchmark_src.approach_interfaces.cell_embedding_interface import CellEmbeddingInterface

logger = logging.getLogger(__name__)


class CellEmbeddingComponent(CellEmbeddingInterface):
    """
    Cell embedding component for TaBERT.

    For each row TaBERT is called with that row's cell values as sample values,
    yielding a ``column_encoding`` of shape ``(1, num_cols, hidden)``.  These
    per-row column encodings are stacked to produce the full cell embedding
    matrix.

    Header embeddings (row 0) are obtained by encoding the table with the first
    non-null value of each column as the sample value — consistent with how
    ``get_column_embeddings`` works in the main approach.

    Output shape: ``(num_rows + 1, num_cols, embedding_dim)``
      - Row 0: column/header-level embeddings
      - Rows 1+: per-row cell embeddings (row i+1 corresponds to data row i)
    """

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, dataset_information: dict = None):
        self.approach_instance.load_trained_model()
        logger.info("TaBERT model loaded for cell embedding task")

    def create_cell_embeddings_for_table(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Args:
            input_table: DataFrame to embed.

        Returns:
            np.ndarray of shape (num_rows + 1, num_cols, embedding_dim).
            Row 0 holds column-level embeddings; rows 1+ hold per-data-row
            cell embeddings.
        """
        approach = self.approach_instance
        approach.load_trained_model()
        
        df = approach.preprocessing(input_table)
        model = approach.model
        empty_context = model.tokenizer.tokenize("")

        with torch.no_grad():
            # Row 0: header/column-level embeddings
            header_table = approach._df_to_tabert_table(df, row_idx=None)
            _, col_enc, _ = model.encode(
                contexts=[empty_context],
                tables=[header_table],
            )
            # (1, num_cols, hidden) → (num_cols, hidden)
            header_embs = col_enc[0].cpu().numpy()

            # Rows 1+: per-data-row cell embeddings
            row_embs = []
            for i in range(len(df)):
                row_table = approach._df_to_tabert_table(df, row_idx=i)
                _, col_enc_row, _ = model.encode(
                    contexts=[empty_context],
                    tables=[row_table],
                )
                # (1, num_cols, hidden) → (num_cols, hidden)
                row_embs.append(col_enc_row[0].cpu().numpy())

        # Stack: (num_rows, num_cols, hidden)
        row_embs_arr = np.stack(row_embs, axis=0)

        # Prepend header row: (num_rows + 1, num_cols, hidden)
        cell_embeddings = np.concatenate(
            [header_embs[np.newaxis, :, :], row_embs_arr], axis=0
        )
        logger.debug(f"Generated cell embeddings with shape: {cell_embeddings.shape}")
        return cell_embeddings
