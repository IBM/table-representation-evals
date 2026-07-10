"""
Table embedding component for TUTA.

Uses the [CLS] hidden state from the TUTA backbone as the table-level
embedding.  The CLS token is trained via the TCR (Table Context Retrieval)
objective to attend over cell representations and encode global table context.

For query embeddings (NL2column / table retrieval tasks), the query string
is encoded as a single-cell table and the CLS hidden state is returned.

Output: 768-dim vector.
"""

import logging

import numpy as np
import pandas as pd
import torch

from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface
from benchmark_approaches_src.tuta.approach_utils import table_to_tuta_inputs

logger = logging.getLogger(__name__)


class TableEmbeddingComponent(TableEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self):
        self.approach_instance.load_model()
        logger.info("TUTA model loaded for table embedding task.")

    def create_table_embedding(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Returns the 768-dim [CLS] hidden state for the table.

        Args:
            input_table: DataFrame to embed.

        Returns:
            np.ndarray of shape (768,).
        """
        table_emb = self.approach_instance.get_table_embedding(input_table)
        logger.info(f"TUTA table embedding: {table_emb.shape}")
        return table_emb

    def create_query_embedding(self, query: str) -> np.ndarray:
        """
        Encode a natural-language query string as a 768-dim vector.

        The query is wrapped into a single-cell, single-column DataFrame so
        the same TUTA serialization pipeline is reused.  The [CLS] hidden
        state is returned.

        Args:
            query: Natural-language query string.

        Returns:
            np.ndarray of shape (768,).
        """
        ai = self.approach_instance
        ai.load_model()

        # Wrap query as a one-cell table
        query_table = pd.DataFrame({"query": [query]})

        inputs, _ = table_to_tuta_inputs(
            query_table,
            ai.tokenizer,
            max_seq_len=ai.max_seq_len,
            max_cell_tokens=ai.max_cell_tokens,
            max_rows=1,
        )
        inputs_dev = {k: v.to(ai.device) for k, v in inputs.items()}

        with torch.no_grad():
            encoded_states = ai.model(
                inputs_dev["token_id"],
                inputs_dev["num_mag"],
                inputs_dev["num_pre"],
                inputs_dev["num_top"],
                inputs_dev["num_low"],
                inputs_dev["token_order"],
                inputs_dev["pos_row"],
                inputs_dev["pos_col"],
                inputs_dev["pos_top"],
                inputs_dev["pos_left"],
                inputs_dev["format_vec"],
                inputs_dev["indicator"],
            )  # [1, seq_len, 768]

        query_emb = encoded_states[0, 0, :].cpu().numpy()
        return query_emb
