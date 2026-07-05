#!/usr/bin/env python3
"""
TaBERT Embedding Approach for Tabular Data

This module implements the TaBERT (Pretraining for Joint Understanding of Textual and
Tabular Data) approach for generating row, column, and table embeddings from tabular data.

Based on: https://github.com/facebookresearch/TaBERT
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach

# Add TaBERT source to Python path
TABERT_SRC_PATH = os.path.join(os.path.dirname(__file__), 'tabert_src')
if TABERT_SRC_PATH not in sys.path:
    sys.path.insert(0, TABERT_SRC_PATH)

logger = logging.getLogger(__name__)

# Default embedding dimension (BERT-base hidden size)
DEFAULT_EMBEDDING_DIM = 768


class TaBertEmbedder(BaseTabularEmbeddingApproach):
    """
    TaBERT embedding approach for tabular data.

    Wraps the Facebook Research TaBERT model to produce column-level and
    derived row / table embeddings without requiring a natural-language query.

    The model encodes each row independently:
    * An empty context (blank query) is passed alongside the full schema
      (column names + types) with the per-row cell values used as sample values.
    * ``column_encoding`` (shape ``[1, num_cols, hidden]``) is returned by
      ``model.encode`` and directly yields column representations.
    * Row embeddings are the mean of column encodings for that row.
    * Table embeddings are the mean of all row embeddings.

    Attributes:
        cfg (DictConfig): Hydra configuration object.
        model: Loaded TaBERT model instance.
        device (torch.device): Torch device to run the model on.
        table_row_limit (int): Maximum rows to process; -1 means no limit.
        model_path (str|None): Filesystem path to a pre-trained checkpoint.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.table_row_limit = cfg.approach.table_row_limit
        self.model_path = getattr(cfg.approach, "model_path", None)
        logger.info(f"TaBertEmbedder initialized on device: {self.device}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_trained_model(self):
        """Load the TaBERT model (idempotent – called multiple times safely)."""
        if self.model is not None:
            return

        try:
            from table_bert import TableBertModel
        except ImportError as exc:
            raise ImportError(
                "Failed to import TaBERT. "
                "Please run the tabert setup.sh script first."
            ) from exc

        if self.model_path:
            resolved = Path(get_original_cwd()) / Path(self.model_path)
            if resolved.exists():
                logger.info(f"Loading TaBERT checkpoint from: {resolved}")
                self.model = TableBertModel.from_pretrained(str(resolved))
            else:
                logger.warning(
                    f"Checkpoint path not found ({resolved}). "
                    "Falling back to bert-base-uncased initialisation."
                )
                self.model = TableBertModel.from_pretrained("bert-base-uncased")
        else:
            logger.info("No checkpoint specified – using bert-base-uncased initialisation.")
            self.model = TableBertModel.from_pretrained("bert-base-uncased")

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"TaBERT model loaded on {self.device}")

    # ------------------------------------------------------------------
    # DataFrame → TaBERT Table helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_column_type(series: pd.Series) -> str:
        """Return a simple TaBERT column type string for a pandas Series."""
        if pd.api.types.is_numeric_dtype(series):
            return "real"
        return "text"

    def _df_to_tabert_table(self, df: pd.DataFrame, row_idx: Optional[int] = None):
        """
        Convert a pandas DataFrame (or a single row of it) into a TaBERT Table.

        When *row_idx* is given the sample_value of each column is taken from
        that row; otherwise the first non-null value of each column is used.

        Args:
            df: Input DataFrame (already preprocessed).
            row_idx: Index (0-based) of the row to use as sample values.

        Returns:
            table_bert.Table: Ready-to-tokenize TaBERT Table object.
        """
        from table_bert import Table, Column

        columns: List[Column] = []
        for col_name in df.columns:
            col_type = self._infer_column_type(df[col_name])
            if row_idx is not None:
                sample_val = str(df[col_name].iloc[row_idx])
            else:
                # Use the first non-empty value as the representative sample
                non_null = df[col_name].replace('', pd.NA).dropna()
                sample_val = str(non_null.iloc[0]) if len(non_null) > 0 else ''
            columns.append(Column(col_name, col_type, sample_value=sample_val))

        # data is not used by encode() (sample_value drives the representation)
        table = Table(id="table", header=columns, data=[]).tokenize(self.model.tokenizer)
        return table

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocessing(self, input_table: pd.DataFrame) -> pd.DataFrame:
        """Stringify and normalise NaN values."""
        df = input_table.copy()
        for col in df.columns:
            df[col] = df[col].astype(str).replace(
                ['nan', 'None', 'NaN', 'NaT', '<NA>'], ''
            )
        return df

    # ------------------------------------------------------------------
    # Row embeddings
    # ------------------------------------------------------------------

    def get_row_embeddings(
        self, input_table: pd.DataFrame, train_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate one embedding per row of *input_table*.

        Each row is encoded independently by constructing a TaBERT Table where
        the sample values are drawn from that row.  The row embedding is the
        mean of the per-column encodings returned by ``model.encode``.

        Args:
            input_table: Input DataFrame.
            train_size: If provided, only embeddings for rows *after* index
                        ``train_size`` are returned (test-set only extraction).

        Returns:
            np.ndarray of shape ``(num_rows, embedding_dim)``, or
            ``(num_test_rows, embedding_dim)`` when *train_size* is set.
        """
        self.load_trained_model()
        df = self.preprocessing(input_table)

        row_embeddings = []
        empty_context = self.model.tokenizer.tokenize("")

        with torch.no_grad():
            for i in range(len(df)):
                table = self._df_to_tabert_table(df, row_idx=i)
                _, col_enc, _ = self.model.encode(
                    contexts=[empty_context],
                    tables=[table],
                )
                # col_enc: (1, num_cols, hidden) → mean over columns → (hidden,)
                row_emb = col_enc[0].mean(dim=0).cpu().numpy()
                row_embeddings.append(row_emb)

        embeddings = np.stack(row_embeddings, axis=0)

        if train_size is not None:
            embeddings = embeddings[train_size:]
            logger.info(f"Returning {len(embeddings)} test-row embeddings.")

        logger.info(f"Generated row embeddings with shape: {embeddings.shape}")
        return embeddings

    # ------------------------------------------------------------------
    # Column embeddings
    # ------------------------------------------------------------------

    def get_column_embeddings(self, input_table: pd.DataFrame):
        """
        Generate one embedding per column.

        The full table is encoded once (using the first non-null cell value per
        column as the sample value).  The ``column_encoding`` output of
        ``model.encode`` directly gives one vector per column.

        Args:
            input_table: Input DataFrame.

        Returns:
            Tuple ``(column_embeddings, column_names)`` where
            *column_embeddings* has shape ``(num_columns, embedding_dim)``.
        """
        self.load_trained_model()
        df = self.preprocessing(input_table)
        column_names = list(df.columns)
        empty_context = self.model.tokenizer.tokenize("")

        try:
            from table_bert.input_formatter import TableTooLongError
        except ImportError:
            TableTooLongError = ValueError

        try:
            with torch.no_grad():
                table = self._df_to_tabert_table(df)
                _, col_enc, _ = self.model.encode(
                    contexts=[empty_context],
                    tables=[table],
                )
            # col_enc: (1, num_cols, hidden) → (num_cols, hidden)
            column_embeddings = col_enc[0].cpu().numpy()
        except TableTooLongError:
            logger.warning(
                f"Table with {len(column_names)} columns is too long for TaBERT "
                f"(first column exceeds max token length). "
                f"Returning zero embeddings."
            )
            column_embeddings = np.zeros((len(column_names), DEFAULT_EMBEDDING_DIM), dtype=np.float32)

        logger.info(f"Generated column embeddings with shape: {column_embeddings.shape}")
        return column_embeddings, column_names

    # ------------------------------------------------------------------
    # Table embedding
    # ------------------------------------------------------------------

    def get_table_embedding(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Generate a single embedding for the entire table.

        The table embedding is the mean of all column embeddings, which
        captures the aggregate schema-level semantics of the table.

        Args:
            input_table: Input DataFrame.

        Returns:
            np.ndarray of shape ``(embedding_dim,)``.
        """
        column_embeddings, _ = self.get_column_embeddings(input_table)
        table_embedding = column_embeddings.mean(axis=0)
        logger.info(f"Generated table embedding with shape: {table_embedding.shape}")
        return table_embedding
