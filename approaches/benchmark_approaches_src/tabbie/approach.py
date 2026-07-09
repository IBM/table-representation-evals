"""
TABBIE Embedding Approach for Tabular Data.

Implements the TABBIE (Tabular Information Embedding) model for generating
row, column, cell, and table embeddings.

Architecture: alternating row/col BERT-based transformers (12 layers each).
Max per-forward-pass: 30 rows × 20 columns (predictor clips; model pos.
embeddings support up to 35 × 25).

Larger tables are handled by:
  - Row dim > MAX_ROWS: non-overlapping row windows, per-window CLS tokens
    concatenated.
  - Col dim > MAX_COLS: non-overlapping col windows, each column's embedding
    extracted from the window that contains it.
  - Both: double loop.

Reference: https://github.com/SFIG611/tabbie
Paper:     https://arxiv.org/abs/2105.02584
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach

logger = logging.getLogger(__name__)

# Strict per-forward-pass limits (same as original predictor).
# MAX_ROWS is 29 (not 30): the pretrained model's row_pos_embedding has 31 entries
# (indices 0–30). Each forward pass needs n_rows+2 position embeddings, so
# n_rows ≤ 29 keeps the max index at 30, within bounds.
MAX_ROWS: int = 29
MAX_COLS: int = 20

# Number of row windows batched into a single get_tabemb call.
# All windows in a batch share the same column slice and are padded to the
# same row count; a table mask prevents padded rows from contributing.
WINDOW_BATCH_SIZE: int = 8


class TABBIEEmbedder(BaseTabularEmbeddingApproach):
    """
    TABBIE embedding approach.

    Loads the pretrained TABBIE model (mix variant by default) from an
    AllenNLP archive (.tar.gz) without requiring AllenNLP to be installed.

    Config keys (approach yaml):
        archive_path   — path to mix.tar.gz relative to repo root
        cls_col_path   — path to clscol.npy
        cls_row_path   — path to clsrow.npy
        bert_model_name — HuggingFace BERT identifier (default: bert-base-uncased)
        table_row_limit — soft limit for row/col embeddings; -1 = no limit
                          (chunking handles large tables automatically)
        max_cell_len    — BERT max token length per cell (default: 16)
        embedding_dim   — 768 (row/col); cell dim is 1536
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        repo_root = Path(cfg.project_root)
        approach_cfg = cfg.approach

        self.archive_path = str(
            repo_root / approach_cfg.archive_path
        )
        self.cls_col_path = str(repo_root / approach_cfg.cls_col_path)
        self.cls_row_path = str(repo_root / approach_cfg.cls_row_path)
        self.bert_model_name = getattr(approach_cfg, "bert_model_name", "bert-base-uncased")
        self.table_row_limit = getattr(approach_cfg, "table_row_limit", -1)
        self.max_cell_len = getattr(approach_cfg, "max_cell_len", 16)

        # Add tabbie_src to sys.path so tabbie_model can be imported
        tabbie_src = str(Path(__file__).parent)
        if tabbie_src not in sys.path:
            sys.path.insert(0, tabbie_src)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_trained_model(self) -> None:
        """Load TABBIE model (idempotent)."""
        if self.model is not None:
            return

        from tabbie_model import load_tabbie_model  # noqa: E402

        logger.info("Loading TABBIE model from %s", self.archive_path)
        self.model = load_tabbie_model(
            archive_path=self.archive_path,
            cls_col_path=self.cls_col_path,
            cls_row_path=self.cls_row_path,
            bert_model_name=self.bert_model_name,
            device=str(self.device),
        )
        logger.info("TABBIE model loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stringify all cells; replace NaN sentinel strings with ''."""
        df = df.copy()
        for col in df.columns:
            df[col] = df[col].astype(str).replace(
                {"nan": "", "None": "", "NaN": "", "NaT": "", "<NA>": ""}
            )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _df_to_headers_rows(
        df: pd.DataFrame,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> Tuple[List[str], List[List[str]]]:
        """Slice df to a window and return headers + rows."""
        cols = list(df.columns[col_start:col_end])
        sub = df.iloc[row_start:row_end, col_start:col_end]
        rows = [
            [str(sub.iloc[r, c]) for c in range(sub.shape[1])]
            for r in range(sub.shape[0])
        ]
        return cols, rows

    def _run_model(
        self,
        df: pd.DataFrame,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run TABBIE on a (row_window, col_window) slice.

        Returns:
            row_embs: (n_rows+2, n_cols+1, 768)
            col_embs: same shape
        """
        headers, rows = self._df_to_headers_rows(df, row_start, row_end, col_start, col_end)
        return self.model.embed_table(headers, rows, max_cell_len=self.max_cell_len)

    # ------------------------------------------------------------------
    # Row embeddings
    # ------------------------------------------------------------------

    def get_row_embeddings(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Generate one embedding per data row using the row-CLS token.

        Row CLS = row_embs[i+2, 0, :] for data row i (skip table-CLS at
        row_embs[0] and header at row_embs[1]; col-CLS is at col dim 0).

        For tables with > MAX_ROWS rows: chunked into non-overlapping windows
        of MAX_ROWS; WINDOW_BATCH_SIZE windows are processed together in a
        single get_tabemb call (batched over the bs dimension).

        For tables with > MAX_COLS cols: non-overlapping column windows are
        processed separately and their row-CLS embeddings averaged.

        Returns:
            np.ndarray of shape (num_rows, 768)
        """
        self.load_trained_model()
        df = self.preprocessing(input_table)
        n_rows, n_cols = len(df), len(df.columns)
        n_row_windows = (n_rows + MAX_ROWS - 1) // MAX_ROWS

        # Accumulate row-CLS embeddings per row window across column windows.
        # row_embs_accum[i] has shape (window_rows_i, 768); starts as None.
        row_embs_accum: List[Optional[np.ndarray]] = [None] * n_row_windows
        n_col_windows = 0

        for c_start in range(0, n_cols, MAX_COLS):
            c_end = min(c_start + MAX_COLS, n_cols)
            headers = list(df.columns[c_start:c_end])

            # Build all row windows for this column slice
            windows: List[List[List[str]]] = []
            for r_start in range(0, n_rows, MAX_ROWS):
                r_end = min(r_start + MAX_ROWS, n_rows)
                sub = df.iloc[r_start:r_end, c_start:c_end]
                rows = [
                    [str(sub.iloc[r, c]) for c in range(sub.shape[1])]
                    for r in range(sub.shape[0])
                ]
                windows.append(rows)

            # Process WINDOW_BATCH_SIZE windows per get_tabemb call
            for b_start in range(0, len(windows), WINDOW_BATCH_SIZE):
                batch = windows[b_start : b_start + WINDOW_BATCH_SIZE]
                embs_list = self.model.embed_table_batch(
                    headers, batch, max_cell_len=self.max_cell_len
                )
                for local_idx, (rows, row_embs) in enumerate(zip(batch, embs_list)):
                    win_idx = b_start + local_idx
                    window_rows = len(rows)
                    # row_embs: (max_batch_rows + 2, n_cols + 1, 768)
                    # Row-CLS for data rows: [2 : 2 + window_rows, 0, :]
                    cls_embs = row_embs[2 : 2 + window_rows, 0, :]  # (window_rows, 768)
                    if row_embs_accum[win_idx] is None:
                        row_embs_accum[win_idx] = cls_embs.copy()
                    else:
                        row_embs_accum[win_idx] += cls_embs

            n_col_windows += 1

        embeddings = np.concatenate(
            [acc / n_col_windows for acc in row_embs_accum], axis=0
        )
        logger.info("TABBIE row embeddings shape: %s", embeddings.shape)
        return embeddings

    # ------------------------------------------------------------------
    # Column embeddings
    # ------------------------------------------------------------------

    def get_column_embeddings(
        self, input_table: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate one embedding per column using the column-CLS token.

        Col CLS = col_embs[0, j+1, :] for column j in the window
        (row dim 0 = table-level CLS row; col dim 0 = per-row CLS col).

        For tables with > MAX_COLS cols: chunked into non-overlapping
        windows of MAX_COLS; each window contributes CLS tokens for its cols.

        For tables with > MAX_ROWS rows: averaged over row windows (each
        row window provides a col-CLS computed from a different row subset).

        Returns:
            (col_embeddings, column_names)
            col_embeddings: np.ndarray of shape (num_cols, 768)
        """
        self.load_trained_model()
        df = self.preprocessing(input_table)
        n_rows, n_cols = len(df), len(df.columns)
        column_names = list(df.columns)

        col_embeddings = np.zeros((n_cols, 768), dtype=np.float32)
        n_row_windows = 0

        for r_start in range(0, n_rows, MAX_ROWS):
            r_end = min(r_start + MAX_ROWS, n_rows)
            col_embs_row_window = np.zeros((n_cols, 768), dtype=np.float32)

            for c_start in range(0, n_cols, MAX_COLS):
                c_end = min(c_start + MAX_COLS, n_cols)
                _, col_embs = self._run_model(df, r_start, r_end, c_start, c_end)
                # col_embs shape: (n_rows_window+2, n_cols_window+1, 768)
                # Col-CLS at row 0, cols 1..n_cols_window
                window_n_cols = c_end - c_start
                col_embs_row_window[c_start:c_end] += col_embs[0, 1 : 1 + window_n_cols, :]

            col_embeddings += col_embs_row_window
            n_row_windows += 1

        col_embeddings /= n_row_windows
        logger.info("TABBIE column embeddings shape: %s", col_embeddings.shape)
        return col_embeddings, column_names

    # ------------------------------------------------------------------
    # Cell embeddings
    # ------------------------------------------------------------------

    def get_cell_embeddings(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Generate cell embeddings as the concatenation of row and col
        transformer outputs: dim = 1536.

        Output shape: (n_rows + 1, n_cols, 1536)
          - Row 0: header row embeddings
          - Rows 1..n: data row cell embeddings

        For large tables, processed in (MAX_ROWS, MAX_COLS) windows;
        each cell's embedding comes from the window that contains it.
        Windows do NOT overlap, so boundary cells only appear once.

        Returns:
            np.ndarray of shape (num_rows + 1, num_cols, 1536)
        """
        self.load_trained_model()
        df = self.preprocessing(input_table)
        n_rows, n_cols = len(df), len(df.columns)

        # (n_rows+1, n_cols, 1536): row 0 = header
        cell_embs_out = np.zeros((n_rows + 1, n_cols, 1536), dtype=np.float32)

        for r_start in range(0, n_rows, MAX_ROWS):
            r_end = min(r_start + MAX_ROWS, n_rows)
            window_n_rows = r_end - r_start

            for c_start in range(0, n_cols, MAX_COLS):
                c_end = min(c_start + MAX_COLS, n_cols)
                window_n_cols = c_end - c_start

                row_embs, col_embs = self._run_model(df, r_start, r_end, c_start, c_end)
                # row_embs / col_embs: (window_n_rows+2, window_n_cols+1, 768)
                # After [1:, 1:, :] → (window_n_rows+1, window_n_cols, 768)
                r_slice = row_embs[1:, 1:, :]  # header + data rows, data cols
                c_slice = col_embs[1:, 1:, :]

                combined = np.concatenate([r_slice, c_slice], axis=-1)
                # combined: (window_n_rows+1, window_n_cols, 1536)
                # row 0 = header, rows 1..window_n_rows = data rows r_start..r_end-1

                # Write header row (row 0 in output, col window)
                cell_embs_out[0, c_start:c_end, :] = combined[0, :window_n_cols, :]

                # Write data rows
                out_row_start = 1 + r_start
                out_row_end = 1 + r_end
                cell_embs_out[out_row_start:out_row_end, c_start:c_end, :] = (
                    combined[1 : 1 + window_n_rows, :window_n_cols, :]
                )

        logger.info("TABBIE cell embeddings shape: %s", cell_embs_out.shape)
        return cell_embs_out

    # ------------------------------------------------------------------
    # Table embedding
    # ------------------------------------------------------------------

    def get_table_embedding(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Single vector for the whole table: mean of row-CLS embeddings.

        Returns:
            np.ndarray of shape (768,)
        """
        row_embs = self.get_row_embeddings(input_table)
        table_emb = row_embs.mean(axis=0)
        norm = np.linalg.norm(table_emb)
        if norm > 0:
            table_emb = table_emb / norm
        logger.info("TABBIE table embedding shape: %s", table_emb.shape)
        return table_emb
