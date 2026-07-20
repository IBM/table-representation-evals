"""
TARTE Embedding Approach for Tabular Data.

Paper: Table Foundation Models: on knowledge pre-training for tabular learning
       (TMLR 2025, arXiv:2505.14415)
Repo:  https://github.com/soda-inria/tarte-ai

TARTE (Transformer Augmented Representation of Table Entries) pre-trains a
transformer on knowledge graph data (Wikidata / YAGO4.5), giving it semantic
grounding that purely synthetic-data or task-supervised models lack.

Encoding pipeline per row
--------------------------
1. FastText (300-dim, cc.en.300.bin) embeds every column name and every
   categorical cell value as a 300-dim vector.
2. Numerical / datetime columns: scalar_value × FastText(column_name) → 300-dim.
3. A learnable readout token is prepended to the per-row sequence of 300-dim
   cell/column-name vectors.
4. A 3-layer transformer (768-dim, 24 heads) runs cross-column self-attention
   within each row — rows are processed independently, not jointly.
5. The readout token's hidden state, averaged across (a subset of) transformer
   layers, is the row embedding. Shape: (768,).

Note on y
---------
y is stored by TARTE_TablePreprocessor and threaded into each output tuple at
position 4: (row_idx, x, edge_attr, mask, y_).  TARTE_TableEncoder.transform()
only reads positions 1-3 and ignores y entirely.  We always pass y=None so
that embeddings are identical regardless of label availability — consistent
with the dummy-label approach used by other ICL models for unsupervised tasks.

Supported embeddings
--------------------
Row embeddings    — 768-dim vectors, one per row.  Shape: (n_rows, 768).
Table embedding   — mean-pool of row embeddings.   Shape: (768,).
Cell embeddings   — per-cell transformer outputs.  Shape: (n_rows, n_cols, 768).
                    Missing cells get a zero vector.
Column embeddings — mean-pooled cell outputs over sample rows.
                    Shape: (n_cols, 768).

Node sequence layout per row (from _preprocess_data):
  pos 0              : readout token (not a data cell)
  pos 1..n_cat       : non-null categorical cells, in cat_col_names_ order
  pos n_cat+1..+n_num: non-null numerical cells,   in num_col_names_ order
  pos ..+n_dat       : non-null datetime cells,    in dat_col_names_ order
  rest               : padding (masked)

Predictive ML   — uses row embeddings as features with an external classifier
                  (same as HyTrel); no fine-tuning of TARTE is performed.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach

logger = logging.getLogger(__name__)


class TARTEEmbedder(BaseTabularEmbeddingApproach):
    """
    TARTE-based row and table embedder.

    Config parameters (tarte.yaml):
        device:       "cuda" | "cpu" | "auto"   (default: auto)
        layer_index:  "all" | int | list[int]   (default: "all")
                      Which transformer layers to average for the row
                      embedding.  "all" averages all 3 layers.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg

        import torch
        device_cfg = cfg.approach.get("device", "auto")
        if device_cfg == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_cfg

        layer_cfg = cfg.approach.get("layer_index", "all")
        # OmegaConf may return a ListConfig; convert to a plain Python list.
        if hasattr(layer_cfg, "__iter__") and not isinstance(layer_cfg, str):
            self.layer_index = list(layer_cfg)
        else:
            self.layer_index = layer_cfg

        self._encoder = None  # TARTE_TableEncoder — weights loaded once

        logger.info(
            f"TARTEEmbedder initialised "
            f"(device={self.device}, layer_index={self.layer_index})"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_trained_model(self):
        """
        Load TARTE pretrained weights (idempotent).

        On the first call this downloads from HuggingFace:
          - inria-soda/tarte  : tarte_pretrained_weights.pt + configs (~500 MB)
          - hi-paris/fastText : cc.en.300.bin (~4.2 GB, cached after first run)
        Subsequent calls return immediately (weights already in memory).
        """
        if self._encoder is not None:
            return

        try:
            from tarte_ai import TARTE_TableEncoder
        except ImportError as exc:
            raise ImportError(
                "tarte-ai not found. Run "
                "approaches/benchmark_approaches_src/tarte/setup.sh first."
            ) from exc

        logger.info(
            "Loading TARTE encoder "
            "(downloads weights from HuggingFace on first run)..."
        )
        self._encoder = TARTE_TableEncoder(
            device=self.device,
            layer_index=self.layer_index,
        )
        # fit() loads pretrain_model_dict_ + pretrained_model_configs_ from HF hub.
        # X is not used inside fit(); we pass [] as a harmless placeholder.
        self._encoder.fit([])

        logger.info("TARTE encoder ready.")

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def get_row_embeddings(
        self,
        input_table: pd.DataFrame,
        train_size: Optional[int] = None,
        train_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract 768-dim row embeddings for every row in input_table.

        A fresh TARTE_TablePreprocessor is created for each table because it
        must fit a PowerTransformer on that table's numerical columns and build
        a FastText-lookup dict for that table's column/value names.

        The encoder's internal SimpleImputer (used for any NaN in the 768-dim
        embedding vectors) is reset before each table so it re-fits on the
        current table's embedding statistics rather than a stale prior table.

        Returns:
            np.ndarray of shape (n_rows, 768).
        """
        self.load_trained_model()

        try:
            from tarte_ai import TARTE_TablePreprocessor
        except ImportError as exc:
            raise ImportError(
                "tarte-ai not found. Run "
                "approaches/benchmark_approaches_src/tarte/setup.sh first."
            ) from exc

        # Fresh preprocessor: fits PowerTransformer and FastText lookups for
        # this specific table's schema.
        preprocessor = TARTE_TablePreprocessor()
        data = preprocessor.fit_transform(input_table, y=None)

        # Process rows in batches to avoid OOM from the O(N²) attention matrix
        # when N (rows) × L (seq_len) is large (e.g. wikidata_books: 1203 × 282).
        TARTE_BATCH_SIZE = 128
        self._encoder.is_fitted_ = False
        all_embeddings = []
        for start in range(0, len(data), TARTE_BATCH_SIZE):
            batch = data[start : start + TARTE_BATCH_SIZE]
            batch_embs = self._encoder.transform(batch)   # (batch_size, 768)
            all_embeddings.append(batch_embs)
        embeddings = np.vstack(all_embeddings)

        logger.debug(f"TARTE row embeddings shape: {embeddings.shape}")
        return embeddings

    def get_table_embedding(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Returns a single (768,) table-level embedding as the mean of all
        row embeddings.
        """
        row_embs = self.get_row_embeddings(input_table)
        return row_embs.mean(axis=0)

    # ------------------------------------------------------------------
    # Full-sequence helpers (for cell / column embeddings)
    # ------------------------------------------------------------------

    def _build_tarte_model(self):
        """
        Instantiate and load the full TARTE model (all 3 transformer layers).
        Returns the model in eval mode on self.device.
        Must be called after load_trained_model().
        """
        import torch
        import torch.nn as nn
        from tarte_ai.tarte_model import TARTE_Pretrain_NN

        enc = self._encoder
        cfg = enc.pretrained_model_configs_

        model_config = {
            "dim_input": cfg["dim_input"],
            "num_heads": cfg["num_heads"],
            "num_layers_transformer": cfg["num_layers_transformer"],
            "dropout": cfg["dropout"],
            "dim_transformer": cfg["dim_transformer"],
            "dim_feedforward": cfg["dim_feedforward"],
            "dim_projection": cfg["dim_transformer"],  # unused; we skip the projection head
        }

        model = TARTE_Pretrain_NN(**model_config)
        model.load_state_dict(enc.pretrain_model_dict_, strict=False)
        # Replace layer_norm with a fresh one (same as _extract_per_layer does)
        model.layer_norm = nn.LayerNorm(model_config["dim_transformer"])
        model = model.to(self.device)
        model.eval()
        return model

    def _get_full_sequence(self, data, model) -> np.ndarray:
        """
        Run the TARTE transformer and return the full hidden-state sequence.

        Args:
            data: list of (r_idx, x, edge_attr, mask, y_) from TARTE_TablePreprocessor
            model: loaded TARTE model from _build_tarte_model()

        Returns:
            np.ndarray of shape (n_rows, seq_len, 768)
        """
        import torch

        x_ = torch.stack([x for (_, x, _, _, _) in data]).to(self.device)
        edge_attr_ = torch.stack([ea for (_, _, ea, _, _) in data]).to(self.device)
        mask_ = torch.stack([m for (_, _, _, m, _) in data]).to(self.device)

        with torch.no_grad():
            z = model.tarte_base(x_, edge_attr_, mask_)  # (n_rows, seq_len, 768)
            z = model.layer_norm(z)                       # (n_rows, seq_len, 768)

        return z.cpu().numpy()

    def _build_col_position_map(self, preprocessor, row: pd.Series) -> dict:
        """
        For one row of the original (pre-preprocessed) table, return a dict
        mapping column_name → position_in_transformer_sequence (1-based; pos 0
        is the readout token).

        Columns absent from the row (NaN) are excluded — they get no embedding.
        """
        cat_present = [
            c for c in preprocessor.cat_col_names_
            if c in row.index and not pd.isna(row[c])
        ]
        num_present = [
            c for c in preprocessor.num_col_names_
            if c in row.index and not pd.isna(row[c])
        ]
        dat_present = [
            c for c in preprocessor.dat_col_names_
            if c in row.index and not pd.isna(row[c])
        ]

        pos_map: dict = {}
        pos = 1
        for c in cat_present:
            pos_map[c] = pos
            pos += 1
        for c in num_present:
            pos_map[c] = pos
            pos += 1
        for c in dat_present:
            pos_map[c] = pos
            pos += 1
        return pos_map

    # ------------------------------------------------------------------
    # Cell and column embedding extraction
    # ------------------------------------------------------------------

    def get_cell_embeddings(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Extract per-cell transformer hidden states.

        Returns:
            np.ndarray of shape (n_rows + 1, n_cols, 768).
            Row 0: per-column header embeddings — mean of cell embeddings
                   across all data rows (TARTE has no separate header token).
            Rows 1..n_rows: transformer output at each column's sequence
                   position for that row.  Missing / NaN cells get a zero
                   vector.
        """
        self.load_trained_model()

        try:
            from tarte_ai import TARTE_TablePreprocessor
        except ImportError as exc:
            raise ImportError(
                "tarte-ai not found. Run "
                "approaches/benchmark_approaches_src/tarte/setup.sh first."
            ) from exc

        n_rows, n_cols = input_table.shape
        col_names = list(input_table.columns)
        DIM = 768

        preprocessor = TARTE_TablePreprocessor()
        data = preprocessor.fit_transform(input_table, y=None)

        model = self._build_tarte_model()
        seq = self._get_full_sequence(data, model)  # (n_rows, seq_len, 768)

        # result[0] = header row (accumulated then averaged), result[1:] = data rows
        result = np.zeros((n_rows + 1, n_cols, DIM), dtype=np.float32)
        col_counts = np.zeros(n_cols, dtype=np.int64)

        for row_i in range(n_rows):
            row = input_table.iloc[row_i]
            pos_map = self._build_col_position_map(preprocessor, row)
            for col_j, col_name in enumerate(col_names):
                if col_name in pos_map:
                    cell_emb = seq[row_i, pos_map[col_name]]
                    result[row_i + 1, col_j] = cell_emb
                    result[0, col_j] += cell_emb
                    col_counts[col_j] += 1

        # Normalise header row to per-column mean; columns absent from all rows stay zero
        valid = col_counts > 0
        result[0, valid] /= col_counts[valid, np.newaxis]

        logger.debug(f"TARTE cell embeddings shape: {result.shape}")
        return result

    def get_column_embeddings(
        self,
        input_table: pd.DataFrame,
        n_sample_rows: int = 50,
    ) -> np.ndarray:
        """
        Extract per-column embeddings by mean-pooling transformer outputs
        over a sample of rows.

        Each column's embedding is the average of the transformer hidden states
        at the positions where that column appears across the sampled rows.
        Columns absent from all sampled rows receive a zero vector.

        Args:
            input_table:   The full table.
            n_sample_rows: How many rows to sample (default 50).

        Returns:
            np.ndarray of shape (n_cols, 768).
        """
        self.load_trained_model()

        try:
            from tarte_ai import TARTE_TablePreprocessor
        except ImportError as exc:
            raise ImportError(
                "tarte-ai not found. Run "
                "approaches/benchmark_approaches_src/tarte/setup.sh first."
            ) from exc

        n_rows, n_cols = input_table.shape
        col_names = list(input_table.columns)
        DIM = 768

        # Sample rows without replacement (or use all if table is small)
        if n_rows > n_sample_rows:
            sample_idx = np.random.choice(n_rows, n_sample_rows, replace=False)
            sample_table = input_table.iloc[sample_idx].reset_index(drop=True)
        else:
            sample_table = input_table.reset_index(drop=True)
        n_sample = len(sample_table)

        preprocessor = TARTE_TablePreprocessor()
        data = preprocessor.fit_transform(sample_table, y=None)

        model = self._build_tarte_model()
        seq = self._get_full_sequence(data, model)  # (n_sample, seq_len, 768)

        col_sums = np.zeros((n_cols, DIM), dtype=np.float64)
        col_counts = np.zeros(n_cols, dtype=np.int64)

        for row_i in range(n_sample):
            row = sample_table.iloc[row_i]
            pos_map = self._build_col_position_map(preprocessor, row)
            for col_j, col_name in enumerate(col_names):
                if col_name in pos_map:
                    col_sums[col_j] += seq[row_i, pos_map[col_name]]
                    col_counts[col_j] += 1

        result = np.zeros((n_cols, DIM), dtype=np.float32)
        valid = col_counts > 0
        result[valid] = (col_sums[valid] / col_counts[valid, np.newaxis]).astype(np.float32)

        logger.debug(f"TARTE column embeddings shape: {result.shape}")
        return result
