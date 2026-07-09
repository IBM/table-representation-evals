"""
Standalone PyTorch implementation of the TABBIE TableEmbedder.

Mirrors the exact AllenNLP module naming so that the pretrained weights.th
(from the AllenNLP archive) can be loaded with load_state_dict without any
key remapping.

AllenNLP version target: 1.x (gamma/beta for LayerNorm, _feedfoward_ typo, etc.)

Reference: https://github.com/SFIG611/tabbie
Paper: https://arxiv.org/abs/2105.02584
"""

from __future__ import annotations

import json
import math
import os
import tarfile
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AllenNLP-compatible stubs (matching parameter names in weights.th)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """AllenNLP 1.x LayerNorm uses 'gamma'/'beta' parameter names."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dimension))
        self.beta = nn.Parameter(torch.zeros(dimension))
        self._dimension = dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + 1e-12) + self.beta


class FeedForward(nn.Module):
    """
    AllenNLP FeedForward with _linear_layers ModuleList and _dropout.
    Supports arbitrary depth; activations are applied between layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activations: List[str],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [input_dim] + hidden_dims
        self._linear_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden_dims))]
        )
        self._activations = activations  # list of strings, not modules
        self._dropout = nn.Dropout(dropout)

    @staticmethod
    def _apply_activation(x: torch.Tensor, name: str) -> torch.Tensor:
        if name == "relu":
            return F.relu(x)
        if name == "gelu":
            return F.gelu(x)
        if name == "tanh":
            return torch.tanh(x)
        if name in ("linear", "none", "identity"):
            return x
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, act in zip(self._linear_layers, self._activations):
            x = self._apply_activation(layer(x), act)
            x = self._dropout(x)
        return x

    def get_output_dim(self) -> int:
        return self._linear_layers[-1].out_features


class MultiHeadSelfAttention(nn.Module):
    """
    AllenNLP-style multi-head self-attention.

    Parameter names match AllenNLP 1.x:
      _combined_projection  — Linear(input_dim, 2*attention_dim + values_dim)
      _output_projection    — Linear(values_dim, output_dim)
      _attention_dropout    — Dropout
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int,
        values_dim: int,
        num_heads: int,
        output_projection_dim: Optional[int] = None,
        attention_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._input_dim = input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim
        self._output_dim = output_projection_dim or input_dim
        self._scale = (attention_dim // num_heads) ** 0.5

        self._combined_projection = nn.Linear(
            input_dim, 2 * attention_dim + values_dim
        )
        self._output_projection = nn.Linear(values_dim, self._output_dim)
        self._attention_dropout = nn.Dropout(attention_dropout_prob)

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: (batch, seq_len, input_dim)
            mask:   (batch, seq_len) — 1 = valid, 0 = padding

        Returns:
            (batch, seq_len, output_dim)
        """
        bs, seq_len, _ = inputs.shape
        h = self._num_heads

        combined = self._combined_projection(inputs)
        queries, keys, values = combined.split(
            [self._attention_dim, self._attention_dim, self._values_dim], dim=-1
        )

        # Reshape to (batch*heads, seq_len, dim_per_head)
        dh = self._attention_dim // h
        vh = self._values_dim // h

        queries = queries.view(bs, seq_len, h, dh).transpose(1, 2)  # (bs,h,sl,dh)
        keys    = keys.view(bs, seq_len, h, dh).transpose(1, 2)
        values  = values.view(bs, seq_len, h, vh).transpose(1, 2)   # (bs,h,sl,vh)

        attn = torch.matmul(queries, keys.transpose(-2, -1)) / self._scale  # (bs,h,sl,sl)

        # Mask: shape (batch, seq_len) → (batch, 1, 1, seq_len)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2).float()
            attn = attn + (1.0 - mask_expanded) * -1e9

        attn = F.softmax(attn, dim=-1)
        attn = self._attention_dropout(attn)

        context = torch.matmul(attn, values)            # (bs,h,sl,vh)
        context = context.transpose(1, 2).contiguous()  # (bs,sl,h,vh)
        context = context.view(bs, seq_len, self._values_dim)

        return self._output_projection(context)


class StackedSelfAttentionEncoder(nn.Module):
    """
    AllenNLP-style stacked self-attention encoder.

    Parameter names match AllenNLP 1.x exactly (including the 'feedfoward' typo):
      _attention_layers
      _feedfoward_layers            ← typo intentional, matches original
      _layer_norm_layers
      _feed_forward_layer_norm_layers
      dropout
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        projection_dim: int,
        feedforward_hidden_dim: int,
        num_layers: int,
        num_attention_heads: int,
        use_positional_encoding: bool = False,
        dropout_prob: float = 0.1,
        residual_dropout_prob: float = 0.2,
        attention_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self._use_positional_encoding = use_positional_encoding
        self._input_dim = input_dim
        self._output_dim = hidden_dim

        attention_dim = projection_dim * num_attention_heads
        values_dim = projection_dim * num_attention_heads

        self._attention_layers: nn.ModuleList = nn.ModuleList()
        self._feedfoward_layers: nn.ModuleList = nn.ModuleList()   # note: typo matches src
        self._layer_norm_layers: nn.ModuleList = nn.ModuleList()
        self._feed_forward_layer_norm_layers: nn.ModuleList = nn.ModuleList()

        for _ in range(num_layers):
            self._attention_layers.append(
                MultiHeadSelfAttention(
                    input_dim=hidden_dim,
                    attention_dim=attention_dim,
                    values_dim=values_dim,
                    num_heads=num_attention_heads,
                    output_projection_dim=hidden_dim,
                    attention_dropout_prob=attention_dropout_prob,
                )
            )
            self._feedfoward_layers.append(
                FeedForward(
                    input_dim=hidden_dim,
                    hidden_dims=[feedforward_hidden_dim, hidden_dim],
                    activations=["relu", "linear"],
                    dropout=dropout_prob,
                )
            )
            self._layer_norm_layers.append(LayerNorm(hidden_dim))
            self._feed_forward_layer_norm_layers.append(LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(residual_dropout_prob)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, seq_len, hidden_dim)
            mask:   (batch, seq_len)

        Returns:
            (batch, seq_len, hidden_dim)
        """
        output = inputs
        for attn, ff, ln_attn, ln_ff in zip(
            self._attention_layers,
            self._feedfoward_layers,
            self._layer_norm_layers,
            self._feed_forward_layer_norm_layers,
        ):
            residual = output
            attended = attn(output, mask)
            output = ln_attn(residual + self.dropout(attended))

            residual = output
            fwd = ff(output)
            output = ln_ff(residual + self.dropout(fwd))

        return output


# ---------------------------------------------------------------------------
# Utility helpers (mirrors embedder_util.TableUtil)
# ---------------------------------------------------------------------------

def _add_cls_tokens(
    bert_data: torch.Tensor,
    cls_col: torch.Tensor,
    cls_row: torch.Tensor,
    bs: int,
    n_rows_cls: int,
    n_cols_cls: int,
) -> torch.Tensor:
    """
    Prepend CLS row (dim=0) and CLS column (dim=2).

    bert_data before: (bs, n_rows+1, n_cols, 768)
      (where row 0 = header row)
    Returns: (bs, n_rows+2, n_cols+1, 768)
      row 0 = table-level CLS row, row 1 = header, rows 2..n+1 = data
      col 0 = per-row CLS token, cols 1..m = data columns
    """
    # Expand cls_col: (1, 768) → (bs, 1, n_cols_cls-1, 768)
    cls_col_exp = cls_col.view(1, 1, 1, 768).expand(bs, 1, n_cols_cls - 1, 768)
    # Stack header as second row of the data block (bert_data already has header at row 0)
    bert_data_with_header = torch.cat(
        [cls_col_exp, bert_data.reshape(bs, 1 + (n_rows_cls - 1), n_cols_cls - 1, 768)],
        dim=1,
    )  # (bs, n_rows_cls+1, n_cols_cls-1, 768)

    # Expand cls_row: (1, 768) → (bs, n_rows_cls+1, 1, 768)
    cls_row_exp = cls_row.view(1, 1, 1, 768).expand(bs, n_rows_cls + 1, 1, 768)
    bert_data_out = torch.cat([cls_row_exp, bert_data_with_header], dim=2)
    # (bs, n_rows_cls+1, n_cols_cls, 768)
    return bert_data_out


def _add_cls_mask(
    table_mask: torch.Tensor,
    bs: int,
    max_rows_cls: int,
    max_cols_cls: int,
    device: torch.device,
    nrows: List[int],
    ncols: List[int],
) -> torch.Tensor:
    """Add mask columns/rows for CLS tokens."""
    cls_mask_col = torch.ones([bs, 1, max_cols_cls - 1], device=device)
    cls_mask_row = torch.ones([bs, max_rows_cls + 1, 1], device=device)
    for k in range(bs):
        cls_mask_col[k, 0, ncols[k]:] = 0
        cls_mask_row[k, (nrows[k] + 2):, 0] = 0
    table_mask = torch.cat([cls_mask_col, table_mask], dim=1)
    table_mask = torch.cat([cls_mask_row, table_mask], dim=2)
    return table_mask


def _get_table_mask(
    bs: int,
    n_rows: int,
    n_cols: int,
    nrows: List[int],
    ncols: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Binary mask (bs, n_rows+1, n_cols); 1=valid cell."""
    mask = torch.ones(bs, n_rows + 1, n_cols, device=device)
    for k in range(bs):
        mask[k, nrows[k] + 1:, :] = 0
        mask[k, :, ncols[k]:] = 0
    return mask


def _get_row_embs(
    bert_data: torch.Tensor,
    bs: int,
    max_rows: int,
    max_cols: int,
    table_mask: torch.Tensor,
    transformer: StackedSelfAttentionEncoder,
) -> torch.Tensor:
    """Row-wise self-attention: each row is a sequence of columns."""
    bert_data_mod = bert_data * table_mask.unsqueeze(-1).float()
    # (bs, max_rows+1, max_cols, 768) → (bs*(max_rows+1), max_cols, 768)
    table_mask_row = table_mask.reshape(bs * (max_rows + 1), max_cols)
    bert_data_mod = bert_data_mod.reshape(bs * (max_rows + 1), max_cols, 768)
    row_embs = transformer(bert_data_mod, table_mask_row)
    return row_embs.reshape(bs, max_rows + 1, max_cols, 768)


def _get_col_embs(
    bert_data: torch.Tensor,
    bs: int,
    max_rows: int,
    max_cols: int,
    table_mask: torch.Tensor,
    transformer: StackedSelfAttentionEncoder,
) -> torch.Tensor:
    """Column-wise self-attention: each column is a sequence of rows."""
    bert_data_mod = bert_data * table_mask.unsqueeze(-1).float()
    # permute to (bs, max_cols, max_rows+1, 768)
    bert_data_mod = bert_data_mod.permute(0, 2, 1, 3)
    table_mask_col = table_mask.permute(0, 2, 1)
    # (bs*max_cols, max_rows+1, 768)
    bert_data_mod = bert_data_mod.reshape(bs * max_cols, max_rows + 1, 768)
    table_mask_col = table_mask_col.reshape(bs * max_cols, max_rows + 1)
    col_embs = transformer(bert_data_mod, table_mask_col)
    col_embs = col_embs.reshape(bs, max_cols, max_rows + 1, 768)
    return col_embs.permute(0, 2, 1, 3)  # (bs, max_rows+1, max_cols, 768)


# ---------------------------------------------------------------------------
# Main model — mirrors TableEmbedder from pretrain.py
# ---------------------------------------------------------------------------

class TABBIEModel(nn.Module):
    """
    Pure-PyTorch TABBIE TableEmbedder for inference.

    Parameter names mirror the AllenNLP TableEmbedder so that weights.th
    from the pretrained archive can be loaded with load_state_dict().

    The bert_embedder is NOT part of weights.th (it is instantiated lazily
    in forward() in the original code) and is therefore held separately.
    """

    NUM_MAX_ROW_POS = 35
    NUM_MAX_COL_POS = 25
    NUM_LAYERS = 12  # number of alternating row/col passes

    # Max BERT sequences per forward pass.  With max_cell_len=16 and
    # bert-base-uncased, 256 sequences ≈ 150 MB of GPU activations — safe on
    # any modern GPU.
    BERT_BATCH_SIZE = 256

    def __init__(
        self,
        transformer_params: Dict,
        feedforward_params: Dict,
        row_pos_params: Dict,
        col_pos_params: Dict,
    ) -> None:
        """
        All *_params dicts are parsed from the archive's config.json.

        transformer_params keys (same for row and col):
            input_dim, hidden_dim, projection_dim, feedforward_hidden_dim,
            num_layers, num_attention_heads, dropout_prob,
            residual_dropout_prob, attention_dropout_prob
        """
        super().__init__()
        self.device_internal = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Position embeddings (AllenNLP Embedding → just nn.Parameter weight)
        n_row = row_pos_params.get("num_embeddings", self.NUM_MAX_ROW_POS)
        n_col = col_pos_params.get("num_embeddings", self.NUM_MAX_COL_POS)
        emb_dim = row_pos_params.get("embedding_dim", 768)
        self.row_pos_embedding = _AllenNLPEmbedding(n_row, emb_dim)
        self.col_pos_embedding = _AllenNLPEmbedding(n_col, emb_dim)

        # FeedForward for the corruption detection head (not used for embeddings)
        ff_p = feedforward_params
        self.feedforward = FeedForward(
            input_dim=ff_p.get("input_dim", 1536),
            hidden_dims=ff_p.get("hidden_dims", [256, 2]),
            activations=ff_p.get("activations", ["relu", "linear"]),
            dropout=ff_p.get("dropout", 0.0),
        )

        # 12 row + 12 col transformers
        for i in range(1, self.NUM_LAYERS + 1):
            t = StackedSelfAttentionEncoder(**transformer_params)
            setattr(self, f"transformer_col{i}", t)
            t2 = StackedSelfAttentionEncoder(**transformer_params)
            setattr(self, f"transformer_row{i}", t2)

        # CLS tokens (loaded from .npy files, not in weights.th). cls_col/cls_row
        # are the raw numpy arrays; cls_col_t/cls_row_t are the same values cached
        # as tensors already on the target device, built once in load_tabbie_model
        # so get_tabemb doesn't redo the numpy->tensor conversion and host->device
        # transfer on every forward call.
        self.cls_col: Optional[np.ndarray] = None
        self.cls_row: Optional[np.ndarray] = None
        self.cls_col_t: Optional[torch.Tensor] = None
        self.cls_row_t: Optional[torch.Tensor] = None

        # BERT embedder (loaded separately, not in weights.th)
        self.bert_embedder = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_transformer(self, kind: str, idx: int) -> StackedSelfAttentionEncoder:
        return getattr(self, f"transformer_{kind}{idx}")

    def _embed_bert(
        self,
        headers: List[str],
        rows: List[List[str]],
        max_cell_len: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed table headers and cells with BERT (CLS token).

        Returns:
            bert_header: (1, n_cols, 768)
            bert_cell:   (1, n_rows, n_cols, 768)
        """
        from transformers import BertTokenizer, BertModel  # lazy import

        if self.bert_embedder is None:
            raise RuntimeError("Call set_bert_embedder() before embedding.")

        tokenizer, bert_model = self.bert_embedder
        device = self.device_internal

        def _encode(texts: List[str]) -> torch.Tensor:
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_cell_len,
            ).to(device)
            with torch.no_grad():
                out = bert_model(**enc)
            return out.last_hidden_state[:, 0, :]  # CLS tokens

        # Headers: (n_cols, 768)
        h_emb = _encode(headers)

        # Cells: (n_rows, n_cols, 768) — encode all cells at once
        n_cols = len(headers)
        n_rows = len(rows)
        flat_cells = [str(rows[r][c]) for r in range(n_rows) for c in range(n_cols)]
        if flat_cells:
            c_emb = _encode(flat_cells).view(n_rows, n_cols, 768)
        else:
            c_emb = torch.zeros(0, n_cols, 768, device=device)

        return h_emb.unsqueeze(0), c_emb.unsqueeze(0)  # add batch dim

    # ------------------------------------------------------------------
    # Core 2-D transformer forward
    # ------------------------------------------------------------------

    def get_tabemb(
        self,
        bert_header: torch.Tensor,
        bert_data: torch.Tensor,
        n_rows: int,
        n_cols: int,
        table_mask: torch.Tensor,
        nrows: List[int],
        ncols: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the alternating row/col transformer stack.

        Args:
            bert_header: (bs, n_cols, 768)
            bert_data:   (bs, n_rows, n_cols, 768)
            n_rows, n_cols: actual max dimensions in this batch
            table_mask:  (bs, n_rows+1, n_cols)
            nrows, ncols: per-sample row/col counts

        Returns:
            row_embs: (bs, n_rows+2, n_cols+1, 768)
            col_embs: same shape
        """
        bs = bert_header.shape[0]
        device = bert_header.device

        row_pos_ids = torch.arange(0, self.NUM_MAX_ROW_POS, device=device, dtype=torch.long)
        col_pos_ids = torch.arange(0, self.NUM_MAX_COL_POS, device=device, dtype=torch.long)

        n_rows_cls = n_rows + 1  # +1 for row-CLS token
        n_cols_cls = n_cols + 1  # +1 for col-CLS token

        cls_col = self.cls_col_t
        cls_row = self.cls_row_t

        row_pos_embs = self.row_pos_embedding(row_pos_ids[: n_rows_cls + 1])  # (n_rows+2, 768)
        col_pos_embs = self.col_pos_embedding(col_pos_ids[:n_cols_cls])       # (n_cols+1, 768)

        # Assemble full grid: (bs, n_rows+2, n_cols+1, 768)
        # First build (bs, n_rows+1, n_cols, 768) = [header_row; data_rows]
        bert_grid = torch.cat(
            [bert_header.unsqueeze(1), bert_data], dim=1
        )  # (bs, n_rows+1, n_cols, 768)

        # Add positional embeddings (before CLS prepend)
        # row pos: (n_rows+2, 768) → broadcast over (bs, n_rows+1, n_cols, 768)
        # The row positions go to rows 1..n+1 (row 0 will be CLS row, gets pos 0)
        # Following original code pattern
        bert_grid_cls = _add_cls_tokens(bert_grid, cls_col, cls_row, bs, n_rows_cls, n_cols_cls)
        # bert_grid_cls: (bs, n_rows+2, n_cols+1, 768)

        # Add positional embeddings
        # row_pos_embs[0..n_rows+1] → each row position
        row_pos = row_pos_embs.view(1, n_rows_cls + 1, 1, 768).expand(
            bs, n_rows_cls + 1, n_cols_cls, 768
        )
        col_pos = col_pos_embs.view(1, 1, n_cols_cls, 768).expand(
            bs, n_rows_cls + 1, n_cols_cls, 768
        )
        bert_grid_cls = bert_grid_cls + row_pos + col_pos

        table_mask_cls = _add_cls_mask(
            table_mask, bs, n_rows_cls, n_cols_cls, device, nrows, ncols
        )

        # Alternating 12-layer pass
        ave_embs = bert_grid_cls
        row_embs = col_embs = None
        for i in range(1, self.NUM_LAYERS + 1):
            t_row = self._get_transformer("row", i)
            t_col = self._get_transformer("col", i)
            if i == 1:
                row_embs = _get_row_embs(
                    bert_grid_cls, bs, n_rows_cls, n_cols_cls, table_mask_cls, t_row
                )
                col_embs = _get_col_embs(
                    bert_grid_cls, bs, n_rows_cls, n_cols_cls, table_mask_cls, t_col
                )
            else:
                row_embs = _get_row_embs(
                    ave_embs, bs, n_rows_cls, n_cols_cls, table_mask_cls, t_row
                )
                col_embs = _get_col_embs(
                    ave_embs, bs, n_rows_cls, n_cols_cls, table_mask_cls, t_col
                )
            ave_embs = (row_embs + col_embs) / 2.0

        return row_embs, col_embs

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        max_cell_len: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed a single table.

        Args:
            headers: list of column header strings, length n_cols
            rows:    list of rows, each a list of cell strings, length n_rows

        Returns:
            row_embs: np.ndarray (n_rows+2, n_cols+1, 768)
            col_embs: np.ndarray (n_rows+2, n_cols+1, 768)
              - row_embs[i, 0, :] = row-CLS for row i (table-CLS at i=0, header at i=1)
              - col_embs[0, j, :] = col-CLS for col j (table-CLS at j=0)
        """
        device = self.device_internal
        self.eval()

        n_cols = len(headers)
        n_rows = len(rows)

        bert_header, bert_cell = self._embed_bert(headers, rows, max_cell_len)
        bert_header = bert_header.to(device)
        bert_cell = bert_cell.to(device)

        table_mask = _get_table_mask(1, n_rows, n_cols, [n_rows], [n_cols], device)

        row_embs, col_embs = self.get_tabemb(
            bert_header, bert_cell, n_rows, n_cols, table_mask, [n_rows], [n_cols]
        )

        return row_embs[0].cpu().numpy(), col_embs[0].cpu().numpy()

    @torch.no_grad()
    def embed_table_batch(
        self,
        headers: List[str],
        windows: List[List[List[str]]],
        max_cell_len: int = 16,
    ) -> List[np.ndarray]:
        """
        Embed multiple row windows that share the same column slice in a single
        get_tabemb call.

        All windows must have the same columns (same ``headers``).  Windows may
        differ in row count — the last window of a table is typically shorter.
        Shorter windows are zero-padded to ``max(n_rows_i)`` internally; the
        table mask ensures padded rows are ignored by the transformer.

        Args:
            headers: column header strings, length n_cols (shared by all windows)
            windows: list of row windows; each window is a
                     ``List[List[str]]`` of shape ``(n_rows_i, n_cols)``
            max_cell_len: BERT max token length per cell (truncation)

        Returns:
            List of np.ndarray, one per window, each shape
            ``(n_rows_i + 2, n_cols + 1, 768)``.  Same layout as
            ``embed_table()``: row-CLS for data row r is at
            ``output[r + 2, 0, :]``.  Rows beyond ``n_rows_i`` contain
            zero-padding and should not be used.
        """
        device = self.device_internal
        self.eval()

        n_cols = len(headers)
        bs = len(windows)
        if bs == 0:
            return []

        tokenizer, bert_model = self.bert_embedder

        def _encode_chunked(texts: List[str]) -> torch.Tensor:
            """Encode texts through BERT in chunks of BERT_BATCH_SIZE."""
            if not texts:
                return torch.zeros(0, 768, device=device)
            parts: List[torch.Tensor] = []
            for i in range(0, len(texts), self.BERT_BATCH_SIZE):
                enc = tokenizer(
                    texts[i : i + self.BERT_BATCH_SIZE],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_cell_len,
                ).to(device)
                out = bert_model(**enc)
                parts.append(out.last_hidden_state[:, 0, :])
            return torch.cat(parts, dim=0)

        # --- headers: encode once, broadcast to all windows ---
        h_emb = _encode_chunked(headers)  # (n_cols, 768)
        bert_header = h_emb.unsqueeze(0).expand(bs, -1, -1).contiguous()  # (bs, n_cols, 768)

        # --- cells: encode all windows' cells in one chunked pass ---
        nrows = [len(w) for w in windows]
        max_rows = max(nrows)

        flat_cells = [
            str(cell)
            for rows in windows
            for row in rows
            for cell in row
        ]
        all_cell_embs = _encode_chunked(flat_cells)  # (Σ n_rows_i * n_cols, 768)

        # Assemble padded bert_data: (bs, max_rows, n_cols, 768)
        bert_data = torch.zeros(bs, max_rows, n_cols, 768, device=device)
        offset = 0
        for i, rows in enumerate(windows):
            nr = len(rows)
            n_cells = nr * n_cols
            if n_cells > 0:
                bert_data[i, :nr] = all_cell_embs[offset : offset + n_cells].view(
                    nr, n_cols, 768
                )
            offset += n_cells

        table_mask = _get_table_mask(
            bs, max_rows, n_cols, nrows, [n_cols] * bs, device
        )

        row_embs_batch, _ = self.get_tabemb(
            bert_header, bert_data, max_rows, n_cols, table_mask,
            nrows, [n_cols] * bs,
        )
        # row_embs_batch: (bs, max_rows + 2, n_cols + 1, 768)

        return [row_embs_batch[i].cpu().numpy() for i in range(bs)]

    def set_bert_embedder(self, tokenizer, model) -> None:
        self.bert_embedder = (tokenizer, model)


class _AllenNLPEmbedding(nn.Module):
    """
    Mirrors AllenNLP's Embedding module.
    State-dict key: '<name>.weight' — same as nn.Embedding.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.weight)


# ---------------------------------------------------------------------------
# Archive loading
# ---------------------------------------------------------------------------

def _parse_transformer_params(cfg: dict) -> dict:
    """
    Extract StackedSelfAttentionEncoder kwargs from an AllenNLP config node.
    Falls back to TABBIE paper defaults if keys are absent.
    """
    defaults = dict(
        input_dim=768,
        hidden_dim=768,
        projection_dim=64,
        feedforward_hidden_dim=2048,
        num_layers=1,
        num_attention_heads=8,
        use_positional_encoding=False,
        dropout_prob=0.1,
        residual_dropout_prob=0.2,
        attention_dropout_prob=0.1,
    )
    for k in defaults:
        if k in cfg:
            defaults[k] = cfg[k]
    return defaults


def _parse_feedforward_params(cfg: dict) -> dict:
    dropout = cfg.get("dropout", 0.0)
    # AllenNLP FeedForward supports per-layer dropout as a list; our implementation
    # uses a single Dropout layer, so take the first element if a list is provided.
    if isinstance(dropout, list):
        dropout = dropout[0] if dropout else 0.0
    return dict(
        input_dim=cfg.get("input_dim", 1536),
        hidden_dims=cfg.get("hidden_dims", [256, 2]),
        activations=cfg.get("activations", ["relu", "linear"]),
        dropout=dropout,
    )


def load_tabbie_model(
    archive_path: str,
    cls_col_path: str,
    cls_row_path: str,
    bert_model_name: str = "bert-base-uncased",
    device: Optional[str] = None,
) -> TABBIEModel:
    """
    Load the TABBIE model from an AllenNLP .tar.gz archive.

    Steps:
      1. Extract config.json and weights.th from the archive.
      2. Parse hyperparams from config.json.
      3. Build TABBIEModel with matching parameter names.
      4. Load weights.th via load_state_dict(strict=False).
      5. Load CLS numpy files.
      6. Load HuggingFace BERT for cell embeddings.

    Args:
        archive_path:   Path to mix.tar.gz or freq.tar.gz
        cls_col_path:   Path to clscol.npy
        cls_row_path:   Path to clsrow.npy
        bert_model_name: HuggingFace BERT model name
        device:         'cpu', 'cuda', or None (auto)
    """
    from transformers import BertTokenizer, BertModel

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    # --- 1. Extract archive ---
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive_path, "r:*") as tar:  # r:* auto-detects gz/plain
            tar.extractall(tmp)

        cfg_path = os.path.join(tmp, "config.json")
        # mix variant uses best.th; freq/other variants use weights.th
        wts_path = os.path.join(tmp, "weights.th")
        if not os.path.exists(wts_path):
            wts_path = os.path.join(tmp, "best.th")

        with open(cfg_path) as f:
            cfg = json.load(f)

        weights = torch.load(wts_path, map_location=device_obj, weights_only=False)

    # --- 2. Parse config ---
    model_cfg = cfg.get("model", cfg)  # AllenNLP wraps under "model"
    t_cfg = model_cfg.get("transformer_row1", {})
    ff_cfg = model_cfg.get("feedforward", {})
    rp_cfg = model_cfg.get("row_pos_embedding", {})
    cp_cfg = model_cfg.get("col_pos_embedding", {})

    transformer_params = _parse_transformer_params(t_cfg)
    feedforward_params = _parse_feedforward_params(ff_cfg)

    # Override projection_dim from actual checkpoint weights.
    # AllenNLP config stores projection_dim as the *total* attention dim (shared across all
    # heads), but StackedSelfAttentionEncoder here treats it as per-head.  Deriving from the
    # weight shape is unambiguous: _combined_projection is (2*attn_dim + values_dim, input_dim)
    # where attn_dim == values_dim, so attn_dim = shape[0] // 3.
    _probe_key = "transformer_col1._attention_layers.0._combined_projection.weight"
    if _probe_key in weights:
        _total_proj = weights[_probe_key].shape[0]  # = 3 * attention_dim
        _attention_dim = _total_proj // 3
        _num_heads = transformer_params["num_attention_heads"]
        transformer_params["projection_dim"] = _attention_dim // _num_heads

    # --- 3. Build model ---
    model = TABBIEModel(
        transformer_params=transformer_params,
        feedforward_params=feedforward_params,
        row_pos_params=rp_cfg,
        col_pos_params=cp_cfg,
    )
    model.device_internal = device_obj

    # --- 4. Load weights ---
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        _important = [k for k in missing if "pos_embedding" in k or "_attention" in k]
        if _important:
            import logging
            logging.getLogger(__name__).warning(
                "TABBIE: %d missing keys in state_dict (first 10): %s",
                len(missing), missing[:10],
            )
    if unexpected:
        import logging
        logging.getLogger(__name__).debug(
            "TABBIE: %d unexpected keys (first 5): %s", len(unexpected), unexpected[:5]
        )

    # --- 5. CLS tokens ---
    model.cls_col = np.load(cls_col_path).astype(np.float32)
    model.cls_row = np.load(cls_row_path).astype(np.float32)

    # --- 6. BERT embedder ---
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert = BertModel.from_pretrained(bert_model_name)
    bert.eval()
    bert.to(device_obj)
    for p in bert.parameters():
        p.requires_grad_(False)
    model.set_bert_embedder(tokenizer, bert)

    model.to(device_obj)
    model.eval()

    model.cls_col_t = torch.from_numpy(model.cls_col).to(device_obj)
    model.cls_row_t = torch.from_numpy(model.cls_row).to(device_obj)

    return model
