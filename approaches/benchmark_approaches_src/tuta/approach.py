"""
TUTA Embedding Approach for Tabular Data.

Paper:  TUTA: Tree-based Transformers for Generally Structured Table
        Pre-training (KDD 2021)
Code:   https://github.com/microsoft/TUTA_table_understanding

Three tasks are supported:
  * Cell embeddings  — mean-pool body tokens per cell; shape (n_rows+1, n_cols, 768)
                       Row 0 = header embeddings; rows 1+ = data-row cell embeddings.
  * Row embeddings   — mean of the cell embeddings for each data row; (n_rows, 768).
  * Table embedding  — the [CLS] hidden state from the backbone; (768,).

Pretrained weights (Google Drive, manual download):
  TUTA:          https://drive.google.com/file/d/1pEdrCqHxNjGM4rjpvCxeAUchdJzCYr1g
  TUTA-explicit: https://drive.google.com/file/d/1FPwn2lQKEf-cGlgFHr4_IkDk_6WThifW
  TUTA-base:     https://drive.google.com/file/d/1j5qzw3c2UwbVO7TTHKRQmTvRki8vDO0l

Place the downloaded .bin file at the path specified by cfg.approach.checkpoint_path.
If no checkpoint is provided the backbone starts from random initialisation (weak but
functional — useful for sanity-checking the pipeline).
"""

import dataclasses
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
from benchmark_approaches_src.tuta.approach_utils import (
    extract_cell_embeddings,
    table_to_tuta_inputs,
)

# Add TUTA source to Python path so we can import its model classes.
# Two entries are needed:
#   tuta_src/       → enables "from tuta.model.backbones import …"
#   tuta_src/tuta/  → satisfies TUTA's own bare "import model.embeddings as emb"
#                     inside backbones.py (the repo uses non-relative intra-package
#                     imports that assume the tuta/ subdirectory is on sys.path)
TUTA_SRC_PATH = os.path.join(os.path.dirname(__file__), "tuta_src")
TUTA_MODEL_PATH = os.path.join(TUTA_SRC_PATH, "tuta")
for _p in (TUTA_SRC_PATH, TUTA_MODEL_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

HIDDEN_SIZE = 768


# ---------------------------------------------------------------------------
# TUTA model configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TUTAConfig:
    """Mirrors the argparse Namespace expected by TUTA model classes."""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_encoder_layers: int = 12
    vocab_size: int = 30522          # bert-base-uncased vocab size
    magnitude_size: int = 10
    precision_size: int = 10
    top_digit_size: int = 10
    low_digit_size: int = 10
    row_size: int = 256
    column_size: int = 256
    tree_depth: int = 4
    node_degree: dataclasses.field(default_factory=lambda: [32, 32, 64, 256]) = None
    num_format_feature: int = 11
    attention_distance: int = 8
    attention_step: int = 0
    layer_norm_eps: float = 1e-6
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    hidden_act: str = "gelu"
    attn_method: str = "add"
    max_cell_length: int = 64
    target: str = "tuta"

    def __post_init__(self):
        if self.node_degree is None:
            self.node_degree = [32, 32, 64, 256]


# ---------------------------------------------------------------------------
# Main approach class
# ---------------------------------------------------------------------------

class TUTAEmbedder(BaseTabularEmbeddingApproach):
    """
    TUTA-based cell, row, and table embedder.

    The TUTA backbone (BbForTuta) produces hidden states for every token
    in the serialized table.  We aggregate these into per-cell, per-row,
    and table-level vectors.

    Config parameters (tuta.yaml):
        checkpoint_path:  Path to a pretrained TUTA .bin state-dict file.
                          If null/missing, uses random initialisation.
        lm:               HuggingFace tokenizer name (default: bert-base-uncased).
        max_seq_len:      Max token sequence length (default: 512).
        max_cell_tokens:  Max body tokens per cell (default: 16).
        max_rows:         Max data rows included per forward pass (default: 50).
        target:           TUTA variant — "tuta" or "tuta_explicit" (default: tuta).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.checkpoint_path: Optional[str] = cfg.approach.get("checkpoint_path", None)
        self.lm_name: str = cfg.approach.get("lm", "bert-base-uncased")
        self.max_seq_len: int = int(cfg.approach.get("max_seq_len", 512))
        self.max_cell_tokens: int = int(cfg.approach.get("max_cell_tokens", 8))
        self.max_rows: int = int(cfg.approach.get("max_rows", 50))
        self.target: str = cfg.approach.get("target", "tuta")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer = None

        logger.info(f"TUTAEmbedder initialised (device={self.device})")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load tokenizer and TUTA backbone (idempotent)."""
        if self.model is not None:
            return

        # Tokenizer
        logger.info(f"Loading tokenizer: {self.lm_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_name)

        # TUTA backbone
        try:
            from tuta.model.backbones import BACKBONES
        except ImportError as exc:
            raise ImportError(
                "Could not import TUTA model. "
                "Run approaches/benchmark_approaches_src/tuta/setup.sh first."
            ) from exc

        config = TUTAConfig(target=self.target)
        backbone_cls = BACKBONES.get(self.target)
        if backbone_cls is None:
            raise ValueError(
                f"Unknown TUTA target '{self.target}'. "
                f"Valid options: {list(BACKBONES.keys())}"
            )

        model = backbone_cls(config)

        # Load pretrained weights when available
        if self.checkpoint_path:
            ckpt_path = Path(self.checkpoint_path)
            if not ckpt_path.exists():
                logger.warning(
                    f"Checkpoint not found at {ckpt_path}. "
                    "Falling back to random initialisation."
                )
            else:
                logger.info(f"Loading TUTA checkpoint from {ckpt_path}")
                state_dict = torch.load(str(ckpt_path), map_location="cpu")

                # Checkpoints may be saved as the full TUTA model
                # (keys: "backbone.embeddings.…") — strip the prefix.
                if any(k.startswith("backbone.") for k in state_dict):
                    state_dict = {
                        k[len("backbone."):]: v
                        for k, v in state_dict.items()
                        if k.startswith("backbone.")
                    }

                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.debug(f"Missing keys ({len(missing)}): {missing[:5]} ...")
                if unexpected:
                    logger.debug(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
                logger.info("TUTA weights loaded.")
        else:
            logger.warning(
                "No checkpoint_path specified — using random initialisation. "
                "Set cfg.approach.checkpoint_path to a pretrained .bin file."
            )

        self.model = model.to(self.device).eval()
        logger.info(f"TUTA backbone ready on {self.device}")

    # ------------------------------------------------------------------
    # Embedding helpers used by components
    # ------------------------------------------------------------------

    def _encode_chunk(self, table: pd.DataFrame, row_start: int, row_end: int) -> torch.Tensor:
        """
        Encode one chunk of data rows [row_start, row_end) together with the
        header row.  Returns cell embeddings of shape
        (n_chunk_data_rows + 1, n_cols, 768) where row 0 = header.
        """
        chunk = table.iloc[row_start:row_end]
        n_cols = len(table.columns)
        n_chunk_rows = len(chunk) + 1  # +1 for header

        inputs, cell_body_positions = table_to_tuta_inputs(
            chunk,
            self.tokenizer,
            max_seq_len=self.max_seq_len,
            max_cell_tokens=self.max_cell_tokens,
            max_rows=None,  # chunk is already sized; no additional cap
        )

        inputs_dev = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            encoded_states = self.model(
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
            )
        encoded_states = encoded_states.cpu()

        return extract_cell_embeddings(
            encoded_states,
            cell_body_positions,
            num_rows=n_chunk_rows,
            num_cols=n_cols,
            hidden_size=HIDDEN_SIZE,
        )  # [n_chunk_rows, n_cols, 768]

    def get_cell_embeddings(self, table: pd.DataFrame) -> np.ndarray:
        """
        Returns cell embeddings of shape (n_data_rows + 1, n_cols, 768).
        Row 0  = header (column name) embeddings.
        Rows 1+= per-data-row cell embeddings for ALL rows in the table,
                 processed in chunks of `max_rows` each.
        """
        self.load_model()
        n_cols = len(table.columns)
        n_data_rows = len(table)

        header_embs: Optional[torch.Tensor] = None
        data_embs_list: List[torch.Tensor] = []

        for chunk_start in range(0, max(n_data_rows, 1), self.max_rows):
            chunk_end = min(chunk_start + self.max_rows, n_data_rows)
            chunk_embs = self._encode_chunk(table, chunk_start, chunk_end)
            # chunk_embs: [n_chunk_rows, n_cols, 768]
            # row 0 = header; rows 1+ = data rows for this chunk

            if header_embs is None:
                header_embs = chunk_embs[0:1]  # [1, n_cols, 768]

            if chunk_end > chunk_start:
                data_embs_list.append(chunk_embs[1:])  # [chunk_size, n_cols, 768]

        if header_embs is None:
            # Empty table — return just a header row of zeros
            return np.zeros((1, n_cols, HIDDEN_SIZE), dtype=np.float32)

        parts = [header_embs] + data_embs_list
        cell_embs = torch.cat(parts, dim=0)  # [n_data_rows + 1, n_cols, 768]
        return cell_embs.numpy()

    def get_table_embedding(self, table: pd.DataFrame) -> np.ndarray:
        """
        Returns the [CLS] hidden state as a 768-dim table embedding.
        Only the first chunk (header + first max_rows data rows) is used;
        the CLS token summarises the table content visible in that chunk.
        """
        self.load_model()
        inputs, _ = table_to_tuta_inputs(
            table,
            self.tokenizer,
            max_seq_len=self.max_seq_len,
            max_cell_tokens=self.max_cell_tokens,
            max_rows=self.max_rows,
        )
        inputs_dev = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            encoded_states = self.model(
                inputs_dev["token_id"], inputs_dev["num_mag"],
                inputs_dev["num_pre"], inputs_dev["num_top"],
                inputs_dev["num_low"], inputs_dev["token_order"],
                inputs_dev["pos_row"], inputs_dev["pos_col"],
                inputs_dev["pos_top"], inputs_dev["pos_left"],
                inputs_dev["format_vec"], inputs_dev["indicator"],
            )
        # CLS is always at position 0
        return encoded_states[0, 0, :].cpu().numpy()
