"""
Utilities for serializing a pandas DataFrame into TUTA model inputs.

TUTA sequence layout:
    [CLS] | [SEP] tok... | [SEP] tok... | ... | [PAD]...
     ind=-1   ind=1  ind=2   ind=3  ind=4         ind=0

Each cell occupies one SEP token (indicator = 2*cell_num - 1) followed by
zero or more body tokens (indicator = 2*cell_num).  The CLS token uses
indicator = -1 and padding uses indicator = 0.

We include the header row (column names) as row 0, followed by data rows.
Tree positional encoding is simplified for flat relational tables:
  pos_top[token] = [c % 32, TOTAL_NODE, TOTAL_NODE, TOTAL_NODE]  (column c)
  pos_left[token] = [TOTAL_NODE, TOTAL_NODE, TOTAL_NODE, TOTAL_NODE]  (no row headers)
"""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

# ---- Tree position constants ----
# node_degree = [32, 32, 64, 256]; total = 384
TOTAL_NODE = 384
TREE_DEPTH = 4
DEFAULT_TREE = [TOTAL_NODE] * TREE_DEPTH

# ---- Numeric encoding defaults (non-numeric cell) ----
NON_MAG = 11   # magnitude_size = 10 → sentinel
NON_PRE = 11   # precision_size = 10 → sentinel
NON_TOP = 10   # top_digit_size = 10 → sentinel (digits 0-9)
NON_LOW = 10   # low_digit_size = 10 → sentinel


def _encode_number(text: str) -> Tuple[int, int, int, int]:
    """
    Return (magnitude, precision, top_digit, low_digit) for a numeric string.
    Non-numeric cells return (NON_MAG, NON_PRE, NON_TOP, NON_LOW).

    Matches TUTA's own tokenize_digit definition (tuta/tokenizer.py upstream):
    magnitude/precision are the digit-counts of the integer/decimal parts of
    the literal text, and top/low digit are the first digit of the integer
    part and the last digit of the full number - computed directly from the
    text, never via float()/scientific-notation formatting, so a cell like
    "250000000" isn't corrupted by exponent digits bleeding into the count.
    """
    cleaned = re.sub(r"[,$%]", "", text).strip()
    if cleaned.startswith(("+", "-")):
        cleaned = cleaned[1:]

    parts = [p for p in cleaned.split(".") if p]
    if not (1 <= len(parts) <= 2) or not all(p.isdigit() for p in parts):
        return NON_MAG, NON_PRE, NON_TOP, NON_LOW

    magnitude = min(9, len(parts[0]))
    precision = min(9, len(parts[1])) if len(parts) == 2 else 0
    top_digit = int(parts[0][0])
    low_digit = int(parts[-1][-1])

    return magnitude, precision, top_digit, low_digit


def table_to_tuta_inputs(
    table: pd.DataFrame,
    tokenizer,
    max_seq_len: int = 512,
    max_cell_tokens: int = 8,
    max_rows: Optional[int] = None,
    row_offset: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[int, int], List[int]], int]:
    """
    Serialize a DataFrame into TUTA backbone input tensors.

    Row 0 of the output corresponds to the header (column names).
    Rows 1..N correspond to data rows 0..N-1 of `table`.

    A data row only counts as "included" if every one of its cells got at
    least a SEP token appended before the max_seq_len budget ran out - a row
    that started but didn't fully fit is excluded entirely (not left half
    populated), so callers can re-queue it into the next chunk instead of it
    silently ending up with a mix of real and zero cell embeddings.

    Args:
        table:           Input DataFrame (values already stringified).
        tokenizer:       HuggingFace tokenizer (bert-base-uncased).
        max_seq_len:     Token budget for the full sequence (padded to this).
        max_cell_tokens: Max body tokens per cell (SEP separator not counted).
        max_rows:        If set, cap data rows at this number.
        row_offset:      Added to the pos_row feature (not to cell_body_positions'
                         row_idx keys, which stay local to this call) for data
                         rows only, so a caller processing one logical table
                         across several chunked forward passes can make
                         pos_row reflect the row's position in the whole
                         table rather than always restarting at 1 per chunk.
                         Still clamped at TUTA's native row_size=256 vocabulary.

    Returns:
        inputs:               Dict of tensors, each shape [1, seq_len] (or
                              [1, seq_len, tree_depth] / [1, seq_len, 11]).
        cell_body_positions:  {(row_idx, col_idx): [token_positions]} where
                              row_idx=0 is the header.  Positions point into
                              the body tokens; falls back to the SEP position
                              if a cell has no body tokens (empty value).
        n_rows_included:      Number of data rows (of len(all_rows) - 1
                              requested) that fully fit in this sequence.
    """
    df = table.copy().astype(str).fillna("")
    if max_rows is not None:
        df = df.iloc[:max_rows]

    headers = list(table.columns)
    data_rows = df.values.tolist()
    all_rows: List[List[str]] = [headers] + data_rows  # row 0 = header

    # ---- Sequence buffers ----
    seq_token_id: List[int] = []
    seq_num_mag: List[int] = []
    seq_num_pre: List[int] = []
    seq_num_top: List[int] = []
    seq_num_low: List[int] = []
    seq_token_order: List[int] = []
    seq_pos_row: List[int] = []
    seq_pos_col: List[int] = []
    seq_pos_top: List[List[int]] = []
    seq_pos_left: List[List[int]] = []
    seq_format_vec: List[List[float]] = []
    seq_indicator: List[int] = []

    def _append_token(
        tid: int,
        mag: int,
        pre: int,
        top: int,
        low: int,
        order: int,
        row: int,
        col: int,
        pt: List[int],
        pl: List[int],
        ind: int,
    ) -> None:
        seq_token_id.append(tid)
        seq_num_mag.append(mag)
        seq_num_pre.append(pre)
        seq_num_top.append(top)
        seq_num_low.append(low)
        seq_token_order.append(order)
        seq_pos_row.append(min(row, 255))
        seq_pos_col.append(min(col, 255))
        seq_pos_top.append(pt)
        seq_pos_left.append(pl)
        seq_format_vec.append([0.0] * 11)
        seq_indicator.append(ind)

    # CLS token
    _append_token(
        tokenizer.cls_token_id,
        NON_MAG, NON_PRE, NON_TOP, NON_LOW,
        0, 0, 0, DEFAULT_TREE, DEFAULT_TREE, -1,
    )

    cell_body_positions: Dict[Tuple[int, int], List[int]] = {}
    cell_num = 1
    n_rows_included = 0  # count of fully-included data rows (r_idx >= 1)

    for r_idx, row_vals in enumerate(all_rows):
        row_start_token_count = len(seq_token_id)
        row_keys_added: List[Tuple[int, int]] = []
        row_complete = True

        for c_idx, cell_val in enumerate(row_vals):
            if len(seq_token_id) >= max_seq_len - 1:
                # No room for even the SEP; stop early
                row_complete = False
                break

            cell_text = str(cell_val).strip()
            pos_row_value = r_idx if r_idx == 0 else r_idx + row_offset

            # Body tokens (may be empty for blank cells)
            body_ids = tokenizer.encode(
                cell_text,
                add_special_tokens=False,
                max_length=max_cell_tokens,
                truncation=True,
            )

            # Tree position: column c maps to a depth-0 node (0..31)
            col_node = c_idx % 32
            pt = [col_node] + [TOTAL_NODE] * (TREE_DEPTH - 1)
            pl = DEFAULT_TREE

            mag, pre, top_d, low_d = _encode_number(cell_text)
            sep_pos = len(seq_token_id)

            # SEP separator for this cell (indicator = 2*cell_num - 1)
            _append_token(
                tokenizer.sep_token_id,
                NON_MAG, NON_PRE, NON_TOP, NON_LOW,
                0, pos_row_value, c_idx, pt, pl,
                cell_num * 2 - 1,
            )

            # Body tokens (indicator = 2*cell_num)
            body_positions: List[int] = []
            for t_idx, tid in enumerate(body_ids):
                if len(seq_token_id) >= max_seq_len:
                    break
                body_positions.append(len(seq_token_id))
                _append_token(
                    tid,
                    mag if t_idx == 0 else NON_MAG,
                    pre if t_idx == 0 else NON_PRE,
                    top_d if t_idx == 0 else NON_TOP,
                    low_d if t_idx == 0 else NON_LOW,
                    min(t_idx + 1, max_cell_tokens - 1), pos_row_value, c_idx, pt, pl,
                    cell_num * 2,
                )

            # Fall back to SEP position if cell was empty
            cell_body_positions[(r_idx, c_idx)] = body_positions or [sep_pos]
            row_keys_added.append((r_idx, c_idx))
            cell_num += 1

        if row_complete:
            if r_idx >= 1:
                n_rows_included += 1
            if len(seq_token_id) >= max_seq_len:
                break
        else:
            # This row didn't fully fit. Keep its partial tokens only if
            # nothing else has fit yet (header row, or the very first data
            # row of this call) so the caller still gets *something* rather
            # than an empty sequence - otherwise roll it back so it isn't
            # left half-populated, and let the caller re-queue it whole into
            # the next chunk.
            if r_idx >= 1 and n_rows_included > 0:
                del seq_token_id[row_start_token_count:]
                del seq_num_mag[row_start_token_count:]
                del seq_num_pre[row_start_token_count:]
                del seq_num_top[row_start_token_count:]
                del seq_num_low[row_start_token_count:]
                del seq_token_order[row_start_token_count:]
                del seq_pos_row[row_start_token_count:]
                del seq_pos_col[row_start_token_count:]
                del seq_pos_top[row_start_token_count:]
                del seq_pos_left[row_start_token_count:]
                del seq_format_vec[row_start_token_count:]
                del seq_indicator[row_start_token_count:]
                for key in row_keys_added:
                    del cell_body_positions[key]
            break

    # ---- Pad to max_seq_len ----
    pad_len = max_seq_len - len(seq_token_id)
    seq_token_id    += [tokenizer.pad_token_id] * pad_len
    seq_num_mag     += [NON_MAG] * pad_len
    seq_num_pre     += [NON_PRE] * pad_len
    seq_num_top     += [NON_TOP] * pad_len
    seq_num_low     += [NON_LOW] * pad_len
    seq_token_order += [0] * pad_len
    seq_pos_row     += [0] * pad_len
    seq_pos_col     += [0] * pad_len
    seq_pos_top     += [DEFAULT_TREE] * pad_len
    seq_pos_left    += [DEFAULT_TREE] * pad_len
    seq_format_vec  += [[0.0] * 11] * pad_len
    seq_indicator   += [0] * pad_len

    inputs = {
        "token_id":    torch.tensor([seq_token_id],    dtype=torch.long),
        "num_mag":     torch.tensor([seq_num_mag],     dtype=torch.long),
        "num_pre":     torch.tensor([seq_num_pre],     dtype=torch.long),
        "num_top":     torch.tensor([seq_num_top],     dtype=torch.long),
        "num_low":     torch.tensor([seq_num_low],     dtype=torch.long),
        "token_order": torch.tensor([seq_token_order], dtype=torch.long),
        "pos_row":     torch.tensor([seq_pos_row],     dtype=torch.long),
        "pos_col":     torch.tensor([seq_pos_col],     dtype=torch.long),
        # shape [1, seq_len, tree_depth]
        "pos_top":     torch.tensor([seq_pos_top],     dtype=torch.long),
        "pos_left":    torch.tensor([seq_pos_left],    dtype=torch.long),
        # shape [1, seq_len, 11]
        "format_vec":  torch.tensor([seq_format_vec],  dtype=torch.float),
        "indicator":   torch.tensor([seq_indicator],   dtype=torch.long),
    }

    return inputs, cell_body_positions, n_rows_included


def extract_cell_embeddings(
    encoded_states: torch.Tensor,
    cell_body_positions: Dict[Tuple[int, int], List[int]],
    num_rows: int,
    num_cols: int,
    hidden_size: int = 768,
) -> torch.Tensor:
    """
    Extract per-cell embeddings from the backbone encoded_states.

    For each (row, col) pair, mean-pools the body token hidden states.
    Cells that did not fit in the sequence receive zero embeddings.

    Args:
        encoded_states:      Shape [1, seq_len, hidden_size] (on CPU or GPU).
        cell_body_positions: Mapping from (r, c) → list of token indices.
        num_rows:            Total rows including the header row (row 0).
        num_cols:            Number of columns.
        hidden_size:         Hidden dimension size.

    Returns:
        Tensor of shape [num_rows, num_cols, hidden_size] on CPU.
    """
    states = encoded_states.squeeze(0)  # [seq_len, hidden_size]
    out = torch.zeros(num_rows, num_cols, hidden_size)

    for (r, c), positions in cell_body_positions.items():
        if r >= num_rows or c >= num_cols:
            continue
        if not positions:
            continue
        pos_t = torch.tensor(positions, dtype=torch.long)
        out[r, c] = states[pos_t].mean(dim=0).cpu()

    return out
