import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import Levenshtein
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from benchmark_src.dataset_creation.target.collect_all_target_datasets import get_target_dataset_by_name

logger = logging.getLogger(__name__)

MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 3
eps = 1e-8


def convert_array_to_markdown(table_array: List[List[Any]], max_rows: int = -1) -> str:
    """
    Converts a list of lists (where the first list is headers)
    into a Markdown table string.

    Args:
        table_array: A list of lists.
                     Example: [["col1", "col2"], ["data1", "data2"]]
        max_rows: The maximum number of data rows to include. If -1, there is no limit.
    """
    if not table_array or not table_array[0]:
        logger.error("Empty table array provided.")
        return ""

    headers = [str(h) for h in table_array[0]]
    lines: List[str] = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    data_rows = table_array[1:]

    if max_rows != -1 and len(data_rows) > max_rows:
        data_rows = data_rows[:max_rows]

    for row in data_rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")

    # Join all lines with a newline and add a final newline
    return "\n".join(lines) + "\n"

# -----------------------------
# Textual-change metric: normalized Levenshtein
# -----------------------------
def normalized_levenshtein(a: str, b: str) -> float:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return Levenshtein.distance(a, b) / max_len

def shuffle_rows(table: List[List[Optional[str]]], pos_strength: float
                ) -> Tuple[List[List[Optional[str]]], List[int], List[int]]:
    n_rows = len(table)
    # Only shuffle data rows (index 1 to N)
    data_row_indices = list(range(1, n_rows))
    k = int(round(pos_strength * len(data_row_indices)))

    fwd_map = list(range(n_rows))

    if n_rows <= 2 or pos_strength <= 0.0 or k <= 1:
        return [row[:] for row in table], fwd_map, []

    src_indices = random.sample(data_row_indices, k)

    dst_indices = src_indices[:]
    random.shuffle(dst_indices)

    new_T = [row[:] for row in table]
    for src, dst in zip(src_indices, dst_indices):
        new_T[dst] = table[src][:]
        fwd_map[src] = dst

    return new_T, fwd_map, src_indices


def shuffle_columns(table: List[List[Optional[str]]], pos_strength: float
                    ) -> Tuple[List[List[Optional[str]]], List[int], List[str]]:
    n_rows = len(table)
    n_cols = len(table[0]) if n_rows > 0 else 0
    k = max(1, int(round(pos_strength * n_cols)))

    fwd_map = list(range(n_cols))

    if n_cols == 0 or pos_strength <= 0.0 or k <= 1:
        return [row[:] for row in table], fwd_map, []

    col_indices = list(range(n_cols))

    src_indices = random.sample(col_indices, k)

    dst_indices = src_indices[:]
    random.shuffle(dst_indices)

    new_T = []
    for r in table:
        new_row = r[:]
        for src, dst in zip(src_indices, dst_indices):
            new_row[dst] = r[src]
        new_T.append(new_row)

    for src, dst in zip(src_indices, dst_indices):
        fwd_map[src] = dst

    chosen_col_names = [table[0][c] for c in src_indices]

    return new_T, fwd_map, chosen_col_names


def shuffle_within_columns(table: List[List[Optional[str]]], chosen_cols: List[int], neg_degree: float
                           ) -> Tuple[List[List[Optional[str]]], Dict[str, List[int]], List[str]]:
    n_rows = len(table)
    data_rows = list(range(1, n_rows))
    L = len(data_rows)
    S = int(round(neg_degree * L))

    new_T = [row[:] for row in table]
    col_forward_maps: Dict[str, List[int]] = {}
    chosen_col_names = []

    if n_rows <= 2 or not chosen_cols or neg_degree <= 0.0 or S <= 1:
        return new_T, {}, []

    for c in chosen_cols:
        col_name = table[0][c]
        chosen_col_names.append(col_name)

        fwd_mapping = list(range(n_rows))

        src_indices = random.sample(data_rows, S)

        dst_indices = src_indices[:]
        random.shuffle(dst_indices)

        for src, dst in zip(src_indices, dst_indices):
            new_T[dst][c] = table[src][c]
            fwd_mapping[src] = dst

        col_forward_maps[col_name] = fwd_mapping

    return new_T, col_forward_maps, chosen_col_names


def generate_positive(table: List[List[Optional[str]]], pos_type: str, pos_strength: float):
    assert 0.0 <= pos_strength <= 1.0
    assert pos_type in {"row_reorder", "col_reorder", "both"}
    T = [row[:] for row in table]

    # Minimal metadata container (only per-triplet details, no global config)
    pos_meta: Dict[str, Any] = {}

    if pos_type in ("row_reorder", "both"):
        new_T, fwd_map, affected_rows = shuffle_rows(T, pos_strength)
        T = new_T
        pos_meta["row_permutation_index"] = fwd_map
        pos_meta["affected_row_indices"] = affected_rows

    if pos_type in ("col_reorder", "both"):
        new_T, fwd_map, affected_cols = shuffle_columns(T, pos_strength)
        T = new_T
        pos_meta["col_permutation_index"] = fwd_map
        pos_meta["affected_col_names"] = affected_cols

    return T, pos_meta


def generate_negative(table: List[List[Optional[str]]], neg_columns_frac: float, neg_degree: float):
    assert 0.0 <= neg_columns_frac <= 1.0
    assert 0.0 <= neg_degree <= 1.0

    n_rows = len(table)
    n_cols = len(table[0]) if n_rows > 0 else 0
    T = [row[:] for row in table]

    # Only store per-triplet details here; global config lives in the root summary
    neg_meta: Dict[str, Any] = {}

    if n_cols == 0 or n_rows <= 2:
        return T, neg_meta

    num_cols_to_perm = int(round(neg_columns_frac * n_cols))
    if num_cols_to_perm <= 0:
        return T, neg_meta

    chosen_cols = random.sample(list(range(n_cols)), num_cols_to_perm)

    new_T, col_forward_map, affected_names = shuffle_within_columns(T, chosen_cols, neg_degree)
    T = new_T

    neg_meta["col_permutation_indexes"] = col_forward_map
    neg_meta["affected_col_names"] = affected_names

    return T, neg_meta

# -----------------------------
# Triplet generation
# -----------------------------
def generate_triplets_from_dataset(
    dataset: List[Dict[str, Any]],
    triplets_per_anchor: int,
    pos_params: Dict[str, Any],
    neg_params: Dict[str, Any],
    variation_name: str,
):
    triplets = []
    deltas_pos = []
    deltas_neg = []

    for anchor_rec in tqdm(dataset, desc=f"Generating triplets for variation '{variation_name}'"):
        anchor_table = anchor_rec["table"]
        serialized_anchor = convert_array_to_markdown(anchor_table)

        for _ in range(triplets_per_anchor):
            pos_table, pos_meta = generate_positive(anchor_table, pos_params["pos_type"], pos_params["pos_strength"])
            neg_table, neg_meta = generate_negative(anchor_table, neg_params["neg_columns_frac"], neg_params["neg_degree"])

            delta_pos = normalized_levenshtein(serialized_anchor, convert_array_to_markdown(pos_table))
            delta_neg = normalized_levenshtein(serialized_anchor, convert_array_to_markdown(neg_table))

            # When doing randomized perturbations, it's possible to swap identical rows/columns or end up with no effective change.
            # Filter out these cases to ensure a cleaner dataset and more meaningful metrics.
            if delta_neg <= eps or delta_pos <= eps:
                continue

            triplets.append({
                "database_id": anchor_rec["database_id"],
                "table_id": anchor_rec["table_id"],
                "anchor_table": anchor_table,
                "pos_table": pos_table,
                "neg_table": neg_table,
                "delta_pos": delta_pos,
                "delta_neg": delta_neg,
                "pos_meta": pos_meta,
                "neg_meta": neg_meta,
            })
            deltas_pos.append(delta_pos)
            deltas_neg.append(delta_neg)

    avg_delta_pos = sum(deltas_pos) / len(deltas_pos) if deltas_pos else 0.0
    avg_delta_neg = sum(deltas_neg) / len(deltas_neg) if deltas_neg else 0.0

    return triplets, avg_delta_pos, avg_delta_neg


def table_2d_to_df(table_2d):
    """
    table_2d: list[list] where table_2d[0] is the header row.
    Returns a pandas DataFrame.
    """
    if table_2d is None or len(table_2d) == 0:
        return pd.DataFrame()

    header = list(table_2d[0])
    rows = table_2d[1:]

    df = pd.DataFrame(rows, columns=header)

    df = df.dropna(how="all")

    return df

def summarize_triplets(
    triplets,
    out_path,
    variation_name: str,
    dataset_name: str,
    pos_params: Dict[str, Any],
    neg_params: Dict[str, Any],
    avg_delta_pos: float,
    avg_delta_neg: float,
):
    """
    Write a single JSON summary per variation.

    Global perturbation configs (pos_params / neg_params) live at the root,
    while per-triplet metadata only holds triplet-specific details.
    """
    triplet_summaries = []
    for idx, t in enumerate(triplets):
        pos_meta = t.get("pos_meta", {})
        neg_meta = t.get("neg_meta", {})

        triplet_summaries.append(
            {
                "triplet_id": idx,
                "database_id": t.get("database_id"),
                "table_id": t.get("table_id"),
                "delta_pos": t.get("delta_pos"),
                "delta_neg": t.get("delta_neg"),
                "positive": {
                    "row_reorder": {
                        "permutation_index": pos_meta.get("row_permutation_index", []),
                        "affected_indices": pos_meta.get("affected_row_indices", []),
                    },
                    "column_reorder": {
                        "permutation_index": pos_meta.get("col_permutation_index", []),
                        "affected_indices": pos_meta.get("affected_col_names", []),
                    },
                },
                "negative": {
                    "intra_column_shuffle": {
                        "permutation_indexes": neg_meta.get(
                            "col_permutation_indexes", {}
                        ),
                        "affected_indices": neg_meta.get("affected_col_names", []),
                    },
                },
            }
        )

    summary_root = {
        "variation_name": variation_name,
        "dataset_name": dataset_name,
        "pos_params": pos_params,
        "neg_params": neg_params,
        "avg_delta_pos": avg_delta_pos,
        "avg_delta_neg": avg_delta_neg,
        "num_triplets": len(triplet_summaries),
        "triplets": triplet_summaries,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_root, f, ensure_ascii=False, indent=2)


def load_table_shuffling_config() -> DictConfig:
    """
    Load the table_shuffling dataset config from YAML.
    """
    try:
        config_path = (
            Path(get_original_cwd())
            / "benchmark_src"
            / "config"
            / "dataset"
            / "table_shuffling.yaml"
        )
    except ValueError:
        # Fallback if not running under Hydra
        config_path = (
            Path.cwd()
            / "benchmark_src"
            / "config"
            / "dataset"
            / "table_shuffling.yaml"
        )

    if not config_path.exists():
        raise FileNotFoundError(f"table_shuffling config file not found at {config_path}")

    return OmegaConf.load(str(config_path))


def _get_int_or_unbound(dataset_cfg: DictConfig, key: str) -> int:
    """Get int from config; -1 or unset means unbounded (return -1 for caller to interpret)."""
    v = OmegaConf.select(dataset_cfg, key, default=-1)
    if v is None:
        return -1
    return int(v)


def _resolve_dataset_limits(dataset_cfg: DictConfig) -> Tuple[int, Optional[int], int, Optional[int], Optional[int]]:
    raw_min_rows = _get_int_or_unbound(dataset_cfg, "min_rows")
    raw_max_rows = _get_int_or_unbound(dataset_cfg, "max_rows")
    raw_min_cols = _get_int_or_unbound(dataset_cfg, "min_cols")
    raw_max_cols = _get_int_or_unbound(dataset_cfg, "max_cols")
    raw_max_tables = _get_int_or_unbound(dataset_cfg, "max_tables")

    min_rows = MIN_TABLE_ROWS if raw_min_rows == -1 else raw_min_rows
    max_rows = None if raw_max_rows == -1 else raw_max_rows
    min_cols = MIN_TABLE_COLS if raw_min_cols == -1 else raw_min_cols
    max_cols = None if raw_max_cols == -1 else raw_max_cols
    max_tables = None if raw_max_tables == -1 else raw_max_tables

    return min_rows, max_rows, min_cols, max_cols, max_tables


def _select_corpus_subset(
    corpus,
    dataset_cfg: DictConfig,
) -> List[Dict[str, Any]]:
    """
    Apply filtering based on the dataset block:
      - keep tables whose #rows and #cols fall into the configured ranges
      - randomly sample up to max_tables from the qualifying tables or all if max_tables is unset.
    """
    min_rows, max_rows, min_cols, max_cols, max_tables = _resolve_dataset_limits(dataset_cfg)

    selected_indices: List[int] = []

    for idx, rec in enumerate(corpus):
        table = rec.get("table")
        if not table or not table[0]:
            continue

        num_rows = len(table) - 1  # exclude header
        num_cols = len(table[0])

        if num_rows < min_rows or (max_rows is not None and num_rows > max_rows):
            continue
        if num_cols < min_cols or (max_cols is not None and num_cols > max_cols):
            continue

        selected_indices.append(idx)

    if not selected_indices:
        return corpus.to_list()

    if max_tables is not None and len(selected_indices) > max_tables:
        selected_indices = random.sample(selected_indices, max_tables)

    return corpus.select(selected_indices).to_list()


def run_variation(
    dataset_id: str,
    dataset_cfg: DictConfig,
    variation_id: str,
    variation_cfg: DictConfig,
    base_output_dir: Path,
    root_random_seed: int,
) -> None:
    """
    Run triplet generation for a single (dataset, variation) pair and write outputs to disk.

    - dataset_id: logical dataset key (e.g. "fetaqa"), used for get_target_dataset_by_name.
    - dataset_cfg: per-dataset config (currently only max_tables).
    - variation_id: global variation id (e.g. "v0"), defining filters and perturbation params.
    - variation_cfg: variation config from table_shuffling.yaml.
    """
    # Derive a deterministic seed per dataset×variation from the root seed and ids
    seed = int(root_random_seed) + abs(hash(f"{dataset_id}@@{variation_id}")) % (2**16)
    random.seed(seed)

    corpus = get_target_dataset_by_name(dataset_id).corpus

    # Global row/column filters live on the variation under "dataset".
    filters_cfg = variation_cfg["dataset"]

    # Combine global filters with per-dataset settings (e.g. max_tables).
    # dataset_cfg is expected to only add/override keys like max_tables.
    corpus_list = _select_corpus_subset(corpus, OmegaConf.merge(filters_cfg, dataset_cfg))

    pos_params = dict(variation_cfg.pos_params)
    neg_params = dict(variation_cfg.neg_params)

    triplets, avg_delta_pos, avg_delta_neg = generate_triplets_from_dataset(
        dataset=corpus_list,
        triplets_per_anchor=int(variation_cfg.triplets_per_anchor),
        pos_params=pos_params,
        neg_params=neg_params,
        variation_name=f"{dataset_id}@@{variation_id}",
    )

    variation_dir = base_output_dir / dataset_id / variation_id
    variation_dir.mkdir(parents=True, exist_ok=True)

    # Write summary JSONL
    summary_path = variation_dir / "triplet_generation_summary.jsonl"
    summarize_triplets(
        triplets=triplets,
        out_path=summary_path,
        variation_name=variation_id,
        dataset_name=dataset_id,
        pos_params=pos_params,
        neg_params=neg_params,
        avg_delta_pos=avg_delta_pos,
        avg_delta_neg=avg_delta_neg,
    )

    # Write one CSV per table in each triplet (anchor / pos / neg)
    for triplet_id, t in enumerate(triplets):
        for kind in ["anchor", "pos", "neg"]:
            table_key = f"{kind}_table"
            table_df = table_2d_to_df(t.get(table_key))
            csv_path = variation_dir / f"{triplet_id}_{kind}.csv"
            table_df.to_csv(csv_path, index=False)

    logger.info(f"[{dataset_id}@@{variation_id}] Generated {len(triplets)} triplets (avg delta_text pos={avg_delta_pos:.4f}, neg={avg_delta_neg:.4f})")


def load_table_shuffling_context():
    cfg = load_table_shuffling_config()
    root_seed = int(cfg["random_seed"])

    try:
        project_root = Path(get_original_cwd())
    except ValueError:
        project_root = Path.cwd()

    base_output_dir = project_root / "cache" / "table_shuffling"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    datasets_cfg = cfg["datasets"]
    variations_cfg = cfg["variations"]

    if not datasets_cfg:
        raise ValueError("No datasets defined in table_shuffling config.")
    if not variations_cfg:
        raise ValueError("No variations defined in table_shuffling config.")

    return root_seed, base_output_dir, datasets_cfg, variations_cfg


def run_variation_by_name(dataset_id: str, variation_id: str) -> None:
    root_seed, base_output_dir, datasets_cfg, variations_cfg = load_table_shuffling_context()

    if dataset_id not in datasets_cfg:
        available = ", ".join(sorted(datasets_cfg.keys()))
        raise ValueError(
            f"Unknown table_shuffling dataset_id '{dataset_id}'. Available datasets: {available}"
        )
    if variation_id not in variations_cfg:
        available = ", ".join(sorted(variations_cfg.keys()))
        raise ValueError(
            f"Unknown table_shuffling variation_id '{variation_id}'. Available variations: {available}"
        )

    run_variation(
        dataset_id=dataset_id,
        dataset_cfg=datasets_cfg[dataset_id],
        variation_id=variation_id,
        variation_cfg=variations_cfg[variation_id],
        base_output_dir=base_output_dir,
        root_random_seed=root_seed,
    )


def main() -> None:
    root_seed, base_output_dir, datasets_cfg, variations_cfg = load_table_shuffling_context()

    for dataset_id, dataset_cfg in datasets_cfg.items():
        for variation_id, variation_cfg in variations_cfg.items():
            logger.info(f"Running dataset='{dataset_id}', variation='{variation_id}'")
            run_variation(
                dataset_id=dataset_id,
                dataset_cfg=dataset_cfg,
                variation_id=variation_id,
                variation_cfg=variation_cfg,
                base_output_dir=base_output_dir,
                root_random_seed=root_seed,
            )

if __name__ == "__main__":
    main()