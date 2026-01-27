import json
import random
import string
from typing import List, Dict, Any, Optional
# from benchmark_src.dataset_creation.target.collect_all_target_datasets import get_target_dataset_by_name


# -----------------------------
# Synthetic table generation
# -----------------------------
def random_string(k=5):
    return "".join(random.choices(string.ascii_lowercase, k=k))

def generate_random_table(num_rows: int, num_cols: int):
    col_types = [random.choice(["int", "float", "str"]) for _ in range(num_cols)]

    table = []
    for _ in range(num_rows):
        row = []
        for col_type in col_types:
            if random.random() < 0.05:
                cell = None
            elif col_type == "int":
                cell = random.randint(0, 100)
            elif col_type == "float":
                cell = random.random()
            elif col_type == "str":
                cell = random_string()
            else:
                raise ValueError(f"Unknown column type: {col_type}")

            row.append(cell)
        table.append(row)

    return table

def normalize_table(table: List[List[Any]]) -> List[List[Optional[str]]]:
    return [
        [str(cell) if cell is not None else None for cell in row]
        for row in table
    ]

def generate_synthetic_table_dataset(
    num_databases: int = 1,
    tables_per_db: int = 10,
    min_rows: int = 3,
    max_rows: int = 6,
    min_cols: int = 3,
    max_cols: int = 6,
):
    rows = []
    for db_idx in range(num_databases):
        database_id = f"db_{db_idx}"
        for t_idx in range(tables_per_db):
            table_id = f"table_{t_idx}"
            table = generate_random_table(
                num_rows=random.randint(min_rows, max_rows),
                num_cols=random.randint(min_cols, max_cols),
            )
            table = normalize_table(table)
            rows.append({
                "database_id": database_id,
                "table_id": table_id,
                "table": table
            })
    return rows

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
        print("ERROR: Empty table array provided.") #TODO: change the whole file to use logging
        return ""

    headers = [str(h) for h in table_array[0]]
    lines: List[str] = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    data_rows = table_array[1:]

    if max_rows != -1 and len(data_rows) > max_rows:
        data_rows = data_rows[:max_rows]
        # print(f"Limiting table (with {len(table_array)-1} rows) to first {max_rows} rows.")

    for row in data_rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")

    # Join all lines with a newline and add a final newline
    return "\n".join(lines) + "\n"

def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1,
                         cur[j-1] + 1,
                         prev[j-1] + cost)
        prev = cur
    return prev[lb]

# -----------------------------
# Textual-change metric: normalized Levenshtein
# -----------------------------
def normalized_levenshtein(a: str, b: str) -> float:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(a, b) / max_len


def generate_positive(table: List[List[Optional[str]]], pos_type: str, pos_strength: float):
    assert 0.0 <= pos_strength <= 1.0
    n_rows = len(table)
    n_cols = len(table[0]) if n_rows > 0 else 0

    # Work on deep copy
    T = [row[:] for row in table]

    # If pos_strength is fractional and we want arbitrary rows, do:
    if pos_type in ("row_reorder", "both") and n_rows > 0:
        row_indices = list(range(n_rows))
        k = max(1, int(round(pos_strength * n_rows))) if pos_strength > 0 else 0
        if k > 1:
            chosen = random.sample(row_indices, k)
            permuted = chosen[:]
            random.shuffle(permuted)
            # apply permutation to chosen indices
            new_T = T[:]
            for src_i, dst_i in zip(chosen, permuted):
                new_T[dst_i] = T[src_i]
            T = new_T

    if pos_type in ("col_reorder", "both") and n_cols > 0:
        col_indices = list(range(n_cols))
        k = max(1, int(round(pos_strength * n_cols))) if pos_strength > 0 else 0
        if k > 1:
            chosen_cols = random.sample(col_indices, k)
            permuted = chosen_cols[:]
            random.shuffle(permuted)
            # build new table with columns permuted among chosen positions
            new_T = []
            for r in T:
                new_row = r[:]  # will be replaced for chosen cols
                for src_c, dst_c in zip(chosen_cols, permuted):
                    new_row[dst_c] = r[src_c]
                new_T.append(new_row)
            T = new_T

    return T

def generate_negative(table: List[List[Optional[str]]], neg_columns_frac: float, neg_degree: float):
    # neg_columns_frac: fraction of columns to touch
    # neg_degree: 0..1, indicates how shuffled each chosen column becomes
    assert 0.0 <= neg_columns_frac <= 1.0
    assert 0.0 <= neg_degree <= 1.0

    n_rows = len(table)
    n_cols = len(table[0]) if n_rows > 0 else 0
    T = [row[:] for row in table]

    if n_cols == 0 or n_rows == 0:
        return T

    num_cols_to_perm = max(1, int(round(neg_columns_frac * n_cols))) if neg_columns_frac > 0 else 0
    cols = list(range(n_cols))
    chosen_cols = random.sample(cols, num_cols_to_perm) if num_cols_to_perm > 0 else []

    for c in chosen_cols:
        # perform a number of random swaps proportional to neg_degree
        col_vals = [T[r][c] for r in range(n_rows)]
        S = max(1, int(round(neg_degree * n_rows))) if neg_degree > 0 else 0
        for _ in range(S):
            i, j = random.randrange(n_rows), random.randrange(n_rows)
            col_vals[i], col_vals[j] = col_vals[j], col_vals[i]
        for r in range(n_rows):
            T[r][c] = col_vals[r]

    return T

# -----------------------------
# Triplet generation
# -----------------------------
def generate_triplets_from_dataset(
    dataset: List[Dict[str, Any]],
    triplets_per_anchor: int,
    pos_params: Dict[str, Any],
    neg_params: Dict[str, Any],
):
    triplets = []
    deltas_pos = []
    deltas_neg = []

    for anchor_rec in dataset:
        anchor_table = anchor_rec["table"]
        serialized_anchor = convert_array_to_markdown(anchor_table)

        for _ in range(triplets_per_anchor):
            pos_table = generate_positive(anchor_table, pos_params["pos_type"], pos_params["pos_strength"])
            neg_table = generate_negative(anchor_table, neg_params["neg_columns_frac"], neg_params["neg_degree"])

            s_pos = convert_array_to_markdown(pos_table)
            s_neg = convert_array_to_markdown(neg_table)

            delta_pos = normalized_levenshtein(serialized_anchor, s_pos)
            delta_neg = normalized_levenshtein(serialized_anchor, s_neg)

            triplets.append({
                "database_id": anchor_rec["database_id"],
                "table_id": anchor_rec["table_id"],
                "anchor_table": anchor_table,
                "pos_table": pos_table,
                "neg_table": neg_table,
                "s_anchor": serialized_anchor,
                "s_pos": s_pos,
                "s_neg": s_neg,
                "delta_pos": delta_pos,
                "delta_neg": delta_neg,
            })
            deltas_pos.append(delta_pos)
            deltas_neg.append(delta_neg)

    avg_delta_pos = sum(deltas_pos) / len(deltas_pos) if deltas_pos else 0.0
    avg_delta_neg = sum(deltas_neg) / len(deltas_neg) if deltas_neg else 0.0

    return triplets, avg_delta_pos, avg_delta_neg

def main():
    config = {
        "dataset_name": "fetaqa",
        "random_seed": 42,
        "dataset": {
            "num_databases": 1,
            "tables_per_db": 20,
            "min_rows": 10,
            "max_rows": 50,
            "min_cols": 3,
            "max_cols": 10
        },
        "triplets_per_anchor": 1,
        "pos_params": {
            "pos_type": "row_reorder",   # one of "row_reorder", "col_reorder", "both"
            "pos_strength": 0.5          # fraction [0..1] of rows/cols to shuffle
        },
        "neg_params": {
            "neg_columns_frac": 0.3,     # fraction [0..1] of columns to permute
            "neg_degree": 0.8            # [0..1] how shuffled each chosen column is
        }
    }

    random.seed(config["random_seed"])

    print("Generating dataset...")

    dataset = generate_synthetic_table_dataset(
        num_databases=config["dataset"]["num_databases"],
        tables_per_db=config["dataset"]["tables_per_db"],
        min_rows=config["dataset"]["min_rows"],
        max_rows=config["dataset"]["max_rows"],
        min_cols=config["dataset"]["min_cols"],
        max_cols=config["dataset"]["max_cols"],
    )

    print("Generating triplets...")
    triplets, avg_delta_pos, avg_delta_neg = generate_triplets_from_dataset(
        dataset=dataset,
        triplets_per_anchor=config["triplets_per_anchor"],
        pos_params=config["pos_params"],
        neg_params=config["neg_params"]
    )

    print(f"Generated {len(triplets)} triplets")
    print(f"Average normalized textual-change Δ_text (pos): {avg_delta_pos:.4f}")
    print(f"Average normalized textual-change Δ_text (neg): {avg_delta_neg:.4f}")

    out_summary = [{
        "database_id": t["database_id"],
        "table_id": t["table_id"],
        "s_anchor": t["s_anchor"],
        "s_pos": t["s_pos"],
        "s_neg": t["s_neg"],
        "delta_pos": t["delta_pos"],
        "delta_neg": t["delta_neg"],
    } for t in triplets]

    with open("triplet_generation_summary.jsonl", "w", encoding="utf-8") as fo:
        for rec in out_summary:
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Wrote triplet textual summary to triplet_generation_summary.jsonl")

if __name__ == "__main__":
    main()
