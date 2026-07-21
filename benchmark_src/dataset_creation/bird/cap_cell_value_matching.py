"""
Cap cell_value_matching_queries.json down from the full categorized pool
produced by analyze_queries.py to a size comparable to the benchmark's
historical scale, without an arbitrary random sample.

Two deterministic caps are applied, in order:
1. At most 2 queries per (db_id, SQL template) -- SQL with string/number
   literals blanked out -- to remove near-duplicate query patterns that
   differ only in which literal value they filter on (e.g. the same
   "WHERE city = ?" query repeated for many different cities).
2. At most MAX_PER_DB queries per database, to keep a few large/verbose
   databases (e.g. public_review_platform, works_cycles) from dominating
   the benchmark.

Run from repo root, after analyze_queries.py has produced
cell_value_matching_queries.json:
    python benchmark_src/dataset_creation/bird/cap_cell_value_matching.py
"""

import json
import re
from collections import Counter
from pathlib import Path

INPUT_FILE = Path("benchmark_src/dataset_creation/bird/cell_value_matching_queries.json")
OUTPUT_FILE = Path("benchmark_src/dataset_creation/bird/cell_value_matching_queries.json")

MAX_PER_TEMPLATE = 2
MAX_PER_DB = 20

_STRING_LITERAL_RE = re.compile(r"'[^']*'")
_NUMBER_LITERAL_RE = re.compile(r"\b\d+\b")


def normalize_sql(sql):
    s = _STRING_LITERAL_RE.sub("?", sql)
    s = _NUMBER_LITERAL_RE.sub("?", s)
    return s


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries from {INPUT_FILE}")

    template_counts = Counter()
    deduped = []
    for q in queries:
        key = (q["db_id"], normalize_sql(q["SQL"]))
        if template_counts[key] < MAX_PER_TEMPLATE:
            template_counts[key] += 1
            deduped.append(q)
    print(f"After template dedup (max {MAX_PER_TEMPLATE} per db+template): {len(deduped)}")

    db_counts = Counter()
    capped = []
    for q in deduped:
        if db_counts[q["db_id"]] < MAX_PER_DB:
            db_counts[q["db_id"]] += 1
            capped.append(q)
    print(f"After per-db cap (max {MAX_PER_DB} per db_id): {len(capped)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(capped, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(capped)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
