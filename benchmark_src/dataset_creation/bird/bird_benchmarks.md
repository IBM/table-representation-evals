# BIRD Schema Linking Benchmarks

Three benchmark files derived from the BIRD training set (9,428 queries), each targeting a different schema linking capability. All files use the same JSON format and are consumed by the `nl2column_mapping` and `nl2cell2column_mapping` tasks.

## Benchmark files

| File | Entries | Task |
|---|---|---|
| `pure_concept_mapping_queries.json` | 2,974 | Concept-level NL→column (no cell values needed) |
| `cell_value_matching_queries.json` | 1,317 | Exact cell value matching (NL value == DB value) |
| `semantic_fuzzy_matching.json` | 195 | Fuzzy/semantic value matching (NL value ≠ DB value) |


---

## How each benchmark was created

### Concept and exact benchmarks

`analyze_queries.py` parses every BIRD training query using **sqlparse** to extract literal values from WHERE/HAVING clauses, then checks whether each value appears verbatim in the NL question or evidence. Queries are categorised as:

- **concept-only** — all required columns can be found from schema names alone, no cell value lookup needed
- **exact match** — at least one column is identified because the user mentions its value verbatim (e.g., "released in 2007")
- **fuzzy match** — at least one column requires approximate matching (SQL value ≠ NL value)
- **unknown** — complex subqueries or expressions where columns could not be extracted

### Semantic fuzzy benchmark

`semantic_fuzzy_matching.json` contains 195 hand-curated entries that represent genuine semantic gaps, filtered to exclude boolean flags, arbitrary internal codes, near-identical strings differing only by punctuation, and other noise.

**Two sources:**

1. **BIRD evidence field** — BIRD annotators wrote evidence like `"women refers to gender = 'Female'"` to explain NL→DB mappings. `create_semantic_fuzzy_benchmark.py` parses this pattern with a regex, checks the NL phrase appears in the question and the DB value does *not* (ruling out exact matches), resolves the gold column from the SQL AST, and verifies the DB value exists in the SQLite database.

2. **Typo entries (hardcoded)** — 45 curated entries where the NL phrase is a plausible misspelling of the DB value (e.g. "Goergia" → "Georgia"). These were originally mined from BIRD training data using a character overlap ratio threshold (>0.5) with NL-in-question / DB-not-in-question / DB-in-DB verification, then reviewed and hardcoded directly in `create_semantic_fuzzy_benchmark.py` as `_TYPO_ENTRIES`.

**Quality filtering** discards: boolean/coded values, pure numbers, strings >80 chars, case/punctuation-only differences, pluralisation-only differences, vague phrases ("average", "most"), question fragments, geographic container→country mismatches, verbatim DB text embedded in NL, and pairs where NL and DB share a long prefix but then diverge to different entities.

**Deduplication:** capped at 2 entries per unique (NL, DB) value pair; date_format entries capped at 3 per column.

**Type breakdown:**

| Type | Count | Example |
|---|---|---|
| `synonym` | 89 | "female" → "F", "United States" → "USA" |
| `typo` | 45 | "Goergia" → "Georgia", "colour" → "color" |
| `abbreviation` | 35 | "Tennessee" → "TN", "PostgreSQL" → "PG" |
| `date_format` | 26 | "2010/08/01" → "2010-08-01" |

---

## Data format

All files share the same structure:

```json
{
  "db_id": "database_name",
  "question": "Natural language question",
  "evidence": "Additional hints from BIRD",
  "SQL": "Ground truth SQL",
  "used_tables": ["table1"],
  "gold_columns": [{"table": "t", "column": "c"}],
  "matched_values": ["USA"],
  "extracted_values_from_NL": ["United States"]
}
```

- `gold_columns` — columns identifiable via cell values (fuzzy/exact files); all relevant columns (concept file)
- `matched_values` — actual DB-side values
- `extracted_values_from_NL` — NL-side values (identical to `matched_values` for exact; differs for fuzzy)
- `fuzzy_type` — present only in `semantic_fuzzy_matching.json`: `synonym`, `typo`, `abbreviation`, or `date_format`

---

## Regenerating

```bash
# Concept + exact benchmarks (reads cache/datasets/bird/train/)
python benchmark_src/dataset_creation/bird/analyze_queries.py

# Semantic fuzzy benchmark
python benchmark_src/dataset_creation/bird/create_semantic_fuzzy_benchmark.py
```

Analyse benchmark statistics (pass two files to compare versions):
```bash
python benchmark_src/dataset_creation/bird/analyze_fuzzy_benchmark.py \
    benchmark_src/dataset_creation/bird/semantic_fuzzy_matching.json
```

---

## Config

Each query set has its own dataset config under `configs/dataset/`:
- `bird_column_schema.yaml` → `pure_concept_mapping_queries.json`
- `bird_cell_exact.yaml` → `cell_value_matching_queries.json`
- `bird_cell_fuzzy.yaml` → `semantic_fuzzy_matching.json`

`bird_cell_exact` and `bird_cell_fuzzy` share the same Qdrant cell index via `qdrant_cache_key: bird_cell`, so cell embeddings are only built once when running both datasets.
