# Schema Linking and Value Linking Tasks

Two tasks for evaluating schema linking on BIRD, run by `run_schema_linking_benchmark.py` and
`run_value_linking_benchmark.py` respectively.

## Tasks and datasets

| Task | Dataset | Benchmark file | Queries |
|---|---|---|---|
| `schema_linking` | `bird_column_schema` | `pure_concept_mapping_queries.json` | 2,974 |
| `value_linking` | `bird_cell_exact` | `cell_value_matching_queries.json` | 1,317 |
| `value_linking` | `bird_cell_fuzzy` | `semantic_fuzzy_matching.json` | 195 |

`value_linking` runs on two datasets. `bird_cell_exact` covers queries where the NL value matches the DB value verbatim; `bird_cell_fuzzy` covers cases with synonyms, typos, abbreviations, and date format differences. Both datasets share the same Qdrant cell index (configured via `qdrant_cache_key` in the dataset configs), so cell embeddings are only built once.

## Running

```bash
bash run.sh schema_and_value_linking
```

The `schema_and_value_linking` run config covers both tasks. For individual tasks, reference them in any run config under `tasks:`.

## Key config parameters

**schema_linking** (`configs/task/schema_linking.yaml`):
- `top_k_columns` — columns retrieved per query (default 50)
- `max_rows_per_table` — rows loaded per table for column embedding (default 1000)

**value_linking** (`configs/task/value_linking.yaml`):
- `top_k_columns` — columns retrieved per query via `search_groups` (default 50)
- `max_unique_values_per_column` — unique values indexed per column (default 1000; matched values from benchmark are always included regardless)
- `query_mode` — `extracted_values` (default) or `full_nl`

## Cache reuse

Cell embeddings are cached by `{approach}_{model}_{dataset}` in `cache/qdrant_storage/`. The exact and fuzzy datasets share the same Qdrant collection, so running both with the same approach only embeds cell values once.
