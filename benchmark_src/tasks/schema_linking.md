# NL2Column and NL2Cell2Column Tasks

Three tasks for evaluating schema linking on BIRD, all using the same task runner (`run_NL2*_mapping_benchmark.py`).

## Tasks and datasets

| Task | Dataset | Benchmark file | Queries |
|---|---|---|---|
| `nl2column_mapping` | `bird_column_schema` | `pure_concept_mapping_queries.json` | 2,974 |
| `nl2cell2column_mapping` | `bird_cell_exact` | `cell_value_matching_queries.json` | 1,317 |
| `nl2cell2column_mapping` | `bird_cell_fuzzy` | `semantic_fuzzy_matching.json` | 195 |

`nl2cell2column_mapping` runs on two datasets. `bird_cell_exact` covers queries where the NL value matches the DB value verbatim; `bird_cell_fuzzy` covers cases with synonyms, typos, abbreviations, and date format differences. Both datasets share the same Qdrant cell index (configured via `qdrant_cache_key` in the dataset configs), so cell embeddings are only built once.

## Running

```bash
python run_experiments.py schema_linking
```

The `schema_linking` run config covers all three tasks. For individual tasks, reference them in any run config under `tasks:`.

## Key config parameters

**nl2column** (`configs/task/nl2column_mapping.yaml`):
- `top_k_columns` — columns retrieved per query (default 50)
- `max_rows_per_table` — rows loaded per table for column embedding (default 1000)

**nl2cell2column / nl2cell2column_fuzzy** (`configs/task/nl2cell2column_*.yaml`):
- `top_k_columns` — columns retrieved per query via `search_groups` (default 50)
- `max_unique_values_per_column` — unique values indexed per column (default 1000; matched values from benchmark are always included regardless)
- `query_mode` — `extracted_values` (default) or `full_nl`

## Cache reuse

Cell embeddings are cached by `{approach}_{model}_{dataset}` in `cache/qdrant_storage/`. The exact and fuzzy tasks share the same Qdrant collection, so running both with the same approach only embeds cell values once.
