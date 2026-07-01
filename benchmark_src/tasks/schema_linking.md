# NL2Column and NL2Cell2Column Tasks

Three tasks for evaluating schema linking on BIRD, all using the same task runner (`run_NL2*_mapping_benchmark.py`).

## Tasks

| Task | Evaluates | Benchmark file |
|---|---|---|
| `nl2column_mapping` | Column embeddings — concept-level NL→column matching (no cell values) | `pure_concept_mapping_queries.json` |
| `nl2cell2column_mapping` | Cell embeddings — exact value matching (NL extracted values → DB cells → columns) | `cell_value_matching_queries.json` |
| `nl2cell2column_fuzzy_mapping` | Cell embeddings — fuzzy/semantic value matching | `semantic_fuzzy_matching.json` |

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
