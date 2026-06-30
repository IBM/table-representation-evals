# NL2Column and NL2Cell2Column Mapping Tasks

Quick reference for running Text-to-SQL column mapping benchmarks using the BIRD dataset.

## Tasks Overview

### 1. NL2Column Mapping (Concept Mapping)
**What it does:** Maps natural language queries to relevant database columns using only column names and schemas (no cell values).

**Benchmark file:** `pure_concept_mapping_queries.json`

**Run command:**
```bash
python benchmark_src/run_benchmark.py experiment=nl2column_exact_matching
```

---

### 2. NL2Cell2Column Mapping - Exact Match
**What it does:** Maps natural language queries to columns by finding exact cell value matches in the database.

**Benchmark file:** `cell_value_matching_queries.json`

**Run command:**
```bash
python benchmark_src/run_benchmark.py experiment=nl2cell2column_exact_matching
```

---

### 3. NL2Cell2Column Mapping - Fuzzy Match
**What it does:** Maps natural language queries to columns using fuzzy/semantic cell value matching.

**Benchmark file:** `fuzzy_cell_matching.json`

**Run command:**
```bash
python benchmark_src/run_benchmark.py experiment=nl2cell2column_fuzzy_matching
```

---

## Cache Structure

### Cache Location
```
cache/
├── qdrant_storage/
│   ├── qdrant_nl2column_{approach}_{model}_{dataset}/
│   │   └── collection/
│   │       └── bird_columns_{db_id}/
│   │           ├── COMPLETED (marker file)
│   │           └── [Qdrant vector DB files]
│   │
│   └── qdrant_cell_to_column_{approach}_{model}_{dataset}/
│       └── collection/
│           └── bird_cells_{db_id}/
│               ├── COMPLETED (marker file)
│               └── [Qdrant vector DB files]
│
├── datasets/ (downloaded datasets)
└── models/ (downloaded model weights)
```

### Cache Key Format
`{approach}_{model}_{dataset}`

**Example:** `sentence_transformer_all-MiniLM-L6-v2_bird`

### Cache Reuse
- **Column embeddings** are cached per database in `qdrant_nl2column_*` collections
- **Cell embeddings** are cached per database in `qdrant_cell_to_column_*` collections
- Exact and fuzzy matching experiments **share the same cell embedding cache**
- Cache persists across runs with the same approach, model, and dataset

---

## Results Location

Results are saved to:
```
results/
└── results_per_task/
    ├── nl2column_mapping/
    │   ├── nl2column_mapping_results.xlsx
    │   └── nl2column_mapping_resources.xlsx
    │
    ├── nl2cell2column_mapping/
    │   ├── nl2cell2column_mapping_results.xlsx
    │   └── nl2cell2column_mapping_resources.xlsx
    │
    └── nl2cell2column_fuzzy_mapping/
        ├── nl2cell2column_fuzzy_mapping_results.xlsx
        └── nl2cell2column_fuzzy_mapping_resources.xlsx
```

---

## Configuration

Edit experiment configs in `approaches/configs/experiment/` to:
- Change embedding model: `approach.embedding_model`
- Adjust query limits: `max_queries`
- Modify retrieval parameters: `top_k_cells`, `top_k_columns`
- Change max rows per table: `max_rows_per_table`