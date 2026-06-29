# NL2Cell2Column Mapping Benchmark - Changes Summary

## Overview
Updated `run_NL2cell2column_mapping_benchmark.py` to improve efficiency and support both exact and fuzzy cell matching benchmarks.

## Key Changes

### 1. Unique Value Sampling with SQL
**Function Added:** `get_unique_column_values()`
- Uses SQL `DISTINCT` to get unique values from each column before sampling
- Filters out NULL values
- Significantly reduces redundant embeddings for columns with duplicate values
- More efficient than loading all rows and then deduplicating

```python
query = f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit}'
```

### 2. Always Include Matched Values
**Function Added:** `collect_matched_values_for_database()`
- Collects all matched values from benchmark queries for a specific database
- Groups values by `table.column` for efficient lookup
- Ensures matched values are ALWAYS included in embeddings, regardless of sampling limit

**Updated:** `create_qdrant_collection_for_database()`
- Now accepts `matched_values_by_column` parameter
- Combines unique values from database with matched values from queries
- Marks embeddings with `is_matched_value` flag in metadata

```python
# Combine unique values with matched values
all_values = set(unique_values)
all_values.update(matched_values)  # Always include matched values
```

### 3. Support for Fuzzy Cell Matching
**Configuration Updated:** `benchmark_src/config/dataset/bird.yaml`
- Added `fuzzy_cell_matching.json` as a supported benchmark file
- This file contains fuzzified cell values to test semantic (non-exact) matching

```yaml
# Options:
#   - cell_value_matching_queries.json (exact matching)
#   - fuzzy_cell_matching.json (semantic/fuzzy matching)
nl2cell2column_benchmark_file: "cell_value_matching_queries.json"
```

### 4. Configuration Parameter Update
**Changed:** `max_rows_per_table` → `max_unique_values_per_column`
- More accurate parameter name reflecting the new behavior
- Limits unique values per column, not rows per table
- Default: 1000 unique values per column

**File:** `benchmark_src/config/task/nl2cell2column_mapping.yaml`

## Benefits

1. **Efficiency**: Only embeds unique values, avoiding redundant embeddings
2. **Accuracy**: Always includes matched values from benchmark, ensuring fair evaluation
3. **Flexibility**: Supports both exact and fuzzy/semantic matching benchmarks
4. **Scalability**: Better memory usage for large tables with many duplicate values

## Usage

### For Exact Matching (default):
```yaml
# In bird.yaml
nl2cell2column_benchmark_file: "cell_value_matching_queries.json"
```

### For Fuzzy/Semantic Matching:
```yaml
# In bird.yaml
nl2cell2column_benchmark_file: "fuzzy_cell_matching.json"
```

### Adjust Sampling Limit:
```yaml
# In nl2cell2column_mapping.yaml
max_unique_values_per_column: 1000  # Adjust as needed
```

## Technical Details

### Embedding Process (Per Column):
1. Query database for unique values using SQL DISTINCT
2. Collect matched values from benchmark queries for this column
3. Merge unique values with matched values (matched values always included)
4. Create embeddings for all combined values
5. Store in Qdrant with metadata including `is_matched_value` flag

### Metadata Stored:
- `table`: Table name
- `column`: Column name
- `table_column`: Combined "table.column" identifier
- `value`: The actual cell value (string)
- `is_matched_value`: Boolean flag indicating if from benchmark queries

## Backward Compatibility

The changes maintain backward compatibility:
- Default benchmark file remains `cell_value_matching_queries.json`
- Default sampling limit is 1000 (same effective behavior)
- All existing metrics and evaluation logic unchanged