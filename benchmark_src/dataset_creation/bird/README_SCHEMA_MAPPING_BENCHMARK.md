# BIRD Text2SQL Schema Mapping Benchmark

## Overview
This project categorizes the BIRD benchmark queries to create specialized benchmarks for testing schema mapping components of text2sql systems. The queries are split into three main categories based on how columns can be identified:

1. **Column2Schema Mapping (Pure Concept Mapping)**: Columns mapped through conceptual understanding (NO cell-level information needed)
2. **Exact Cell Value Matching**: Columns identified through exact cell values mentioned in the question
3. **Fuzzy Cell Value Matching**: Columns identified through fuzzy/approximate matching of cell values

## Results Summary

From 9,428 total queries in the BIRD training set:

- **Column2Schema Mapping (Pure Concept)**: 2,974 queries (31.54%)
- **Exact Cell Value Matching**: 1,317 queries (13.97%)
- **Fuzzy Cell Value Matching**: 607 queries (6.44%)
- **Unknown/Unparsed**: 4,530 queries (48.05%)

### Detailed Breakdown
- **Concept-Only**: 2,974 queries (all columns mapped by concept alone)
- **Exact Value Matching**: 1,317 queries (columns found via exact cell value matches)
- **Fuzzy Value Matching**: 607 queries (columns found via fuzzy/approximate cell value matches)
- **Total Cell Value Matching**: 1,924 queries (Exact + Fuzzy = 20.41%)

**Note**: The total Cell Value Matching category (1,924) includes both exact matching (1,317) and fuzzy matching (607) queries. Both require the ability to identify columns through cell values mentioned in the question, but fuzzy matching requires additional normalization/similarity matching capabilities.

## Key Distinctions

The critical differences between the three categories:

### 1. Column2Schema Mapping (Pure Concept Mapping)
Columns can be identified purely through semantic understanding of the question and schema, without needing to look at actual data values.

**Example:**
```
Question: "State the most popular movie? When was it released and who is the director?"
Required Columns:
  - movies.movie_title (concept: "movie" → table, "title" → column)
  - movies.movie_release_year (concept: "when released" → release year)
  - movies.director_name (concept: "director" → director name)
  - movies.movie_popularity (concept: "most popular" → popularity metric)

All columns identified through conceptual understanding alone.
```

### 2. Exact Cell Value Matching
At least one column is identified because a specific value from that column is mentioned in the question, and the value in the question exactly matches the value in the database.

**Example:**
```
Question: "How many films were released in 2007?"
Required Columns:
  - movies.movie_release_year (value: "2007" in question matches "2007" in database)

The mention of "2007" directly matches the cell value in the database.
```

**Another Example:**
```
Question: "How many users gave 'Pavee Lackeen: The Traveller Girl' movie a rating score of 4?"
Required Columns:
  - movies.movie_title (value: exact movie title match)
  - ratings.rating_score (value: "4" matches exactly)
  - ratings.user_id (concept: "how many users" → count users)
```

### 3. Fuzzy Cell Value Matching
At least one column is identified through approximate/fuzzy matching where the value mentioned in the question differs from the actual database value but refers to the same entity.

**Example:**
```
Question: "Who are the employees working for publisher not located in United States?"
Database Value: "USA"
Question Value: "United States"
Required Columns:
  - publishers.country (fuzzy match: "United States" → "USA")

The system must recognize that "United States" and "USA" refer to the same entity.
```

**Another Example:**
```
Question: "What is the average number of Mubi users who love movies directed by Stanley Kubrick?"
Database Value: "S. Kubrick" (hypothetical)
Question Value: "Stanley Kubrick"
Required Columns:
  - movies.director_name (fuzzy match: "Stanley Kubrick" → "S. Kubrick")

The system must perform fuzzy matching to identify the correct column.
```

## Output Files

### 1. `pure_concept_mapping_queries.json`
Contains 2,974 queries where ALL columns can be mapped purely through conceptual understanding (Column2Schema mapping).

**Use Case**: Test a system's ability to:
- Map natural language concepts to schema elements
- Understand semantic relationships between entities
- Identify appropriate tables and columns without data access
- Handle aggregations, comparisons, and operations conceptually

**Structure:**
```json
{
  "db_id": "database_name",
  "question": "Natural language question",
  "evidence": "Additional hints/context",
  "SQL": "Ground truth SQL query",
  "category": "concept_only",
  "used_tables": ["table1", "table2"],
  "gold_columns": [
    {"table": "table1", "column": "column1"},
    {"table": "table1", "column": "column2"}
  ],
  "matched_values": []
}
```

### 2. `cell_value_matching_queries.json`
Contains 1,317 queries where at least one column is identified through **exact** cell value matching.

**Use Case**: Test a system's ability to:
- Recognize entity mentions in questions
- Match exact values to appropriate columns
- Identify columns based on specific data values
- Combine value-based and concept-based column identification

**Structure:**
```json
{
  "db_id": "database_name",
  "question": "Natural language question",
  "evidence": "Additional hints/context",
  "SQL": "Ground truth SQL query",
  "category": "value_only" | "mixed",
  "used_tables": ["table1", "table2"],
  "gold_columns": [
    {"table": "table1", "column": "column1"}
  ],
  "matched_values": ["2007", "4"],
  "extracted_values_from_NL": ["2007", "4"]
}
```

**Important**:
- `gold_columns` contains ONLY the columns that can be identified through cell values mentioned in the question
- `matched_values` contains the actual values from the SQL query
- `extracted_values_from_NL` contains the values extracted from the natural language question
- For exact matching, `matched_values` and `extracted_values_from_NL` are identical

### 3. `fuzzy_cell_matching.json`
Contains 607 queries where at least one column is identified through **fuzzy/approximate** cell value matching.

**Use Case**: Test a system's ability to:
- Perform fuzzy string matching between question values and database values
- Handle variations in entity names (e.g., "United States" vs "USA")
- Normalize and match abbreviated forms
- Apply similarity metrics for value matching
- Handle typos and alternative representations

**Structure:**
```json
{
  "db_id": "database_name",
  "question": "Natural language question",
  "evidence": "Additional hints/context",
  "SQL": "Ground truth SQL query",
  "category": "value_only" | "mixed",
  "used_tables": ["table1", "table2"],
  "gold_columns": [
    {"table": "publishers", "column": "country"}
  ],
  "matched_values": ["USA"],
  "extracted_values_from_NL": ["United States"]
}
```

**Important**:
- `matched_values` contains the actual values from the SQL query (database values)
- `extracted_values_from_NL` contains the values from the natural language question
- For fuzzy matching, `matched_values` ≠ `extracted_values_from_NL` (they differ but refer to the same entity)
- This tests the system's ability to recognize that different representations refer to the same entity

### 4. `categorization_summary.json`
Contains statistics and sample queries from each category.

## Methodology

The categorization algorithm uses **sqlparse**, a robust SQL parser library:

1. **Parse SQL Query**: Use sqlparse to properly parse the SQL statement into an AST
2. **Extract Tables**: Identify all tables from FROM and JOIN clauses
3. **Extract Columns**: Identify all columns referenced throughout the query
4. **Extract Literal Values**: Find all string and numeric literals in WHERE/HAVING clauses using token types
5. **Extract Values from Natural Language**: Extract potential entity values from the question and evidence text
6. **Match Values to Question**: Check if each SQL literal appears in the question or evidence
7. **Identify Value-Dependent Columns**: Use the parsed Comparison nodes to determine which columns are filtered by matched values
8. **Categorize**:
   - If all columns are concept-based → **Column2Schema Mapping (Pure Concept)**
   - If SQL values exactly match NL values → **Exact Cell Value Matching**
   - If SQL values differ from NL values but refer to same entity → **Fuzzy Cell Value Matching**

### Fuzzy vs Exact Matching Detection

The system distinguishes between exact and fuzzy matching by comparing:
- `matched_values`: Values extracted from the SQL query
- `extracted_values_from_NL`: Values extracted from the natural language question

**Exact Match**: When `matched_values[i] == extracted_values_from_NL[i]`
- Example: Question has "2007", SQL has `WHERE year = 2007`

**Fuzzy Match**: When `matched_values[i] != extracted_values_from_NL[i]` but they refer to the same entity
- Example: Question has "United States", SQL has `WHERE country = 'USA'`

This approach is much more reliable than regex-based parsing and handles complex SQL constructs properly.

## Schema Information

Each query includes:
- `db_id`: Database identifier
- `question`: Natural language question
- `evidence`: Additional hints/context
- `SQL`: Ground truth SQL query
- `category`: Query type (concept_only, value_only, or mixed)
- `used_tables`: List of tables needed for the query
- `gold_columns`: **Gold standard columns for benchmarking**
  - For concept_only: all columns needed (test concept mapping)
  - For value_only/mixed: only columns identified via cell values (test value matching)
- `matched_values`: Specific values from the SQL query (database values)
- `extracted_values_from_NL`: Values extracted from the natural language question

This information allows benchmarks to test:
- **Column Selection**: Can the system identify the right columns?
- **Table Selection**: Can the system identify the right tables?
- **Exact Value Matching**: Can the system match exact entity values to columns?
- **Fuzzy Value Matching**: Can the system match approximate/normalized entity values to columns?
- **Concept Mapping**: Can the system understand semantic relationships?

### Key Field Distinctions

| Field | Description | Example |
|-------|-------------|---------|
| `matched_values` | Values from SQL query (what's in the database) | `["USA", "15"]` |
| `extracted_values_from_NL` | Values from natural language question | `["United States", "15"]` |
| **Exact Match** | When both arrays have identical values | Both contain `["2007"]` |
| **Fuzzy Match** | When values differ but refer to same entity | `["USA"]` vs `["United States"]` |

## Benchmark Comparison

### When to Use Each Benchmark

| Benchmark | Use Case | System Requirements | Evaluation Focus |
|-----------|----------|---------------------|------------------|
| **Column2Schema Mapping** | Test pure semantic understanding | Schema access only | Concept → Column mapping accuracy |
| **Exact Cell Value Matching** | Test precise entity recognition | Schema + Database access | Exact value → Column identification |
| **Fuzzy Cell Value Matching** | Test robust entity matching | Schema + Database + Fuzzy matching | Approximate value → Column identification |

### Difficulty Progression

1. **Easiest**: Column2Schema Mapping
   - No data access needed
   - Pure conceptual understanding
   - Example: "most popular movie" → `movie_popularity` column

2. **Medium**: Exact Cell Value Matching
   - Requires data access
   - Direct value lookup
   - Example: "released in 2007" → find column containing "2007"

3. **Hardest**: Fuzzy Cell Value Matching
   - Requires data access + similarity matching
   - Handle variations and normalizations
   - Example: "United States" → find column containing "USA"

### System Capabilities Matrix

| Capability | Column2Schema | Exact Match | Fuzzy Match |
|------------|---------------|-------------|-------------|
| Schema understanding | ✓ | ✓ | ✓ |
| Database access | ✗ | ✓ | ✓ |
| Exact value lookup | ✗ | ✓ | ✓ |
| Fuzzy string matching | ✗ | ✗ | ✓ |
| Normalization/synonyms | ✗ | ✗ | ✓ |
| Abbreviation handling | ✗ | ✗ | ✓ |

## Practical Examples

### Example 1: Pure Column2Schema Mapping
```
Question: "What is the average salary of employees in the engineering department?"

Schema Mapping Required:
- "average" → AVG() function
- "salary" → employees.salary column
- "employees" → employees table
- "engineering department" → department.name = 'Engineering'

No cell values needed - pure conceptual understanding.
```

### Example 2: Exact Cell Value Matching
```
Question: "How many movies were released in 2007?"

Schema Mapping Required:
- "movies" → movies table (concept)
- "released" → release_year column (concept)
- "2007" → WHERE release_year = 2007 (exact value match)

Database lookup: Find column containing exact value "2007"
```

### Example 3: Fuzzy Cell Value Matching
```
Question: "List all publishers in the United States"

Schema Mapping Required:
- "publishers" → publishers table (concept)
- "United States" → WHERE country = 'USA' (fuzzy match)

Database lookup: Find column containing "USA" when question says "United States"
Fuzzy matching needed: "United States" ≈ "USA"
```

### Example 4: Combined Approach (Real-world scenario)
```
Question: "What is the average rating of movies directed by Stanley Kubrick released after 1970?"

Breakdown:
1. Column2Schema: "average rating" → AVG(rating)
2. Fuzzy Match: "Stanley Kubrick" → director_name = 'S. Kubrick'
3. Exact Match: "1970" → release_year > 1970

This query requires all three capabilities!
```

## Usage

### Running the Analysis
```bash
python3 analyze_queries.py
```

This will:
1. Load `bird/train/train.json` (queries)
2. Load `bird/train/train_tables.json` (schema information)
3. Analyze each query
4. Generate the output files

### Testing Schema Mapping Systems

**For Column2Schema Mapping (Pure Concept):**
```python
# Load the benchmark
with open('pure_concept_mapping_queries.json') as f:
    queries = json.load(f)

# Test your system
for query in queries:
    predicted_columns = your_system.map_columns(
        question=query['question'],
        schema=get_schema(query['db_id'])
    )
    
    # Compare with ground truth
    expected_columns = query['gold_columns']  # All columns for concept_only
    accuracy = evaluate(predicted_columns, expected_columns)
```

**For Exact Cell Value Matching:**
```python
# Load the benchmark
with open('cell_value_matching_queries.json') as f:
    queries = json.load(f)

# Test your system
for query in queries:
    # Extract values from question
    extracted_values = query['extracted_values_from_NL']
    
    predicted_columns = your_system.map_columns_with_exact_values(
        question=query['question'],
        schema=get_schema(query['db_id']),
        database=get_database(query['db_id']),
        values=extracted_values
    )
    
    # Compare with gold standard (only value-dependent columns)
    expected_columns = query['gold_columns']
    accuracy = evaluate(predicted_columns, expected_columns)
```

**For Fuzzy Cell Value Matching:**
```python
# Load the benchmark
with open('fuzzy_cell_matching.json') as f:
    queries = json.load(f)

# Test your system
for query in queries:
    # Extract values from question (these differ from DB values)
    nl_values = query['extracted_values_from_NL']
    db_values = query['matched_values']
    
    predicted_columns = your_system.map_columns_with_fuzzy_values(
        question=query['question'],
        schema=get_schema(query['db_id']),
        database=get_database(query['db_id']),
        values=nl_values,
        similarity_threshold=0.8  # Adjust based on your needs
    )
    
    # Compare with gold standard
    expected_columns = query['gold_columns']
    accuracy = evaluate(predicted_columns, expected_columns)
    
    # Also evaluate fuzzy matching quality
    for nl_val, db_val in zip(nl_values, db_values):
        if nl_val != db_val:
            fuzzy_match_quality = evaluate_fuzzy_match(nl_val, db_val)
```

## Notes

- The `evidence` field often provides hints about column mappings
- Uses **sqlparse** library for robust SQL parsing instead of regex
- The 4,530 "unknown" queries typically have complex subqueries or expressions where columns couldn't be fully extracted
- Boolean values (0/1) representing states are treated as concepts, not values
- Numeric values in comparisons (e.g., "rating > 4") are considered value-dependent
- The improved parser captures 51.95% of queries (4,898 out of 9,428)

### Fuzzy Matching Characteristics
- **Common patterns**: Country names (USA/United States), abbreviations, name variations
- **Matching techniques needed**: String similarity (Levenshtein, Jaro-Winkler), normalization, synonym matching
- **Threshold considerations**: Balance between precision and recall based on your use case
- **Entity types**: Names, locations, dates, abbreviations, alternative spellings

## Files

- `analyze_queries.py` - Main analysis script (uses sqlparse for robust SQL parsing)
- `add_extracted_values.py` - Script to add extracted values from natural language
- `pure_concept_mapping_queries.json` - Column2Schema mapping benchmark (2,974 queries)
- `cell_value_matching_queries.json` - Exact cell value matching benchmark (1,317 queries)
- `fuzzy_cell_matching.json` - Fuzzy cell value matching benchmark (607 queries)
- `categorization_summary.json` - Statistics and samples
- `README_SCHEMA_MAPPING_BENCHMARK.md` - This documentation

## Citation

If you use this benchmark categorization, please cite the original BIRD benchmark:

```
@article{li2024bird,
  title={BIRD: A Big Bench for Large-scale Database Grounded Text-to-SQL Evaluation},
  author={Li, Jinyang and Hui, Binyuan and Qu, Ge and Yang, Jiaxi and Li, Binhua and Li, Bowen and Wang, Bailin and Qin, Bowen and Geng, Ruiying and Huo, Nan and others},
  journal={arXiv preprint arXiv:2305.03111},
  year={2024}
}