import json
import re
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison, Token
from sqlparse.tokens import Keyword, DML, Literal

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_schema_for_db(db_id, tables_data):
    """Get schema information for a specific database.

    Uses table_names_original/column_names_original -- the actual SQL
    identifiers -- rather than table_names/column_names, which are
    human-readable relabelings (e.g. 'publisher id' for the real column
    'pub_id') that a query's SQL text never contains.
    """
    for db in tables_data:
        if db['db_id'] == db_id:
            table_names = db['table_names_original']
            schema = {
                'tables': table_names,
                'columns': []
            }
            # Build column list with table associations
            for col_idx, (table_idx, col_name) in enumerate(db['column_names_original']):
                if table_idx >= 0:  # Skip the -1 index (*)
                    schema['columns'].append({
                        'column_name': col_name,
                        'table_name': table_names[table_idx],
                        'table_idx': table_idx,
                        'column_idx': col_idx
                    })
            return schema
    return None

def extract_identifiers(token):
    """Recursively extract identifiers from a token"""
    identifiers = []
    if isinstance(token, IdentifierList):
        for identifier in token.get_identifiers():
            identifiers.extend(extract_identifiers(identifier))
    elif isinstance(token, Identifier):
        identifiers.append(token.get_real_name())
    elif hasattr(token, 'tokens'):
        for t in token.tokens:
            identifiers.extend(extract_identifiers(t))
    return identifiers

_FROM_CLAUSE_TERMINATOR_RE = re.compile(r'^(WHERE|GROUP|ORDER|HAVING|LIMIT)\b', re.IGNORECASE)
_JOIN_KEYWORD_RE = re.compile(r'\bJOIN\b', re.IGNORECASE)
_JOIN_SYNTAX_KEYWORDS = {'ON', 'USING', 'AS'}

def _ends_from_clause(token):
    # sqlparse groups a WHERE clause into its own Where-class token; GROUP BY/
    # ORDER BY/HAVING/LIMIT stay as plain Keyword tokens with a compound value
    # (e.g. 'ORDER BY'), so a bare '== "ORDER"' check never matches them.
    if isinstance(token, Where):
        return True
    return token.ttype is Keyword and bool(_FROM_CLAUSE_TERMINATOR_RE.match(token.value.strip()))

def _starts_from_clause(token):
    if token.ttype is not Keyword:
        return False
    value = token.value.strip().upper()
    return value == 'FROM' or bool(_JOIN_KEYWORD_RE.search(value))

def _is_join_syntax_keyword(token):
    if token.ttype is not Keyword:
        return False
    value = token.value.strip().upper()
    return value in _JOIN_SYNTAX_KEYWORDS or bool(_JOIN_KEYWORD_RE.search(value))

def _table_ref_from_identifier_list_item(item):
    # get_identifiers() elements are normally Identifier tokens, but a table
    # name that collides with sqlparse's keyword list (e.g. a table literally
    # named 'lists') comes back as a bare Token instead of being grouped.
    if isinstance(item, Identifier):
        return item.get_real_name(), item.get_alias()
    return item.value.strip(), None

def extract_table_aliases(parsed):
    """Map every table alias (and the real table name itself) appearing in
    FROM/JOIN clauses to the table's real name, e.g. {'c': 'customers',
    'customers': 'customers'}. Used to resolve qualified column references
    (alias.column) back to the actual table, and to restrict unqualified
    column matching to tables the query actually uses.

    A handful of BIRD table names (e.g. 'lists', 'user', 'transaction')
    collide with sqlparse's keyword list and come back as a bare Keyword
    token instead of being grouped into an Identifier; those are recovered
    explicitly below, including the common 'FROM tablename AS alias' and
    'FROM tablename alias' forms. More convoluted combinations (e.g. a
    comma-joined list mixing such a table with an explicit alias) aren't
    specially handled; since this feeds a restriction rather than an
    inclusion, missing a table here means a query is conservatively
    excluded from the benchmark rather than mislabeled.
    """
    aliases = {}
    from_seen = False
    tokens = [t for t in parsed.tokens if not t.is_whitespace]

    def register(name, alias):
        if not name:
            return
        aliases[name.lower()] = name
        if alias:
            aliases[alias.lower()] = name

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if from_seen:
            if _ends_from_clause(token):
                from_seen = False
            elif isinstance(token, IdentifierList):
                for item in token.get_identifiers():
                    register(*_table_ref_from_identifier_list_item(item))
            elif isinstance(token, Identifier):
                register(token.get_real_name(), token.get_alias())
            elif token.ttype is Keyword and not _is_join_syntax_keyword(token):
                name = token.value.strip()
                alias = None
                nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                if nxt is not None and nxt.ttype is Keyword and nxt.value.strip().upper() == 'AS':
                    nxt2 = tokens[i + 2] if i + 2 < len(tokens) else None
                    if isinstance(nxt2, Identifier) and '.' not in str(nxt2):
                        alias = nxt2.get_real_name()
                        i += 2
                elif isinstance(nxt, Identifier) and '.' not in str(nxt):
                    alias = nxt.get_real_name()
                    i += 1
                register(name, alias)

        if _starts_from_clause(token):
            from_seen = True

        i += 1

    return aliases

def extract_tables_from_parsed(parsed):
    """Extract table names from parsed SQL"""
    return set(name.lower() for name in extract_table_aliases(parsed).values())

def extract_columns_from_parsed(parsed, schema, used_tables, table_aliases):
    """Extract columns with their context from parsed SQL.

    Unqualified column names (no 'table.' prefix) are ambiguous whenever more
    than one table in the schema has a column with that name. We resolve
    qualified references (alias.column) to their real table via
    table_aliases, and restrict unqualified references to tables actually
    referenced in this query's FROM/JOIN clauses (used_tables) rather than
    matching the first same-named column anywhere in the whole database
    schema. This still doesn't disambiguate two *used* tables sharing a
    column name, but that's a real query ambiguity rather than a resolution
    bug.
    """
    columns = []
    column_names_seen = set()

    def process_token(token):
        if isinstance(token, Identifier):
            name = token.get_real_name()
            if name and '.' in str(token):
                # Handle table.column / alias.column format
                parts = str(token).split('.')
                if len(parts) == 2:
                    table_qualifier = parts[0].strip().lower()
                    col_name = parts[1].strip().lower()
                    real_table = table_aliases.get(table_qualifier)
                    match_column(col_name, restrict_table=real_table)
            elif name:
                match_column(name.lower())
        elif isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                process_token(identifier)
        elif hasattr(token, 'tokens'):
            for t in token.tokens:
                if not t.is_whitespace:
                    process_token(t)

    def match_column(col_name, restrict_table=None):
        # Candidate schema columns with this name, restricted to a specific
        # resolved table (when the reference was qualified) or otherwise to
        # tables actually used by this query.
        candidates = [c for c in schema['columns'] if c['column_name'].lower() == col_name]
        if restrict_table is not None:
            candidates = [c for c in candidates if c['table_name'].lower() == restrict_table.lower()]
        else:
            candidates = [c for c in candidates if c['table_name'].lower() in used_tables]

        if not candidates:
            return
        col_info = candidates[0]
        key = (col_info['table_name'], col_name)
        if key not in column_names_seen:
            column_names_seen.add(key)
            columns.append({
                'column_name': col_info['column_name'],
                'table_name': col_info['table_name'],
                'column_name_lower': col_name
            })

    process_token(parsed)
    return columns

def extract_literal_values(parsed):
    """Extract literal values from WHERE and HAVING clauses"""
    literals = []
    in_where_having = False
    
    def process_token(token):
        nonlocal in_where_having
        
        if token.ttype is Keyword:
            if token.value.upper() in ('WHERE', 'HAVING'):
                in_where_having = True
            elif token.value.upper() in ('GROUP', 'ORDER', 'LIMIT', 'FROM', 'SELECT'):
                in_where_having = False
        
        if in_where_having:
            if token.ttype in (Literal.String.Single, Literal.String.Symbol):
                # String literal
                value = token.value.strip("'\"")
                literals.append(('string', value))
            elif token.ttype in (Literal.Number.Integer, Literal.Number.Float):
                # Numeric literal
                literals.append(('number', token.value))
        
        if hasattr(token, 'tokens'):
            for t in token.tokens:
                process_token(t)
    
    process_token(parsed)
    return literals

def check_value_in_question(value, question, evidence):
    """Check if a literal value appears in the question or evidence"""
    combined_text = (question + " " + (evidence or "")).lower()
    value_str = str(value).lower()
    return value_str in combined_text

def find_column_for_value(parsed, value, value_type, columns):
    """Find which column a literal value is associated with"""
    value_str = str(value)
    
    def search_comparison(token):
        if isinstance(token, Comparison):
            # Get the left side (column) and right side (value)
            left = None
            right = None
            for t in token.tokens:
                if isinstance(t, Identifier):
                    left = t.get_real_name().lower()
                elif t.ttype in (Literal.String.Single, Literal.String.Symbol, 
                                Literal.Number.Integer, Literal.Number.Float):
                    right = t.value.strip("'\"")
            
            if right == value_str and left:
                # Match with schema columns
                for col in columns:
                    if col['column_name_lower'] == left or left.endswith('.' + col['column_name_lower']):
                        return col
        
        if hasattr(token, 'tokens'):
            for t in token.tokens:
                result = search_comparison(t)
                if result:
                    return result
        return None
    
    return search_comparison(parsed)

def categorize_query(entry, schema):
    """
    Categorize query columns into:
    - concept_only_columns: Columns that can be mapped purely by concept
    - value_dependent_columns: Columns that are identified via cell values
    """
    question = entry.get('question', '')
    evidence = entry.get('evidence', '')
    sql = entry.get('SQL', '')
    
    # Parse SQL
    try:
        parsed = sqlparse.parse(sql)[0]
    except Exception as e:
        print(f"Error parsing SQL for {entry.get('db_id')}: {e}")
        return {
            'category': 'unknown',
            'used_tables': [],
            'all_columns': [],
            'concept_only_columns': [],
            'value_dependent_columns': [],
            'matched_values': []
        }
    
    # Extract tables and columns
    used_tables = extract_tables_from_parsed(parsed)
    table_aliases = extract_table_aliases(parsed)
    used_columns = extract_columns_from_parsed(parsed, schema, used_tables, table_aliases)
    
    # Extract literal values
    sql_literals = extract_literal_values(parsed)
    
    # Find which values appear in the question and which columns they map to.
    # matched_values and value_dependent_columns are built from the same filtered list of
    # (value, col_info) pairs so they always stay the same length and correctly paired by
    # index downstream (e.g. collect_matched_values_for_database). A literal is dropped
    # entirely if find_column_for_value can't resolve its column (e.g. it's wrapped in a
    # SQL function like strftime(...)) or if it maps to a column already covered by an
    # earlier value in this query.
    value_dependent_columns = []
    matched_values = []

    for value_type, literal in sql_literals:
        if check_value_in_question(literal, question, evidence):
            col_info = find_column_for_value(parsed, literal, value_type, used_columns)
            if col_info and col_info not in value_dependent_columns:
                matched_values.append(literal)
                value_dependent_columns.append(col_info)
    
    # Columns that are NOT value-dependent are concept-only
    concept_only_columns = [col for col in used_columns if col not in value_dependent_columns]
    
    # Determine primary category
    has_value_cols = len(value_dependent_columns) > 0
    has_concept_cols = len(concept_only_columns) > 0
    
    if has_value_cols and not has_concept_cols:
        category = 'value_only'
    elif not has_value_cols and has_concept_cols:
        category = 'concept_only'
    elif has_value_cols and has_concept_cols:
        category = 'mixed'
    else:
        category = 'unknown'
    
    # For benchmarking: only include the relevant gold standard columns
    if category == 'concept_only':
        gold_columns = [{'table': c['table_name'], 'column': c['column_name']} for c in used_columns]
    elif category in ['value_only', 'mixed']:
        gold_columns = [{'table': c['table_name'], 'column': c['column_name']} for c in value_dependent_columns]
    else:
        gold_columns = []
    
    return {
        'category': category,
        'used_tables': sorted(list(used_tables)),
        'gold_columns': gold_columns,
        'matched_values': matched_values
    }

def main():
    print("Loading train.json and train_tables.json...")
    train_data = load_json('bird/train/train.json')
    tables_data = load_json('bird/train/train_tables.json')
    
    print(f"Total queries: {len(train_data)}")
    print(f"Total databases: {len(tables_data)}")
    
    # Categorize all queries
    concept_only_queries = []
    value_dependent_queries = []
    mixed_queries = []
    unknown_queries = []
    
    for idx, entry in enumerate(train_data):
        if idx % 1000 == 0:
            print(f"Processing query {idx}/{len(train_data)}...")
        
        db_id = entry.get('db_id')
        schema = get_schema_for_db(db_id, tables_data)
        
        if not schema:
            print(f"Warning: No schema found for db_id: {db_id}")
            continue
        
        analysis = categorize_query(entry, schema)
        
        # Add analysis info to entry
        entry_with_analysis = entry.copy()
        entry_with_analysis.update(analysis)
        
        if analysis['category'] == 'concept_only':
            concept_only_queries.append(entry_with_analysis)
        elif analysis['category'] == 'value_only':
            value_dependent_queries.append(entry_with_analysis)
        elif analysis['category'] == 'mixed':
            mixed_queries.append(entry_with_analysis)
        else:
            unknown_queries.append(entry_with_analysis)
    
    print(f"\n=== Detailed Breakdown ===")
    print(f"Concept-Only queries (all columns mapped by concept): {len(concept_only_queries)}")
    print(f"Value-Only queries (all columns found via cell values): {len(value_dependent_queries)}")
    print(f"Mixed queries (both types of columns): {len(mixed_queries)}")
    print(f"Unknown queries: {len(unknown_queries)}")
    
    # For the user's request, we want two categories:
    # 1. Queries where columns are linked to concepts (NO cell values needed)
    # 2. Queries where columns can be found by cell values mentioned
    
    # Category 1: Pure concept mapping
    pure_concept_queries = concept_only_queries
    
    # Category 2: Any query that uses cell values to identify columns
    cell_value_queries = value_dependent_queries + mixed_queries
    
    print(f"\n=== Final Categorization for Schema Mapping Benchmark ===")
    print(f"Pure Concept Mapping (NO cell values needed): {len(pure_concept_queries)}")
    print(f"Cell Value Matching (cell values help identify columns): {len(cell_value_queries)}")
    
    # Save categorized queries
    print("\nSaving pure_concept_mapping_queries.json...")
    with open('pure_concept_mapping_queries.json', 'w', encoding='utf-8') as f:
        json.dump(pure_concept_queries, f, indent=2, ensure_ascii=False)
    
    print("Saving cell_value_matching_queries.json...")
    with open('cell_value_matching_queries.json', 'w', encoding='utf-8') as f:
        json.dump(cell_value_queries, f, indent=2, ensure_ascii=False)
    
    # Create detailed summary report
    print("\nCreating detailed summary report...")
    
    # Sample examples from each category
    concept_samples = pure_concept_queries[:10] if len(pure_concept_queries) >= 10 else pure_concept_queries
    value_samples = cell_value_queries[:10] if len(cell_value_queries) >= 10 else cell_value_queries
    
    summary = {
        "total_queries": len(train_data),
        "pure_concept_mapping_count": len(pure_concept_queries),
        "cell_value_matching_count": len(cell_value_queries),
        "breakdown": {
            "concept_only": len(concept_only_queries),
            "value_only": len(value_dependent_queries),
            "mixed": len(mixed_queries),
            "unknown": len(unknown_queries)
        },
        "percentages": {
            "pure_concept_mapping": round(len(pure_concept_queries) / len(train_data) * 100, 2),
            "cell_value_matching": round(len(cell_value_queries) / len(train_data) * 100, 2)
        },
        "pure_concept_samples": concept_samples,
        "cell_value_samples": value_samples
    }
    
    with open('categorization_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\nDone! Files created:")
    print("  - pure_concept_mapping_queries.json (for testing concept-based schema mapping)")
    print("  - cell_value_matching_queries.json (for testing value-based column identification)")
    print("  - categorization_summary.json (statistics and samples)")
    print("\nEach query now includes:")
    print("  - used_tables: List of tables needed")
    print("  - all_columns: All columns with their tables")
    print("  - concept_only_columns: Columns mapped by concept")
    print("  - value_dependent_columns: Columns identified via cell values")

if __name__ == "__main__":
    main()

# Made with Bob
