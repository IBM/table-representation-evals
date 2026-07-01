"""
Create a semantic fuzzy benchmark from BIRD training data.

Mines the evidence field for genuine NL->DB value mismatches where the NL phrase
appears in the question but the DB value does not appear verbatim. Applies quality
filtering to keep only semantically meaningful cases (not typos, not trivial
case/punctuation differences).

Outputs semantic_fuzzy_matching.json in the same format as fuzzy_cell_matching.json,
plus a fuzzy_type field for transparency.

Run from repo root:
    python benchmark_src/dataset_creation/bird/create_semantic_fuzzy_benchmark.py
"""

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import sqlparse
from sqlparse.sql import Comparison, Identifier
from sqlparse.tokens import Literal

BIRD_PATH = Path("cache/datasets/bird")
DB_PATH = BIRD_PATH / "train" / "train_databases"
FUZZY_SYNTHETIC_FILE = Path("benchmark_src/dataset_creation/bird/fuzzy_cell_matching.json")
OUTPUT_FILE = Path("benchmark_src/dataset_creation/bird/semantic_fuzzy_matching.json")


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

def load_schema(tables_data):
    """db_id -> {col_name_lower -> [{table, column}]}"""
    lookup = {}
    for db in tables_data:
        db_id = db["db_id"]
        lookup[db_id] = defaultdict(list)
        for table_idx, col_name in db["column_names"]:
            if table_idx < 0:
                continue
            table_name = db["table_names"][table_idx]
            lookup[db_id][col_name.lower()].append(
                {"table": table_name, "column": col_name}
            )
    return lookup


# ---------------------------------------------------------------------------
# Evidence parsing
# ---------------------------------------------------------------------------

# Matches: "NL phrase refers to col_ref = 'db_val'"
# col_ref may contain dots (table.col), spaces, or underscores
_EVIDENCE_RE = re.compile(
    r"(.+?)\s+refers?\s+to\s+([\w\s.]+?)\s*=\s*['\"](.+?)['\"]",
    re.IGNORECASE,
)


def parse_evidence_segments(evidence):
    """
    Return list of (nl_phrase, col_ref, db_val) tuples from evidence text.
    Only matches segments that have a quoted string value on the right-hand side.
    """
    results = []
    for seg in re.split(r";|\n", evidence):
        seg = seg.strip()
        if "refers" not in seg.lower():
            continue
        m = _EVIDENCE_RE.match(seg)
        if m:
            nl_phrase = m.group(1).strip().strip("\"'")
            col_ref = m.group(2).strip()
            db_val = m.group(3).strip()
            results.append((nl_phrase, col_ref, db_val))
    return results


# ---------------------------------------------------------------------------
# SQL parsing — find which column is filtered by a given literal value
# ---------------------------------------------------------------------------

def _walk_comparisons(token, db_val, schema_lookup, db_id):
    hits = []
    if isinstance(token, Comparison):
        left = None
        right = None
        for t in token.tokens:
            if isinstance(t, Identifier):
                left = t.get_real_name()
            elif t.ttype in (Literal.String.Single, Literal.String.Symbol):
                right = t.value.strip("'\"")
        if right == db_val and left:
            left_lower = left.lower()
            for col_info in schema_lookup.get(db_id, {}).get(left_lower, []):
                hits.append({"table": col_info["table"], "column": col_info["column"]})
    if hasattr(token, "tokens"):
        for t in token.tokens:
            hits.extend(_walk_comparisons(t, db_val, schema_lookup, db_id))
    return hits


def find_gold_column(sql, db_val, schema_lookup, db_id):
    """Return {table, column} for the first comparison matching db_val, or None."""
    try:
        parsed = sqlparse.parse(sql)[0]
    except Exception:
        return None
    hits = _walk_comparisons(parsed, db_val, schema_lookup, db_id)
    return hits[0] if hits else None


def extract_used_tables(sql):
    """Return sorted list of table names referenced in the SQL."""
    tables = set()
    try:
        parsed = sqlparse.parse(sql)[0]
    except Exception:
        return []
    from_seen = False
    for token in parsed.flatten():
        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() in ("FROM", "JOIN"):
            from_seen = True
        elif from_seen and token.ttype is sqlparse.tokens.Name:
            tables.add(token.value.lower())
            from_seen = False
        elif token.ttype is sqlparse.tokens.Keyword:
            from_seen = False
    return sorted(tables)


# ---------------------------------------------------------------------------
# DB verification
# ---------------------------------------------------------------------------

def value_exists_in_db(db_id, table, column, value):
    db_file = DB_PATH / db_id / f"{db_id}.sqlite"
    if not db_file.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute(f'SELECT 1 FROM "{table}" WHERE "{column}" = ? LIMIT 1', (value,))
        found = cursor.fetchone() is not None
        conn.close()
        return found
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------

_DATE_NL_RE = re.compile(
    r"\d{4}[/.-]\d|\d{1,2}/\d{1,2}/\d{4}"
    r"|(january|february|march|april|may|june|july|august"
    r"|september|october|november|december)",
    re.IGNORECASE,
)
_DATE_DB_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def categorize(nl, db):
    if _DATE_NL_RE.search(nl) and _DATE_DB_RE.search(db):
        return "date_format"
    db_s = db.strip()
    # Short uppercase codes (TN, PG, cs) for longer NL phrases
    if (
        len(db_s) <= 4
        and (db_s.isupper() or db_s.islower())
        and db_s.isalpha()
        and len(nl) > len(db_s) * 2
    ):
        return "abbreviation"
    return "synonym"


# ---------------------------------------------------------------------------
# Quality judgement
# ---------------------------------------------------------------------------

_VAGUE = {
    "directed by", "main actors", "average", "maximum", "minimum", "total",
    "first", "last", "oldest", "newest", "highest", "lowest", "most", "least",
    "before", "after", "since", "until", "between", "within", "animators",
    "roof open", "open roof",
}
_BOOLEAN_DB = {"true", "false", "yes", "no", "none", "null", "hang", "active", "inactive"}
_QUESTION_START = re.compile(r"^(how|what|when|where|who|which|why|is|are|was|were)\b", re.I)

# Arbitrary internal codes that have no semantic relation to the NL phrase
_ARBITRARY_CODES = {"pos", "neg", "0000-00-00", "med", "free"}

# Geographic container terms — mapping these to specific countries is a bad annotation
_GEO_CONTAINERS = {
    "asia", "europe", "africa", "north america", "south america",
    "latin america", "middle east", "oceania", "central america",
}

# Special-case exclusions that slip through general filters but are semantically invalid
_EXACT_EXCLUSIONS = {
    ("sherry beef", "sherried beef"),  # morphological recipe-title variant
    ("hilla", "hilaa"),                # insurance code scramble (DB error)
    ("zimbabwean", "zimbabwea"),       # truncated/corrupt DB value
}

def quality_judgement(nl, db, category):
    """
    Returns (keep: bool, reason: str).

    Keeps cases that represent genuine semantic gaps a cell embedding model
    needs to bridge. Discards trivial, ambiguous, or untestable cases.
    """
    nl_l = nl.lower().strip()
    db_l = db.lower().strip()

    if len(nl_l) < 4:
        return False, "nl_too_short"
    if len(db_l) < 2:
        return False, "db_too_short"

    # SQL wildcard patterns, not actual values
    if "%" in db:
        return False, "sql_pattern"

    # Boolean / arbitrary coded values
    if db_l in _BOOLEAN_DB:
        return False, "boolean_or_coded"

    # Arbitrary internal pos/neg/date-sentinel codes
    if db_l in _ARBITRARY_CODES:
        return False, "arbitrary_internal_code"

    # Pure numbers
    try:
        float(db.replace(",", ""))
        return False, "numeric"
    except ValueError:
        pass

    # Long strings (> 80 chars) are likely verbatim comment fragments, not cell values
    # worth testing semantic embedding on
    if len(db) > 80:
        return False, "db_too_long"

    # Formatting-only differences: long strings with very high character overlap
    # (e.g. address with commas vs without, or minor punctuation in long text)
    if len(nl_l) > 30 and len(db_l) > 30:
        overlap = sum(1 for c in nl_l if c in db_l)
        ratio = overlap / max(len(nl_l), len(db_l))
        if ratio > 0.85:
            return False, "formatting_only_long_string"

    # Identical after lowercasing
    if nl_l == db_l:
        return False, "identical_case_insensitive"

    # Differs only by punctuation / whitespace (African-American vs African American)
    if nl_l.replace("-", " ").replace("_", " ") == db_l.replace("-", " ").replace("_", " "):
        return False, "punctuation_only"

    # Differs only by trailing s / es (pluralisation)
    if nl_l.rstrip("s") == db_l.rstrip("s") and abs(len(nl_l) - len(db_l)) <= 2:
        return False, "pluralisation_only"

    # Vague operation phrases — these map concepts, not cell values
    if nl_l in _VAGUE:
        return False, "vague_phrase"

    # Question fragments accidentally extracted
    if _QUESTION_START.match(nl_l):
        return False, "question_fragment"

    # Geographic container → specific country is a wrong annotation (Asia ≠ Japan)
    nl_core = re.sub(r"^(from|in|the|a|an)\s+", "", nl_l).strip()
    if nl_core in _GEO_CONTAINERS:
        return False, "geo_container_to_specific_country"

    # Verbatim DB text appearing inside NL (quote with attribution prefix, e.g.
    # 'saying, "Yuk, more like licorice soda' → 'Yuk, more like licorice soda.')
    db_stripped = db_l.rstrip(".,!?; ")
    if len(db_stripped) > 15 and db_stripped in nl_l:
        return False, "verbatim_db_in_nl"

    # Wrong entity mapping: NL and DB share a long common prefix AND both diverge
    # afterward (e.g. two different movie titles sharing "Pirates of the Caribbean: ").
    # If only the DB has a suffix (NL is just a prefix of DB), it's a valid synonym.
    if len(nl_l) > 20 and len(db_l) > 20:
        prefix_len = 0
        for a, b in zip(nl_l, db_l):
            if a == b:
                prefix_len += 1
            else:
                break
        if prefix_len > 20:
            nl_suffix = nl_l[prefix_len:].strip()
            db_suffix = db_l[prefix_len:].strip()
            if nl_suffix and db_suffix:
                return False, "wrong_entity_same_prefix"

    # Special-case exact exclusions for patterns too subtle for general rules
    if (nl_l.strip(), db_l.strip()) in _EXACT_EXCLUSIONS:
        return False, "exact_exclusion"

    # Stray trailing punctuation in NL (e.g. 'Narrator"') — extraction artifact
    if nl_l and not nl_l[-1].isalpha() and not nl_l[-1].isdigit():
        return False, "extraction_artifact"

    # For date formats: always good if the regex matched
    if category == "date_format":
        return True, "date_format"

    # For abbreviations: always good
    if category == "abbreviation":
        return True, "abbreviation"

    # For synonyms: require the NL phrase to be at least 4 chars and clearly different
    if category == "synonym":
        if " " not in nl_l and len(nl_l) < 6 and abs(len(nl_l) - len(db_l)) <= 2:
            return False, "minor_single_word_variant"
        return True, "synonym"

    return True, "other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if len(a) > len(b):
        a, b = b, a
    row = list(range(len(a) + 1))
    for c2 in b:
        new_row = [row[0] + 1]
        for j, c1 in enumerate(a):
            new_row.append(min(new_row[-1] + 1, row[j + 1] + 1, row[j] + (c1 != c2)))
        row = new_row
    return row[-1]


TYPO_CAP = 45
TYPO_CAP_PER_DB = 3


def mine_typo_pairs(existing_pair_counts: Counter) -> list:
    """
    Load fuzzy_cell_matching.json and extract typo pairs: NL is a plausible
    misspelling of the DB value (high character overlap, different spelling,
    NL appears in question, DB value verified in DB).
    """
    if not FUZZY_SYNTHETIC_FILE.exists():
        print("  fuzzy_cell_matching.json not found, skipping typo mining")
        return []

    with open(FUZZY_SYNTHETIC_FILE) as f:
        source = json.load(f)

    typo_counts = Counter()
    pair_counts = Counter(existing_pair_counts)
    db_counts: Counter = Counter()
    results = []

    for entry in source:
        nl_vals = entry.get("extracted_values_from_NL", [])
        db_vals = entry.get("matched_values", [])
        gold_cols = entry.get("gold_columns", [])
        q = entry.get("question", "")
        db_id = entry.get("db_id", "")

        if not nl_vals or not db_vals or not gold_cols:
            continue

        nl = nl_vals[0]
        db = db_vals[0]
        nl_l = nl.lower().strip()
        db_l = db.lower().strip()

        # Basic quality gates (same as semantic pipeline)
        if len(nl_l) < 3 or len(db_l) < 2:
            typo_counts["too_short"] += 1
            continue
        if nl_l == db_l:
            typo_counts["identical"] += 1
            continue
        if db_l in _BOOLEAN_DB or db_l in _ARBITRARY_CODES:
            typo_counts["boolean_or_coded"] += 1
            continue
        try:
            float(db.replace(",", ""))
            typo_counts["numeric"] += 1
            continue
        except ValueError:
            pass
        if len(db) > 80:
            typo_counts["db_too_long"] += 1
            continue
        # NL must appear in question (typo as typed by user)
        if nl_l not in q.lower():
            typo_counts["nl_not_in_question"] += 1
            continue
        # DB value must not appear verbatim (otherwise it's an exact match)
        if db_l in q.lower():
            typo_counts["db_exact_in_question"] += 1
            continue
        # Plausibility: character overlap ratio > 0.5 — rules out completely
        # unrelated pairs from the alignment noise in the source file
        overlap = sum(1 for c in nl_l if c in db_l)
        ratio = overlap / max(len(nl_l), len(db_l))
        if ratio < 0.5:
            typo_counts["low_overlap"] += 1
            continue
        # Verify DB value exists in the gold column
        gold_col = gold_cols[0]
        if not value_exists_in_db(db_id, gold_col["table"], gold_col["column"], db):
            typo_counts["value_not_in_db"] += 1
            continue

        if len(results) >= TYPO_CAP:
            typo_counts["total_cap"] += 1
            continue
        if db_counts[db_id] >= TYPO_CAP_PER_DB:
            typo_counts["per_db_cap"] += 1
            continue

        pair = (nl_l, db_l)
        if pair_counts[pair] >= 2:
            typo_counts["pair_cap"] += 1
            continue
        pair_counts[pair] += 1
        db_counts[db_id] += 1

        results.append({
            "db_id": db_id,
            "question": q,
            "evidence": entry.get("evidence", ""),
            "SQL": entry.get("SQL", ""),
            "fuzzy_type": "typo",
            "used_tables": entry.get("used_tables", []),
            "gold_columns": [gold_col],
            "matched_values": [db],
            "extracted_values_from_NL": [nl],
        })

    print(f"\nTypo mining: {len(results)} added  (rejections: {dict(typo_counts.most_common())})")
    return results


def main():
    print("Loading BIRD data...")
    with open(BIRD_PATH / "train" / "train.json") as f:
        train = json.load(f)
    with open(BIRD_PATH / "train" / "train_tables.json") as f:
        tables_data = json.load(f)

    schema_lookup = load_schema(tables_data)

    candidates = []
    rejection_counts = Counter()

    for entry in train:
        ev = entry.get("evidence", "")
        q = entry.get("question", "")
        sql = entry.get("SQL", "")
        db_id = entry.get("db_id", "")

        if not ev or not q or not sql:
            continue

        for nl_phrase, col_ref, db_val in parse_evidence_segments(ev):
            # NL phrase must appear in question
            if nl_phrase.lower() not in q.lower():
                rejection_counts["nl_not_in_question"] += 1
                continue
            # DB value must NOT appear verbatim in question (that would be exact match)
            if db_val.lower() in q.lower():
                rejection_counts["db_val_exact_in_question"] += 1
                continue

            cat = categorize(nl_phrase, db_val)
            keep, reason = quality_judgement(nl_phrase, db_val, cat)
            if not keep:
                rejection_counts[reason] += 1
                continue

            gold_col = find_gold_column(sql, db_val, schema_lookup, db_id)
            if gold_col is None:
                rejection_counts["gold_col_not_found"] += 1
                continue

            if not value_exists_in_db(db_id, gold_col["table"], gold_col["column"], db_val):
                rejection_counts["value_not_in_db"] += 1
                continue

            candidates.append({
                "db_id": db_id,
                "question": q,
                "evidence": ev,
                "SQL": sql,
                "fuzzy_type": cat,
                "used_tables": extract_used_tables(sql),
                "gold_columns": [gold_col],
                "matched_values": [db_val],
                "extracted_values_from_NL": [nl_phrase],
            })

    # --- Deduplication and capping ---

    # 1. Deduplicate on (question, nl_phrase): same query with multiple evidence segments
    seen_q = set()
    deduped = []
    for c in candidates:
        key = (c["db_id"], c["question"], c["extracted_values_from_NL"][0])
        if key not in seen_q:
            seen_q.add(key)
            deduped.append(c)

    # 2. Cap at 2 per unique (nl_phrase_lower, db_val_lower) pair — kills repetitive
    #    entries like 30x "Arizona" → "AZ" or 8x "American Airlines Inc." → "..."
    pair_counts: Counter = Counter()
    capped = []
    for c in deduped:
        pair = (c["extracted_values_from_NL"][0].lower(), c["matched_values"][0].lower())
        if pair_counts[pair] < 2:
            pair_counts[pair] += 1
            capped.append(c)

    # 3. Cap date_format at 3 per column — the pattern is the same regardless of date
    col_date_counts: Counter = Counter()
    final = []
    for c in capped:
        if c["fuzzy_type"] == "date_format":
            col_key = (c["gold_columns"][0]["table"], c["gold_columns"][0]["column"])
            if col_date_counts[col_key] >= 3:
                continue
            col_date_counts[col_key] += 1
        final.append(c)

    # --- Typo mining pass from fuzzy_cell_matching.json ---
    typo_entries = mine_typo_pairs(pair_counts)
    final.extend(typo_entries)

    unique = sorted(final, key=lambda x: (x["fuzzy_type"], x["db_id"]))

    # --- Summary ---
    print(f"\nTotal candidates after filtering: {len(unique)}")
    type_counts = Counter(c["fuzzy_type"] for c in unique)
    for t, n in type_counts.most_common():
        print(f"  {t}: {n}")

    print("\nRejection reasons:")
    for reason, n in rejection_counts.most_common():
        print(f"  {reason}: {n}")

    print("\n--- Sample output (first 5 per type) ---")
    by_type = defaultdict(list)
    for c in unique:
        by_type[c["fuzzy_type"]].append(c)

    for ftype, items in sorted(by_type.items()):
        print(f"\n[{ftype}]")
        for item in items[:5]:
            print(f"  NL: {repr(item['extracted_values_from_NL'][0]):50s}  DB: {repr(item['matched_values'][0])}")
            print(f"  Q:  {item['question'][:90]}")
            print(f"  Col: {item['gold_columns'][0]['table']}.{item['gold_columns'][0]['column']}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(unique)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
