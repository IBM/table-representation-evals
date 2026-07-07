"""
Create a semantic fuzzy benchmark from BIRD training data.

Mines the evidence field for genuine NL->DB value mismatches where the NL phrase
appears in the question but the DB value does not appear verbatim. Applies quality
filtering to keep only semantically meaningful cases (not typos, not trivial
case/punctuation differences).

Outputs semantic_fuzzy_matching.json with a fuzzy_type field indicating the type
of semantic gap (synonym, abbreviation, date_format, typo).

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

# Curated typo entries: NL phrase is a misspelling of the DB value.
# Mined from BIRD training data using character overlap filtering and DB verification.
_TYPO_ENTRIES = [
    {"db_id": "address", "question": "What is the area code of Bsihopville, SC?", "evidence": "\"Bishopville\" is the city; 'SC' is the state", "SQL": "SELECT T1.area_code FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.city = 'Bishopville' AND T2.state = 'SC'", "fuzzy_type": "typo", "used_tables": ["area_code", "zip_data"], "gold_columns": [{"table": "zip_data", "column": "city"}], "matched_values": ["Bishopville"], "extracted_values_from_NL": ["Bsihopville"]},
    {"db_id": "address", "question": "List the bad alias of the postal point located in Cmauy.", "evidence": "postal points refer to zip_code; Camuy is the city;", "SQL": "SELECT T1.bad_alias FROM avoid AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.city = 'Camuy'", "fuzzy_type": "typo", "used_tables": ["avoid", "zip_data"], "gold_columns": [{"table": "zip_data", "column": "city"}], "matched_values": ["Camuy"], "extracted_values_from_NL": ["Cmauy"]},
    {"db_id": "address", "question": "Provide the names of bad aliases in the city of Augadilla.", "evidence": "", "SQL": "SELECT T1.bad_alias FROM avoid AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.city = 'Aguadilla'", "fuzzy_type": "typo", "used_tables": ["avoid", "zip_data"], "gold_columns": [{"table": "zip_data", "column": "city"}], "matched_values": ["Aguadilla"], "extracted_values_from_NL": ["Augadilla"]},
    {"db_id": "beer_factory", "question": "How many breweries are there in Asutralia?", "evidence": "Australia refers to Country = 'Australia';", "SQL": "SELECT COUNT(BreweryName) FROM rootbeerbrand WHERE Country = 'Australia'", "fuzzy_type": "typo", "used_tables": ["rootbeerbrand"], "gold_columns": [{"table": "rootbeerbrand", "column": "Country"}], "matched_values": ["Australia"], "extracted_values_from_NL": ["Asutralia"]},
    {"db_id": "beer_factory", "question": "How many Flosom customers prefer to pay with Vsia?", "evidence": "Folsom refers to City = 'Folsom'; Visa refers to CreditCardType = 'Visa';", "SQL": "SELECT COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.City = 'Folsom' AND T2.CreditCardType = 'Visa'", "fuzzy_type": "typo", "used_tables": ["customers", "transaction"], "gold_columns": [{"table": "customers", "column": "City"}], "matched_values": ["Folsom"], "extracted_values_from_NL": ["Flosom"]},
    {"db_id": "books", "question": "Name the streets in Dlalas.", "evidence": "\"Dallas\" is the city; streets refers to street_name", "SQL": "SELECT street_name FROM address WHERE city = 'Dallas'", "fuzzy_type": "typo", "used_tables": ["address"], "gold_columns": [{"table": "address", "column": "city"}], "matched_values": ["Dallas"], "extracted_values_from_NL": ["Dlalas"]},
    {"db_id": "books", "question": "How many customers have an address that is located in the city of Vlileneuve-la-Garenne?", "evidence": "\"Villeneuve-la-Garenne\" is the city", "SQL": "SELECT COUNT(address_id) FROM address WHERE city = 'Villeneuve-la-Garenne'", "fuzzy_type": "typo", "used_tables": ["address"], "gold_columns": [{"table": "address", "column": "city"}], "matched_values": ["Villeneuve-la-Garenne"], "extracted_values_from_NL": ["Vlileneuve-la-Garenne"]},
    {"db_id": "books", "question": "What is the full name of the customers who live in Biayin city?", "evidence": "full name refers to first_name, last_name; 'Baiyin' is the city", "SQL": "SELECT T3.first_name, T3.last_name FROM address AS T1 INNER JOIN customer_address AS T2 ON T1.address_id = T2.address_id INNER JOIN customer AS T3 ON T3.customer_id = T2.customer_id WHERE T1.city = 'Baiyin'", "fuzzy_type": "typo", "used_tables": ["address", "customer", "customer_address"], "gold_columns": [{"table": "address", "column": "city"}], "matched_values": ["Baiyin"], "extracted_values_from_NL": ["Biayin"]},
    {"db_id": "car_retails", "question": "How many employees are there in Sdyney?", "evidence": "sales agent and sales representative are synonyms; Sydney is a city;", "SQL": "SELECT COUNT(employeeNumber) FROM employees WHERE officeCode = ( SELECT officeCode FROM offices WHERE city = 'Sydney' )", "fuzzy_type": "typo", "used_tables": ["employees"], "gold_columns": [{"table": "offices", "column": "city"}], "matched_values": ["Sydney"], "extracted_values_from_NL": ["Sdyney"]},
    {"db_id": "car_retails", "question": "List out full name of employees who are working in Tkoyo?", "evidence": "Tokyo is a city; full name = firstName+lastName;", "SQL": "SELECT T1.firstName, T1.lastName FROM employees AS T1 INNER JOIN offices AS T2 ON T1.officeCode = T2.officeCode WHERE T2.city = 'Tokyo'", "fuzzy_type": "typo", "used_tables": ["employees", "offices"], "gold_columns": [{"table": "offices", "column": "city"}], "matched_values": ["Tokyo"], "extracted_values_from_NL": ["Tkoyo"]},
    {"db_id": "car_retails", "question": "List out full name of employees who are working in Bsoton?", "evidence": "full name = contactFirstName, contactLastName; Boston is a city;", "SQL": "SELECT T1.firstName, T1.lastName FROM employees AS T1 INNER JOIN offices AS T2 ON T1.officeCode = T2.officeCode WHERE T2.city = 'Boston'", "fuzzy_type": "typo", "used_tables": ["employees", "offices"], "gold_columns": [{"table": "offices", "column": "city"}], "matched_values": ["Boston"], "extracted_values_from_NL": ["Bsoton"]},
    {"db_id": "chicago_crime", "question": "How many aldermen have \"Jmaes\" as their first name?", "evidence": "", "SQL": "SELECT COUNT(*) FROM Ward WHERE alderman_first_name = 'James'", "fuzzy_type": "typo", "used_tables": ["ward"], "gold_columns": [{"table": "Ward", "column": "alderman_first_name"}], "matched_values": ["James"], "extracted_values_from_NL": ["Jmaes"]},
    {"db_id": "codebase_comments", "question": "List the summary of the method \"Csatle.MonoRail.Framework.Test.StubViewComponentContext.RenderSection\".", "evidence": "", "SQL": "SELECT DISTINCT Summary FROM Method WHERE Name = 'Castle.MonoRail.Framework.Test.StubViewComponentContext.RenderSection'", "fuzzy_type": "typo", "used_tables": [], "gold_columns": [{"table": "Method", "column": "Name"}], "matched_values": ["Castle.MonoRail.Framework.Test.StubViewComponentContext.RenderSection"], "extracted_values_from_NL": ["Csatle.MonoRail.Framework.Test.StubViewComponentContext.RenderSection"]},
    {"db_id": "codebase_comments", "question": "Give the tokenized name for the method \"Spuay.Irc.Messages.KnockMessage.GetTokens\".", "evidence": "", "SQL": "SELECT NameTokenized FROM Method WHERE Name = 'Supay.Irc.Messages.KnockMessage.GetTokens'", "fuzzy_type": "typo", "used_tables": [], "gold_columns": [{"table": "Method", "column": "Name"}], "matched_values": ["Supay.Irc.Messages.KnockMessage.GetTokens"], "extracted_values_from_NL": ["Spuay.Irc.Messages.KnockMessage.GetTokens"]},
    {"db_id": "codebase_comments", "question": "Provide the tokenized name of the method \"Syk.Excel.ExcelBook.TypeConvert\".", "evidence": "tokenized name refers to NameTokenized; NameTokenized = 'Sky.Excel.ExcelBook.TypeConvert';", "SQL": "SELECT NameTokenized FROM Method WHERE Name = 'Sky.Excel.ExcelBook.TypeConvert'", "fuzzy_type": "typo", "used_tables": [], "gold_columns": [{"table": "Method", "column": "Name"}], "matched_values": ["Sky.Excel.ExcelBook.TypeConvert"], "extracted_values_from_NL": ["Syk.Excel.ExcelBook.TypeConvert"]},
    {"db_id": "disney", "question": "How many voice actors for the movie Aalddin?", "evidence": "Aladdin is the name of the movie which refers to movie = 'Aladdin';", "SQL": "SELECT COUNT('voice-actor') FROM `voice-actors` WHERE movie = 'Aladdin'", "fuzzy_type": "typo", "used_tables": ["voice-actors"], "gold_columns": [{"table": "voice-actors", "column": "movie"}], "matched_values": ["Aladdin"], "extracted_values_from_NL": ["Aalddin"]},
    {"db_id": "disney", "question": "How many voice-actors were involved in the Bmabi movie?", "evidence": "Bambi is the name of the movie which refers to movie = 'Bambi';", "SQL": "SELECT COUNT(DISTINCT 'voice-actor') FROM `voice-actors` WHERE movie = 'Bambi'", "fuzzy_type": "typo", "used_tables": ["voice-actors"], "gold_columns": [{"table": "voice-actors", "column": "movie"}], "matched_values": ["Bambi"], "extracted_values_from_NL": ["Bmabi"]},
    {"db_id": "genes", "question": "For the pairs of genes both from the class APTases, what is the average expression correlation score?", "evidence": "", "SQL": "SELECT AVG(T2.Expression_Corr) FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T1.Class = 'ATPases'", "fuzzy_type": "typo", "used_tables": ["genes", "interactions"], "gold_columns": [{"table": "Genes", "column": "Class"}], "matched_values": ["ATPases"], "extracted_values_from_NL": ["APTases"]},
    {"db_id": "legislator", "question": "How many current legislators chose Rpeublican as their political party?", "evidence": "chose Republican as their political party refers to party = 'Republican'", "SQL": "SELECT COUNT(*) FROM `current-terms` WHERE party = 'Republican'", "fuzzy_type": "typo", "used_tables": ["current-terms"], "gold_columns": [{"table": "current-terms", "column": "party"}], "matched_values": ["Republican"], "extracted_values_from_NL": ["Rpeublican"]},
    {"db_id": "movie", "question": "Which character has the longest screen time in the movie Btaman?", "evidence": "longest screen time refers to max(screentime); movie Batman refers to title = 'Batman'", "SQL": "SELECT T2.`Character Name` FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID WHERE T1.Title = 'Batman' ORDER BY T2.screentime DESC LIMIT 1", "fuzzy_type": "typo", "used_tables": ["characters", "movie", "screentime"], "gold_columns": [{"table": "movie", "column": "Title"}], "matched_values": ["Batman"], "extracted_values_from_NL": ["Btaman"]},
    {"db_id": "movie", "question": "How much longer in percentage is the screen time of the most important character in Btaman than the least important one?", "evidence": "most important character refers to max(screentime); least important character refers to min(screentime); movie Batman refers to title = 'Batman'; percentage = divide(subtract(max(screentime) , min(screentime)) , min(screentime)) * 100%", "SQL": "SELECT (MAX(CAST(SUBSTR(T2.screentime, 3, 2) AS REAL)) - MIN(CAST(SUBSTR(T2.screentime, 3, 2) AS REAL))) * 100 / MIN(CAST(SUBSTR(T2.screentime, 3, 2) AS REAL)) FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID WHERE T1.Title = 'Batman'", "fuzzy_type": "typo", "used_tables": ["characters", "movie"], "gold_columns": [{"table": "movie", "column": "Title"}], "matched_values": ["Batman"], "extracted_values_from_NL": ["Btaman"]},
    {"db_id": "movie", "question": "Count the male actors born in United States that starred in Gohst.", "evidence": "male refers to Gender = 'Male'; born in USA refers to Birth Country = 'USA'; Ghost refers to Title = 'Ghost'", "SQL": "SELECT COUNT(*) FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Title = 'Ghost' AND T3.Gender = 'Male' AND T3.`Birth Country` = 'USA'", "fuzzy_type": "typo", "used_tables": ["actor", "characters", "movie"], "gold_columns": [{"table": "movie", "column": "Title"}], "matched_values": ["Ghost"], "extracted_values_from_NL": ["Gohst"]},
    {"db_id": "movie_3", "question": "How many films are there under the category of \"Hroror\"?", "evidence": "\"Horror\" is the name of category", "SQL": "SELECT COUNT(T1.film_id) FROM film_category AS T1 INNER JOIN category AS T2 ON T1.category_id = T2.category_id WHERE T2.name = 'Horror'", "fuzzy_type": "typo", "used_tables": ["category", "film_category"], "gold_columns": [{"table": "category", "column": "name"}], "matched_values": ["Horror"], "extracted_values_from_NL": ["Hroror"]},
    {"db_id": "movies_4", "question": "The movie 'Gjoira ni-sen mireniamu' is from which country?", "evidence": "movie 'Gojira ni-sen mireniamu' refers to title = 'Gojira ni-sen mireniamu'; which country refers to country_name", "SQL": "SELECT T3.COUNTry_name FROM movie AS T1 INNER JOIN production_COUNTry AS T2 ON T1.movie_id = T2.movie_id INNER JOIN COUNTry AS T3 ON T2.COUNTry_id = T3.COUNTry_id WHERE T1.title = 'Gojira ni-sen mireniamu'", "fuzzy_type": "typo", "used_tables": ["country", "movie", "production_country"], "gold_columns": [{"table": "movie", "column": "title"}], "matched_values": ["Gojira ni-sen mireniamu"], "extracted_values_from_NL": ["Gjoira ni-sen mireniamu"]},
    {"db_id": "public_review_platform", "question": "What is the closing and opening time of businesses located at Tmepe with highest star rating?", "evidence": "located at Tempe refers to city = 'Tempe'; highest star rating refers to max(stars)", "SQL": "SELECT T2.closing_time, T2.opening_time FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id WHERE T1.city LIKE 'Tempe' ORDER BY T1.stars DESC LIMIT 1", "fuzzy_type": "typo", "used_tables": ["business", "business_hours", "stars"], "gold_columns": [{"table": "Business", "column": "city"}], "matched_values": ["Tempe"], "extracted_values_from_NL": ["Tmepe"]},
    {"db_id": "public_review_platform", "question": "What is the closing and opening time of businesses located at Glibert with highest star rating?", "evidence": "\"Gilbert\" is the name of city; highest star rating refers to Max(stars)", "SQL": "SELECT T2.closing_time, T2.opening_time FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id WHERE T1.city LIKE 'Gilbert' ORDER BY T1.stars DESC LIMIT 1", "fuzzy_type": "typo", "used_tables": ["business", "business_hours", "stars"], "gold_columns": [{"table": "Business", "column": "city"}], "matched_values": ["Gilbert"], "extracted_values_from_NL": ["Glibert"]},
    {"db_id": "public_review_platform", "question": "Among the businesses which have short length of review, which one located in Pohenix?", "evidence": "short length of review refers to review_length = 'Short'; in Phoenix refers to city = 'Phoenix'", "SQL": "SELECT DISTINCT T1.business_id FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.city = 'Phoenix' AND T2.review_length = 'Short'", "fuzzy_type": "typo", "used_tables": ["business", "reviews"], "gold_columns": [{"table": "Business", "column": "city"}], "matched_values": ["Phoenix"], "extracted_values_from_NL": ["Pohenix"]},
    {"db_id": "regional_sales", "question": "How many different time zones are there in the Nrotheast region?", "evidence": "", "SQL": "SELECT COUNT(DISTINCT T2.`Time Zone`) FROM Regions AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StateCode = T1.StateCode WHERE T1.Region = 'Northeast'", "fuzzy_type": "typo", "used_tables": ["regions", "store locations"], "gold_columns": [{"table": "Regions", "column": "Region"}], "matched_values": ["Northeast"], "extracted_values_from_NL": ["Nrotheast"]},
    {"db_id": "regional_sales", "question": "Which store in Airzona has the most net profit?", "evidence": "\"Arizona\" is the name of State; most net profit = Max(Subtract( Unit Price, Unit Cost))", "SQL": "SELECT T2.StoreID FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T2.State = 'Arizona' ORDER BY T1.`Unit Price` - T1.`Unit Cost` DESC LIMIT 1", "fuzzy_type": "typo", "used_tables": ["sales orders", "store locations"], "gold_columns": [{"table": "Regions", "column": "State"}], "matched_values": ["Arizona"], "extracted_values_from_NL": ["Airzona"]},
    {"db_id": "retail_complains", "question": "How many complaints have the client Deisel Glaloway filed?", "evidence": "", "SQL": "SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = 'Diesel' AND T1.last = 'Galloway'", "fuzzy_type": "typo", "used_tables": ["client", "t2"], "gold_columns": [{"table": "client", "column": "first"}], "matched_values": ["Diesel"], "extracted_values_from_NL": ["Deisel"]},
    {"db_id": "retail_complains", "question": "List all the complaints narratives made by the customer named Bernda and last name Myaer.", "evidence": "complaints narratives refers to \"Consumer complaint narrative\";", "SQL": "SELECT T2.`Consumer complaint narrative` FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = 'Brenda' AND T1.last = 'Mayer'", "fuzzy_type": "typo", "used_tables": ["client", "t2"], "gold_columns": [{"table": "client", "column": "first"}], "matched_values": ["Brenda"], "extracted_values_from_NL": ["Bernda"]},
    {"db_id": "retail_complains", "question": "How did Kryan Mluler submit his complaint?", "evidence": "how it was submitted refers to \"Submitted via\";", "SQL": "SELECT DISTINCT T2.`Submitted via` FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = 'Kyran' AND T1.last = 'Muller'", "fuzzy_type": "typo", "used_tables": ["client", "t2"], "gold_columns": [{"table": "client", "column": "first"}], "matched_values": ["Kyran"], "extracted_values_from_NL": ["Kryan"]},
    {"db_id": "retail_world", "question": "The sales of how many territories in total do the employees in Lnodon take charge of?", "evidence": "London refers to city = 'London';", "SQL": "SELECT COUNT(T2.TerritoryID) FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.City = 'London'", "fuzzy_type": "typo", "used_tables": ["employees", "employeeterritories"], "gold_columns": [{"table": "Customers", "column": "City"}], "matched_values": ["London"], "extracted_values_from_NL": ["Lnodon"]},
    {"db_id": "retail_world", "question": "Find and list the company name, company contact name, and contact title of customers from Mdarid.", "evidence": "from Madrid refers to City = 'Madrid'", "SQL": "SELECT CompanyName, ContactName, ContactTitle FROM Customers WHERE City = 'Madrid'", "fuzzy_type": "typo", "used_tables": ["customers"], "gold_columns": [{"table": "Customers", "column": "City"}], "matched_values": ["Madrid"], "extracted_values_from_NL": ["Mdarid"]},
    {"db_id": "retail_world", "question": "List the supplier company names located in Gremany.", "evidence": "located in Germany refers to Country = 'Germany';", "SQL": "SELECT CompanyName FROM Suppliers WHERE Country = 'Germany'", "fuzzy_type": "typo", "used_tables": ["suppliers"], "gold_columns": [{"table": "Customers", "column": "Country"}], "matched_values": ["Germany"], "extracted_values_from_NL": ["Gremany"]},
    {"db_id": "shipping", "question": "What is the average number of shipments done by the Kneworth trucks?", "evidence": "\"Kenworth\" is the make of truck; average = Divide (Count(ship_id where make = 'Kenworth'), Count(truck_id where make = 'Kenworth))", "SQL": "SELECT CAST(COUNT(T2.ship_id) AS REAL) / COUNT(DISTINCT T1.truck_id) FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id WHERE T1.make = 'Kenworth'", "fuzzy_type": "typo", "used_tables": ["shipment", "truck"], "gold_columns": [{"table": "truck", "column": "make"}], "matched_values": ["Kenworth"], "extracted_values_from_NL": ["Kneworth"]},
    {"db_id": "shipping", "question": "Among the top 5 heaviest shipments, how many shipments were transported via Mcak?", "evidence": "heaviest shipment refers to Max(weight); via Mack refers to make = 'Mack'", "SQL": "SELECT COUNT(T2.ship_id) FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id WHERE T1.make = 'Mack' ORDER BY T2.weight DESC LIMIT 1", "fuzzy_type": "typo", "used_tables": ["shipment", "truck", "weight"], "gold_columns": [{"table": "truck", "column": "make"}], "matched_values": ["Mack"], "extracted_values_from_NL": ["Mcak"]},
    {"db_id": "simpson_episodes", "question": "How many people were considered as prospective recipients of the \"Ainmation\" award?", "evidence": "prospective recipients refers to result = 'Nominee'", "SQL": "SELECT COUNT(*) FROM Award WHERE award = 'Animation' AND result = 'Nominee';", "fuzzy_type": "typo", "used_tables": ["award"], "gold_columns": [{"table": "Award", "column": "award"}], "matched_values": ["Animation"], "extracted_values_from_NL": ["Ainmation"]},
    {"db_id": "social_media", "question": "How many tweets in total were posted by a user in Agrentina?", "evidence": "\"Argentina\" is the Country", "SQL": "SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.Country = 'Argentina' LIMIT 1", "fuzzy_type": "typo", "used_tables": ["t2", "twitter"], "gold_columns": [{"table": "location", "column": "Country"}], "matched_values": ["Argentina"], "extracted_values_from_NL": ["Agrentina"]},
    {"db_id": "social_media", "question": "What is the average number of tweets posted by the users in a city in Agrentina?", "evidence": "\"Argentina\" is the Country; average number of tweets in a city = Divide (Count(TweetID where Country = 'Argentina'), Count (City))", "SQL": "SELECT SUM(CASE WHEN T2.City = 'Buenos Aires' THEN 1.0 ELSE 0 END) / COUNT(T1.TweetID) AS avg FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.Country = 'Argentina'", "fuzzy_type": "typo", "used_tables": ["t2", "twitter"], "gold_columns": [{"table": "location", "column": "Country"}], "matched_values": ["Argentina"], "extracted_values_from_NL": ["Agrentina"]},
    {"db_id": "social_media", "question": "What is the percentage of the tweets from Claifornia are positive?", "evidence": "\"California\" is the State; positive tweet refers to Sentiment > 0; percentage = Divide (Count(TweetID where Sentiment > 0), Count (TweetID)) * 100", "SQL": "SELECT SUM(CASE WHEN T1.Sentiment > 0 THEN 1.0 ELSE 0 END) / COUNT(T1.TweetID) AS percentage FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE State = 'California'", "fuzzy_type": "typo", "used_tables": ["t2", "twitter"], "gold_columns": [{"table": "location", "column": "State"}], "matched_values": ["California"], "extracted_values_from_NL": ["Claifornia"]},
    {"db_id": "superstore", "question": "Who is the customer from the Esat region that purchased the order with the highest profit?", "evidence": "highest profit refers to MAX(profit); Region = 'East'", "SQL": "SELECT T2.`Customer Name` FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.Region = 'East' ORDER BY T1.Profit DESC LIMIT 1", "fuzzy_type": "typo", "used_tables": ["east_superstore", "people", "profit"], "gold_columns": [{"table": "people", "column": "Region"}], "matched_values": ["East"], "extracted_values_from_NL": ["Esat"]},
    {"db_id": "superstore", "question": "Who is the customer from the Wset region that received the highest discount?", "evidence": "received the highest discount refers to MAX(discount); customer refers to \"Customer Name\"", "SQL": "SELECT T2.`Customer Name` FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.Region = 'West' ORDER BY T1.Discount DESC LIMIT 1", "fuzzy_type": "typo", "used_tables": ["discount", "people", "west_superstore"], "gold_columns": [{"table": "people", "column": "Region"}], "matched_values": ["West"], "extracted_values_from_NL": ["Wset"]},
    {"db_id": "superstore", "question": "List the products ordered by customers in Caochella.", "evidence": "in Coachella refers to City = 'Coachella'; products refers to \"Product Name\"", "SQL": "SELECT DISTINCT T3.`Product Name` FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T2.City = 'Coachella'", "fuzzy_type": "typo", "used_tables": ["people", "product", "west_superstore"], "gold_columns": [{"table": "people", "column": "City"}], "matched_values": ["Coachella"], "extracted_values_from_NL": ["Caochella"]},
    {"db_id": "works_cycles", "question": "What is the currency of Barzil?", "evidence": "", "SQL": "SELECT T1.Name FROM Currency AS T1 INNER JOIN CountryRegionCurrency AS T2 ON T1.CurrencyCode = T2.CurrencyCode INNER JOIN CountryRegion AS T3 ON T2.CountryRegionCode = T3.CountryRegionCode WHERE T3.Name = 'Brazil'", "fuzzy_type": "typo", "used_tables": ["countryregion", "countryregioncurrency", "currency"], "gold_columns": [{"table": "CountryRegion", "column": "Name"}], "matched_values": ["Brazil"], "extracted_values_from_NL": ["Barzil"]},
]


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

    # --- Hardcoded typo entries ---
    final.extend(_TYPO_ENTRIES)

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
