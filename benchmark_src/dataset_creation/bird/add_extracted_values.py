import json
import re
from typing import List, Set

def extract_values_from_nl(question: str, evidence: str) -> List[str]:
    """
    Extract literal values mentioned in the natural language query.
    Only uses question and evidence - does NOT use matched_values.
    """
    extracted = set()
    
    # Combine question and evidence
    text = question + " " + evidence
    
    # 1. Extract values from evidence "= 'value'" or "= value" patterns
    evidence_value_patterns = [
        r"=\s*'([^']+)'",  # = 'value'
        r'=\s*"([^"]+)"',  # = "value"
        r"=\s*(\d+)",      # = number
    ]

    for pattern in evidence_value_patterns:
        for match in re.finditer(pattern, evidence):
            value = match.group(1)
            if value and value not in ['NULL', 'null', 'Empty', 'empty']:
                extracted.add(value)

    # 1b. Evidence often states a value via a definition ("X" is the/a/an <column>) rather
    # than an "=" comparison, e.g. '"ARECIBO" is the county' or 'east is a direction'.
    definition_patterns = [
        r'^"([^"]+)"\s+is\s+(?:a|an|the)\b',
        r"^'([^']+)'\s+is\s+(?:a|an|the)\b",
        r'^([A-Za-z][\w\-]*)\s+is\s+(?:a|an|the)\b',
    ]
    for clause in re.split(r'[;.]', evidence):
        clause = clause.strip()
        for pattern in definition_patterns:
            m = re.match(pattern, clause)
            if m:
                extracted.add(m.group(1))
                break

    # 2. Extract dates in various formats from question
    date_patterns = [
        r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # MM-DD-YYYY or MM/DD/YYYY
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, question):
            date_str = match.group(1)
            # Normalize to YYYY-MM-DD format
            if '/' in date_str:
                parts = date_str.split('/')
                if len(parts) == 3:
                    # Check if it's YYYY/M/D or M/D/YYYY
                    if len(parts[0]) == 4:  # YYYY/M/D
                        date_str = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
                    else:  # M/D/YYYY
                        date_str = f"{parts[2]}-{int(parts[0]):02d}-{int(parts[1]):02d}"
            extracted.add(date_str)
    
    # 3. Extract standalone 4-digit years from question
    year_pattern = r'\b(19\d{2}|20\d{2})\b'
    for match in re.finditer(year_pattern, question):
        extracted.add(match.group(1))
    
    # 4. Extract long numbers (likely IDs) - 6+ digits from question
    id_pattern = r'\b(\d{6,})\b'
    for match in re.finditer(id_pattern, question):
        extracted.add(match.group(1))
    
    # 5. Extract dollar amounts from question (commas allowed as thousands separators)
    dollar_pattern = r'\$([\d,]+(?:\.\d+)?)'
    for match in re.finditer(dollar_pattern, question):
        extracted.add(match.group(1).replace(',', ''))

    # 6. Extract numbers mentioned with context in question (commas allowed)
    contextual_number_patterns = [
        r'user\s+(\d+)',
        r'id\s+(\d+)',
        r'level\s+(\d+)',
        r'over\s+\$?([\d,]+)',
        r'under\s+\$?([\d,]+)',
        r'than\s+\$?([\d,]+)',
    ]

    for pattern in contextual_number_patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            extracted.add(match.group(1).replace(',', ''))

    # 6b. Fallback for standalone numbers not caught above: thousands-separated numbers,
    # decimals (including negative, e.g. coordinates), and other multi-digit integers.
    general_number_pattern = r'-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d{2,}'
    for match in re.finditer(general_number_pattern, text):
        extracted.add(match.group(0).replace(',', ''))

    # 6c. Alphanumeric identifier tokens (e.g. "student829", "word1163") - contain both a
    # letter and a digit, so none of the numeric or word-based patterns above catch them.
    alnum_id_pattern = r'\b(?=[a-zA-Z0-9]*[a-zA-Z])(?=[a-zA-Z0-9]*\d)[a-zA-Z0-9]+\b'
    for match in re.finditer(alnum_id_pattern, text):
        extracted.add(match.group(0))

    # 7. Extract proper names from question - capitalized sequences that are likely names.
    # matched_values stores a name as a single string (e.g. "Francis Ford Coppola"), so keep
    # the whole matched span in addition to its individual word parts.
    name_pattern = r"\b([A-Z][a-zA-Z'\-]*(?:\s+[A-Z][a-zA-Z'\-]*){1,4})\b"
    for match in re.finditer(name_pattern, question):
        name = match.group(1)
        extracted.add(name)
        for part in name.split():
            if len(part) > 0:
                extracted.add(part)
    
    # 8. Extract country/city names and other proper nouns from question
    # Look for capitalized words that aren't at sentence start
    words = question.split()
    for i, word in enumerate(words):
        # Skip first word (might be question word)
        if i == 0:
            continue
        # Check if word is capitalized and not a common word
        if word and word[0].isupper() and word not in [
            'How', 'What', 'When', 'Where', 'Who', 'Which', 'State', 'Give', 
            'Tell', 'Show', 'Among', 'For', 'Was', 'Were', 'Are', 'Is',
            'The', 'A', 'An', 'In', 'On', 'At', 'To', 'From', 'By', 'Of',
            'And', 'Or', 'But', 'Not', 'If', 'Then', 'Else', 'Do', 'Does',
            'Did', 'Has', 'Have', 'Had', 'Can', 'Could', 'Will', 'Would',
            'Should', 'May', 'Might', 'Must', 'Shall', 'Mr', 'Mrs', 'Ms']:
            # Remove punctuation
            clean_word = word.rstrip('.,;:!?')
            if len(clean_word) > 1:
                extracted.add(clean_word)
    
    # 9. Extract quoted strings from question only (not evidence to avoid SQL fragments).
    # Requires a non-word character (or string boundary) on both sides of each quote mark,
    # so an apostrophe inside a contraction/possessive ("don't", "team's") is never treated
    # as an opening or closing quote.
    quoted_patterns = [
        r'(?<!\w)"([^"]{1,300})"(?!\w)',
        r"(?<!\w)'([^']{1,300})'(?!\w)",
    ]
    for pattern in quoted_patterns:
        for match in re.finditer(pattern, question):
            value = match.group(1)
            if value and len(value) > 0:
                extracted.add(value)
    
    # Remove empty strings
    extracted = {v for v in extracted if v and len(v.strip()) > 0}
    
    return sorted(list(extracted))

def process_all_queries():
    """
    Read the JSON file, add extracted_values_from_NL to each query, and save.
    """
    print("Reading cell_value_matching_queries.json...")
    with open('cell_value_matching_queries.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    print(f"Processing {len(queries)} queries...")
    
    for i, query in enumerate(queries):
        if i % 500 == 0:
            print(f"Progress: {i}/{len(queries)} ({i*100//len(queries)}%)")
        
        question = query.get('question', '')
        evidence = query.get('evidence', '')
        
        # Extract values from natural language ONLY
        extracted_values = extract_values_from_nl(question, evidence)
        
        # Add the new field
        query['extracted_values_from_NL'] = extracted_values
    
    print(f"\nProcessed all {len(queries)} queries.")
    print("Writing to cell_value_matching_queries.json...")
    
    with open('cell_value_matching_queries.json', 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    
    print("Done! File updated successfully.")
    
    # Calculate match statistics
    stats = {
        'total': len(queries),
        'full_match': 0,
        'partial_match': 0,
        'no_match': 0,
        'over_extracted': 0
    }
    
    for query in queries:
        matched = set(query.get('matched_values', []))
        extracted = set(query.get('extracted_values_from_NL', []))
        
        if matched == extracted:
            stats['full_match'] += 1
        elif extracted.issuperset(matched) and len(extracted) > len(matched):
            stats['over_extracted'] += 1
        elif extracted.intersection(matched):
            stats['partial_match'] += 1
        else:
            stats['no_match'] += 1
    
    print(f"\nStatistics (comparison with matched_values):")
    print(f"  Total queries: {stats['total']}")
    print(f"  Exact match: {stats['full_match']} ({stats['full_match']*100//stats['total']}%)")
    print(f"  Over-extracted (includes all + more): {stats['over_extracted']} ({stats['over_extracted']*100//stats['total']}%)")
    print(f"  Partial match: {stats['partial_match']} ({stats['partial_match']*100//stats['total']}%)")
    print(f"  No match: {stats['no_match']} ({stats['no_match']*100//stats['total']}%)")

if __name__ == "__main__":
    process_all_queries()
