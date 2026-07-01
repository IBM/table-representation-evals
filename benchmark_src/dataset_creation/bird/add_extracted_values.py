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
    
    # 5. Extract dollar amounts from question
    dollar_pattern = r'\$(\d+(?:\.\d+)?)'
    for match in re.finditer(dollar_pattern, question):
        extracted.add(match.group(1))
    
    # 6. Extract numbers mentioned with context in question
    contextual_number_patterns = [
        r'user\s+(\d+)',
        r'id\s+(\d+)',
        r'level\s+(\d+)',
        r'over\s+\$?(\d+)',
        r'under\s+\$?(\d+)',
        r'than\s+\$?(\d+)',
    ]
    
    for pattern in contextual_number_patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            extracted.add(match.group(1))
    
    # 7. Extract proper names from question
    # Look for capitalized sequences that are likely names
    # Pattern: FirstName MiddleInitial LastName or FirstName LastName
    name_patterns = [
        r'\b([A-Z][a-z]+\s+[A-Z]\s+[A-Z][a-z]+)\b',  # First M Last
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',          # First Last
    ]
    
    for pattern in name_patterns:
        for match in re.finditer(pattern, question):
            name = match.group(1)
            # Split the name into parts
            parts = name.split()
            for part in parts:
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
    
    # 9. Extract quoted strings from question only (not evidence to avoid SQL fragments)
    quoted_pattern = r'"([^"]+)"'
    for match in re.finditer(quoted_pattern, question):
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

# Made with Bob
