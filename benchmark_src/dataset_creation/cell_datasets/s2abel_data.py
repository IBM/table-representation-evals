from pathlib import Path
from typing import List
from hydra.utils import get_original_cwd
import os
import requests
import tarfile
from tqdm import tqdm
import json
import pandas as pd
import re

from benchmark_src.dataset_creation.cell_datasets import s2abel_testcases

def download_s2abel_dataset(output_path: Path):
    """
    Download the S2abEL dataset from the official GitHub repository.
    
    Args:
        output_dir: Directory where the dataset will be saved
    """
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Dataset URL from the official repository
    dataset_url = "https://github.com/allenai/S2abEL/raw/main/data/release_data.tar.gz"
    
    # Path for the downloaded tar file
    tar_path = output_path / "release_data.tar.gz"
    
    print(f"Downloading S2abEL dataset from {dataset_url}")
    print(f"Saving to: {output_path}")

    # Download with progress bar
    response = requests.get(dataset_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(tar_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    
    print(f"\nDownload complete! File saved to: {tar_path}")
    
    # Extract the tar.gz file
    print("\nExtracting dataset...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=output_path)
    
    print(f"Extraction complete! Dataset extracted to: {output_path}")
    
    # List the extracted contents
    print("\nDataset contents:")
    for item in sorted(output_path.rglob('*')):
        if item.is_file() and item != tar_path:
            print(f"  - {item.relative_to(output_path)}")
    
    # Remove the tar file to save space
    tar_path.unlink()
    print("Tar file removed.")
    
patterns = {
            # Remove XML/HTML-like tags
            'xml_tags': re.compile(r'<[^>]+>'),
            # Remove LaTeX references like \ref{...}
            'latex_ref': re.compile(r'\\ref\{[^}]+\}'),
            # Remove LaTeX citations like \cite{...}
            'latex_cite': re.compile(r'\\cite\{[^}]+\}'),
            # Remove other LaTeX commands
            'latex_cmd': re.compile(r'\\[a-zA-Z]+\{[^}]*\}'),
            # Remove standalone LaTeX commands
            'latex_standalone': re.compile(r'\\[a-zA-Z]+'),
            # Clean up multiple spaces
            'multi_space': re.compile(r'\s+'),
            # Remove special unicode characters that might be artifacts
            'unicode_artifacts': re.compile(r'[\u2020\u2217\u00a0]'),
        }

def clean_cell(text: str) -> str:
    """
    Clean a single text string by removing all markup.
    
    Args:
        text: Raw text with potential markup
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return text
    
    cleaned = text
    
    # Apply all cleaning patterns in order
    cleaned = patterns['xml_tags'].sub('', cleaned)
    cleaned = patterns['latex_ref'].sub('', cleaned)
    cleaned = patterns['latex_cite'].sub('', cleaned)
    cleaned = patterns['latex_cmd'].sub('', cleaned)
    cleaned = patterns['latex_standalone'].sub('', cleaned)
    cleaned = patterns['unicode_artifacts'].sub('', cleaned)
    
    # Clean up whitespace
    cleaned = patterns['multi_space'].sub(' ', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

def clean_table_data(table_data: List[List[str]]) -> List[List[str]]:
    """
    Clean all cells in a table.
    
    Args:
        table_data: 2D list representing table rows and columns
        
    Returns:
        Cleaned table data
    """
    cleaned_table = []
    for row in table_data:
        cleaned_row = [clean_cell(cell) for cell in row]
        cleaned_table.append(cleaned_row)
    return cleaned_table

def load_and_clean_papers(data_path: Path):

    papers_path = data_path / "papers.jsonl"

    # load papers.jsonl 
    papers = []
    with open(papers_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                paper = json.loads(line)
                if line_num == 1:
                    print(paper.keys())
                if 'tables' in paper and isinstance(paper['tables'], dict):
                    cleaned_tables = {}
                    for table_name, table_data in paper['tables'].items():
                        cleaned_tables[table_name] = clean_table_data(table_data)
                    paper['tables'] = cleaned_tables

                    # try to load each table into pandas to check if it is valid
                    for table_name, table_data in paper['tables'].items():
                        try:
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            # make sure that there are rows
                            if df.shape[0] == 0:
                                print(f"Warning: Table {table_name} in paper {paper.get('title', 'unknown')} has no rows.")
                        except Exception as e:
                            print(f"Error loading table {table_name} in paper {paper.get('title', 'unknown')} into pandas: {e}")
                    
                    # remove abstract and references field to reduce size
                    del paper['abstract']  
                    del paper['references']
                    
                    papers.append(paper)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Loaded and cleaned {len(papers)} papers")

    # save papers_cleaned.jsonl file
    cleaned_papers_path = data_path / "papers_cleaned.jsonl"
    with open(cleaned_papers_path, 'w', encoding='utf-8') as f:
        for paper in papers:
            json_line = json.dumps(paper)
            f.write(json_line + '\n')

    return papers

    # extract each table, try to load it into pandas, clean the latex code


# # load papers_cleaned.jsonl and save each table as separate csv file
# def save_tables_as_csv(data_path: Path):
#     cleaned_papers_path = data_path / "papers_cleaned.jsonl"
#     tables_output_path = data_path / "tables_csv"
#     tables_output_path.mkdir(parents=True, exist_ok=True)

#     with open(cleaned_papers_path, 'r', encoding='utf-8') as f:
#         for line_num, line in enumerate(f, 1):
#             try:
#                 paper = json.loads(line)
#                 paper_id = paper.get('arxiv_id', f'paper_{line_num}')
#                 if 'tables' in paper and isinstance(paper['tables'], dict):
#                     for table_name, table_data in paper['tables'].items():
#                         try:
#                             # "1903.07785v1/table_03.csv"
#                             table_name = table_name.split('/')[-1].replace('.csv', '')
#                             df = pd.DataFrame(table_data[1:], columns=table_data[0])
#                             csv_filename = f"{paper_id}_{table_name}.csv"
#                             csv_path = tables_output_path / csv_filename
#                             df.to_csv(csv_path, index=False)
#                         except Exception as e:
#                             print(f"Error saving table {table_name} from paper {paper_id} to CSV: {e}")
#             except json.JSONDecodeError as e:
#                 print(f"Error parsing line {line_num}: {e}")
#                 continue

class SimpleTableFilter:
    """
    Strict heuristic filter for detecting relational tables.
    """

    def __init__(
        self,
        min_rows: int = 3,
        min_cols: int = 2,
    ):
        self.min_rows = min_rows
        self.min_cols = min_cols

    def is_relational_table(self, table_data: List[List[str]]) -> bool:
        if not table_data:
            return False

        # ---------- 1. Basic size checks ----------
        if len(table_data) < self.min_rows:
            return False

        col_counts = [len(row) for row in table_data]
        if not col_counts:
            return False

        num_cols = col_counts[0]

        if num_cols < self.min_cols:
            return False

        # ---------- 2. Strict structural consistency ----------
        # No row/column variance allowed
        if any(count != num_cols for count in col_counts):
            return False

        # ---------- 4. Header validation & uniqueness ----------
        header = table_data[0]
        header_cells = [cell.strip() for cell in header if cell and cell.strip()]

        # Require sufficiently populated header
        if len(header_cells) < num_cols * 0.5:
            return False

        # Header values must be unique
        if len(set(header_cells)) != len(header_cells):
            return False

        # ---------- 5. Metadata / annotation row detection ----------
        for row in table_data[1:]:
            non_empty = [cell.strip() for cell in row if cell and cell.strip()]

            # Rows repeating the same value across columns are likely metadata
            if len(non_empty) > 1 and len(set(non_empty)) == 1:
                return False

        # ---------- 6. Column-wise variation check ----------
        columns = list(zip(*table_data))
        low_variance_cols = 0

        for col in columns:
            values = [cell.strip() for cell in col[1:] if cell and cell.strip()]
            if len(values) >= 2 and len(set(values)) == 1:
                low_variance_cols += 1

        # If nearly all columns lack variation, it's not relational
        if low_variance_cols >= num_cols - 1:
            return False

        return True


def save_relational_tables_as_csv(data_path: Path, verbose: bool = False):
    """Save only relational tables as CSV files."""
    
    cleaned_papers_path = data_path / "papers_cleaned.jsonl"
    tables_output_path = data_path / "tables_csv"
    tables_output_path.mkdir(parents=True, exist_ok=True)
    
    table_filter = SimpleTableFilter()
    
    total_tables = 0
    saved_tables = 0
    
    with open(cleaned_papers_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                paper = json.loads(line)
                paper_id = paper.get('arxiv_id', f'paper_{line_num}')
                
                if 'tables' in paper and isinstance(paper['tables'], dict):
                    for table_name, table_data in paper['tables'].items():
                        total_tables += 1
                        
                        # Check if table is relational
                        if not table_filter.is_relational_table(table_data):
                            if verbose:
                                print(f"Skipped: {paper_id}/{table_name}")
                            continue
                        
                        try:
                            # Save table
                            table_name_clean = table_name.split('/')[-1].replace('.csv', '')
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            
                            csv_filename = f"{paper_id}_{table_name_clean}.csv"
                            csv_path = tables_output_path / csv_filename
                            df.to_csv(csv_path, index=False)
                            
                            saved_tables += 1
                            
                        except Exception as e:
                            if verbose:
                                print(f"Error saving {table_name}: {e}")
                
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Error parsing line {line_num}: {e}")
    
    print(f"\nSaved {saved_tables}/{total_tables} tables to {tables_output_path}")
    return saved_tables, total_tables

def create_s2abel_dataset(cfg):
    cache_path_dataset = Path(get_original_cwd()) / Path(cfg.cache_dir) / "cell_level_data" / cfg.dataset_name
    
    # TODO: wieder einfügen
    #download_s2abel_dataset(cache_path_dataset)
    
    #load_and_clean_papers(cache_path_dataset)

    # TODO: continue with saving tables as csv files and creating metadata
    #save_relational_tables_as_csv(cache_path_dataset)

    # create triplet testcases out of entity linking annotations
    entity_linking_data = s2abel_testcases.restructure_entity_linking_annotations(
        file=cache_path_dataset / "entity_linking.jsonl",
        table_folder=cache_path_dataset / "tables_csv"
        )
    
    # s2abel_testcases.generate_triplet_testcases(
    #     entity_links=entity_linking_data,
    #     csv_folder=cache_path_dataset / "tables_csv",
    #     output_dir=cache_path_dataset / "testcases_triplets",
    #     max_testcases=1000
    #     )

    s2abel_testcases.generate_cell_retrieval_testcases(
        entity_links=entity_linking_data,
        csv_folder=cache_path_dataset / "tables_csv",
        output_dir=cache_path_dataset / "testcases_retrieval_consistency",
        num_testcases=1000
        )