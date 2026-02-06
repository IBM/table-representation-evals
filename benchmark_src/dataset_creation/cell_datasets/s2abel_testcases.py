from collections import defaultdict
import os
import random
import json
import pandas as pd
from pathlib import Path

def restructure_entity_linking_annotations(file, table_folder):
    mentions = {}
    all_tables_mentioned = set()
    with open(file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            cell_link_dict = json.loads(line)
            # {"cell_id":"1704.00051v2\/table_06.csv\/1\/2","pwc_url":"https:\/\/paperswithcode.com\/dataset\/squad"}
            paper_id = cell_link_dict['cell_id'].split('/')[0]
            table_id = cell_link_dict['cell_id'].split('/')[1]
            paper_table_id = f"{paper_id}_{table_id}"
            all_tables_mentioned.add(paper_table_id)
            #print(table_folder / Path(paper_table_id))
            # Check if table file exists
            if (table_folder / Path(paper_table_id)).is_file():
                if cell_link_dict["pwc_url"] not in mentions and cell_link_dict["pwc_url"] != "0":
                    mentions[cell_link_dict["pwc_url"]] = {}
                if cell_link_dict["pwc_url"] != "0":
                    if paper_id not in mentions[cell_link_dict["pwc_url"]]:
                        mentions[cell_link_dict["pwc_url"]][paper_id] = {}
                    if table_id not in mentions[cell_link_dict["pwc_url"]][paper_id]:
                        mentions[cell_link_dict["pwc_url"]][paper_id][table_id] = set()
                    mentions[cell_link_dict["pwc_url"]][paper_id][table_id].add(cell_link_dict["cell_id"])
            else:
                #print(f"Skipping {paper_table_id} as table file not found.")
                pass

    # remove mentions with only one paper
    mentions = {k: v for k, v in mentions.items() if len(v) > 1}

    print(f"Total tables mentioned in entity linking: {len(all_tables_mentioned)}, with {len(mentions)} mentions in multiple papers.")

    return mentions



def generate_triplet_testcases(
    entity_links,
    csv_folder="data",
    output_dir="testcases",
    max_testcases=50,
    seed=42
):
    """
    Generate deterministic, diverse single-triplet testcases with hard negatives.
    Ensures no triplets repeat and anchor-positive pairs vary.
    """
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Flatten all cells and map table -> cells
    all_cells = {}
    table_to_cells = {}
    cell_to_class = {}
    for entity in sorted(entity_links):
        papers = entity_links[entity]
        entity_class = entity.split("/")[-2]  # e.g., 'method'
        for paper_id in sorted(papers):
            tables = papers[paper_id]
            for table_id in sorted(tables):
                cells = tables[table_id]
                table_filename = f"{paper_id}_{table_id}"
                table_to_cells.setdefault(table_filename, set())
                for cell_id in cells:
                    paper, table, row, col = cell_id.split("/")
                    row, col = int(row), int(col)
                    all_cells[cell_id] = {
                        "paper_id": paper,
                        "table_id": table,
                        "row": row,
                        "col": col,
                        "entity": entity
                    }
                    table_to_cells[table_filename].add(cell_id)
                    cell_to_class[cell_id] = entity_class

    table_list = list(table_to_cells.keys())
    testcase_counter = 0
    used_triplets = set()  # track (anchor, positive, negative)

    # Precompute all candidate anchor cells and shuffle deterministically
    candidate_cells = [c for tbl in table_list for c in sorted(table_to_cells[tbl])]
    print(f"Total candidate cells for triplet generation: {len(candidate_cells)}")
    random.shuffle(candidate_cells)

    while testcase_counter < max_testcases:
        if not candidate_cells:
            break
        anchor_id = candidate_cells.pop()  # take one anchor

        anchor_entity = all_cells[anchor_id]["entity"]
        anchor_class = cell_to_class[anchor_id]

        # Positive candidates: same entity, different cell
        positive_candidates = [c for c in candidate_cells if all_cells[c]["entity"] == anchor_entity and c != anchor_id]
        if not positive_candidates:
            continue
        random.shuffle(positive_candidates)  # shuffle for diversity
        positive_id = positive_candidates[0]

        # Negative candidates: different entity class
        negative_candidates = [c for c in candidate_cells if cell_to_class[c] != anchor_class]
        if not negative_candidates:
            continue
        random.shuffle(negative_candidates)
        negative_id = negative_candidates[0]

        triplet_key = (anchor_id, positive_id, negative_id)
        if triplet_key in used_triplets:
            continue
        used_triplets.add(triplet_key)

        # Tables used in this triplet
        triplet_tables = sorted(list(set([
            f"{all_cells[anchor_id]['paper_id']}_{all_cells[anchor_id]['table_id']}",
            f"{all_cells[positive_id]['paper_id']}_{all_cells[positive_id]['table_id']}",
            f"{all_cells[negative_id]['paper_id']}_{all_cells[negative_id]['table_id']}"
        ])))

        # Load CSVs for these tables
        table_data = {}
        for tbl in triplet_tables:
            table_path = os.path.join(csv_folder, tbl)
            if os.path.exists(table_path):
                df = pd.read_csv(table_path, header=None, dtype=str).fillna("")
                table_data[tbl] = df

        def get_cell_text_and_header(cell_id):
            info = all_cells[cell_id]
            tbl_name = f"{info['paper_id']}_{info['table_id']}"
            df = table_data[tbl_name]
            row_idx, col_idx = info["row"], info["col"]
            cell_text = df.iloc[row_idx, col_idx]
            header_text = df.iloc[0, col_idx] if row_idx != 0 else df.iloc[row_idx, col_idx]
            return str(cell_text), str(header_text)

        anchor_text, anchor_header = get_cell_text_and_header(anchor_id)
        positive_text, positive_header = get_cell_text_and_header(positive_id)
        negative_text, negative_header = get_cell_text_and_header(negative_id)

        triplet = {
            "tables": triplet_tables,
            "triplets": [
                {
                    "anchor": {**all_cells[anchor_id], "text": anchor_text, "header": anchor_header},
                    "positive": {**all_cells[positive_id], "text": positive_text, "header": positive_header},
                    "negative": {**all_cells[negative_id], "text": negative_text, "header": negative_header}
                }
            ]
        }

        # Save JSON file
        testcase_counter += 1
        filename = os.path.join(output_dir, f"testcase_{testcase_counter:03d}.json")
        with open(filename, "w") as f:
            json.dump(triplet, f, indent=2)


# TODO: Idea: could make testcases more difficult by including cells from the same table but only querying other tables
def generate_cell_retrieval_testcases(
    entity_links,
    csv_folder="data",
    output_dir="retrieval_testcases",
    num_testcases=50,
    seed=42
):
    """
    Generate entity-coherent cell retrieval testcases.

    Each testcase contains:
    - A query cell (the "anchor")
    - At least one other table containing the same entity
    - Ground-truth cells: all other cells of the same entity in the selected tables (query excluded)
    - top_k: number of ground-truth cells

    Testcase JSON structure:
    {
        "tables": [...list of table filenames involved...],
        "query": {paper_id, table_id, row, col, text, header, entity},
        "ground_truth": [{paper_id, table_id, row, col, text, header, entity}, ...],
        "top_k": int
    }

    Args:
        entity_links (dict): Mapping of entity -> paper -> table -> set of cell_ids (paper/table/row/col)
        csv_folder (str): Path to folder containing CSV files
        output_dir (str): Folder to save testcases
        num_testcases (int): Number of testcases to generate
        seed (int): Random seed for determinism
    """
     # ----------------------------- Setup -----------------------------
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    all_cells = {} # cell_id -> metadata (paper, table, row, col, entity)
    entity_to_cells = defaultdict(list)  # entity -> list of cell_ids
    table_to_cells = defaultdict(set) # table_filename -> set of cell_ids

    # --- Flatten cells ---
    for entity in sorted(entity_links):
        # skip certain entity types
        entity_class = entity.split("/")[-2]
        if entity_class in ["dataset"]:
            continue

        papers = entity_links[entity]
        for paper_id in sorted(papers):
            tables = papers[paper_id]
            for table_id in sorted(tables):
                cells = tables[table_id]
                table_filename = f"{paper_id}_{table_id}"
                table_to_cells[table_filename].update(cells)

                for cell_id in cells:
                    paper, table, row, col = cell_id.split("/")
                    all_cells[cell_id] = {
                        "paper_id": paper,
                        "table_id": table,
                        "row": int(row),
                        "col": int(col),
                        "entity": entity
                    }
                    entity_to_cells[entity].append(cell_id)

    # --- Build query pool (all possible (entity, query_cell) pairs) ---
    query_pool = []
    for entity in sorted(entity_to_cells):
        cells = entity_to_cells[entity]
        for cell_id in cells:
            query_pool.append((entity, cell_id))

    random.shuffle(query_pool)

    # ----------------------------- Generate testcases -----------------------------
    seen = set() # to avoid duplicate testcases
    testcase_counter = 0

    for entity, query_id in query_pool:
        if testcase_counter >= num_testcases:
            break

        query_info = all_cells[query_id]
        query_table = f"{query_info['paper_id']}_{query_info['table_id']}"

        # --- Tables containing this entity ---
        tables_with_entity = set(
            f"{all_cells[c]['paper_id']}_{all_cells[c]['table_id']}"
            for c in entity_to_cells[entity]
        )

        # Always include query table
        if query_table not in tables_with_entity:
            tables_with_entity = [query_table] + tables_with_entity

        
        # Skip if not enough tables for a meaningful retrieval testcase
        if len(tables_with_entity) < 2:
            continue

        # --- Limit to 3 tables but keep query table ---
        if len(tables_with_entity) > 3:
            others = sorted(tables_with_entity - {query_table})
            tables_with_entity = [query_table] + random.sample(others, 2)
        else:
            tables_with_entity = list(tables_with_entity)

        # ---------------- Avoid duplicate testcases ----------------
        tables_key = frozenset(tables_with_entity)
        testcase_key = (entity, query_id, tables_key)

        if testcase_key in seen:
            continue
        seen.add(testcase_key)

        # --- Load tables ---
        table_data = {}
        for tbl in tables_with_entity:
            path = os.path.join(csv_folder, tbl)
            if os.path.exists(path):
                table_data[tbl] = pd.read_csv(path, header=None, dtype=str).fillna("")

        # --- Ground truth (exclude query) ---
        ground_truth = []
        for tbl in tables_with_entity:
            df = table_data[tbl]
            for cell_id in sorted(table_to_cells[tbl]):
                if all_cells[cell_id]["entity"] == entity:
                    # skip the query cell
                    if cell_id == query_id:
                        continue
                    r, c = all_cells[cell_id]["row"], all_cells[cell_id]["col"]
                    ground_truth.append({
                        **all_cells[cell_id],
                        "text": str(df.iloc[r, c]),
                        "header": str(df.iloc[0, c] if r != 0 else df.iloc[r, c])
                    })

        if not ground_truth:
            continue # skip if no other cells exist

        # --- Query info ---
        qdf = table_data[query_table]
        r, c = query_info["row"], query_info["col"]

        testcase = {
            "tables": tables_with_entity,
            "query": {
                **query_info,
                "text": str(qdf.iloc[r, c]),
                "header": str(qdf.iloc[0, c] if r != 0 else qdf.iloc[r, c])
            },
            "ground_truth": ground_truth,
            "top_k": len(ground_truth)
        }

        testcase_counter += 1
        with open(
            os.path.join(output_dir, f"retrieval_testcase_{testcase_counter:03d}.json"),
            "w"
        ) as f:
            json.dump(testcase, f, indent=2)

    print(f"Generated {testcase_counter} unique retrieval testcases")