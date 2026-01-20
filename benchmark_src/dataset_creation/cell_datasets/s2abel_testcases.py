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
    for entity, papers in entity_links.items():
        entity_class = entity.split("/")[-2]  # e.g., 'method'
        for paper_id, tables in papers.items():
            for table_id, cells in tables.items():
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
    candidate_cells = [c for tbl in table_list for c in table_to_cells[tbl]]
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


def generate_cell_retrieval_testcases(
    entity_links,
    csv_folder="data",
    output_dir="retrieval_testcases",
    num_testcases=50,
    seed=42
):
    """
    Generate entity-coherent cell retrieval testcases:
    - Each testcase has a query cell
    - All tables that contain the same entity as the query (up to 3) are included
    - Ground truth contains all cells with that entity in those tables
    - top_k = number of matching cells
    """
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # --- Flatten all cells and map entity to cells ---
    all_cells = {}
    entity_to_cells = defaultdict(list)
    table_to_cells = defaultdict(set)

    print(f"Processing entity linking data for testcase generation, having {len(entity_links)} entities.")

    for entity, papers in entity_links.items():
        entity_class = entity.split("/")[-2]  # e.g., 'method'
        if entity_class in ["dataset"]:
            continue  # skip less useful entities
        for paper_id, tables in papers.items():
            for table_id, cells in tables.items():
                table_filename = f"{paper_id}_{table_id}"
                table_to_cells[table_filename].update(cells)
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
                    entity_to_cells[entity].append(cell_id)

    all_entities = list(entity_to_cells.keys())
    random.shuffle(all_entities)

    print(f"Total entities for retrieval testcase generation: {len(all_entities)}")

    testcase_counter = 0

    for entity in all_entities:
        if testcase_counter >= num_testcases:
            break

        entity_cells = entity_to_cells[entity]
        if not entity_cells:
            continue

        # --- Pick a random query cell ---
        query_id = random.choice(entity_cells)
        query_info = all_cells[query_id]

        # --- Find all tables that contain this entity ---
        tables_with_entity = set()
        for cell_id in entity_cells:
            paper_id = all_cells[cell_id]["paper_id"]
            table_id = all_cells[cell_id]["table_id"]
            tables_with_entity.add(f"{paper_id}_{table_id}")

        # --- Always include query's table ---
        query_table = f"{query_info['paper_id']}_{query_info['table_id']}"
        tables_with_entity.add(query_table)

        # --- Limit to at most 3 tables ---
        if len(tables_with_entity) > 3:
            other_tables = list(tables_with_entity - {query_table})
            sampled = random.sample(other_tables, 3 - 1)
            tables_with_entity = [query_table] + sampled
        else:
            tables_with_entity = list(tables_with_entity)

        # --- Load table data ---
        table_data = {}
        for tbl in tables_with_entity:
            table_path = os.path.join(csv_folder, tbl)
            if os.path.exists(table_path):
                df = pd.read_csv(table_path, header=None, dtype=str).fillna("")
                table_data[tbl] = df

        # --- Collect ground truth cells ---
        ground_truth = []
        for tbl in tables_with_entity:
            df = table_data[tbl]
            for cell_id in table_to_cells[tbl]:
                if all_cells[cell_id]["entity"] == entity:
                    row_idx, col_idx = all_cells[cell_id]["row"], all_cells[cell_id]["col"]
                    cell_text = df.iloc[row_idx, col_idx]
                    header_text = df.iloc[0, col_idx] if row_idx != 0 else df.iloc[row_idx, col_idx]
                    ground_truth.append({
                        **all_cells[cell_id],
                        "text": str(cell_text),
                        "header": str(header_text)
                    })

        top_k = len(ground_truth)

        # --- Build query cell info ---
        query_df = table_data[query_table]
        row_idx, col_idx = query_info["row"], query_info["col"]
        query_text = query_df.iloc[row_idx, col_idx]
        query_header = query_df.iloc[0, col_idx] if row_idx != 0 else query_df.iloc[row_idx, col_idx]

        # --- Build testcase JSON ---
        testcase = {
            "tables": tables_with_entity,
            "query": {
                **query_info,
                "text": str(query_text),
                "header": str(query_header)
            },
            "ground_truth": ground_truth,
            "top_k": top_k
        }

        # --- Save JSON ---
        testcase_counter += 1
        filename = os.path.join(output_dir, f"retrieval_testcase_{testcase_counter:03d}.json")
        with open(filename, "w") as f:
            json.dump(testcase, f, indent=2)

    print(f"Generated {testcase_counter} entity-coherent cell-retrieval testcases in '{output_dir}'")
