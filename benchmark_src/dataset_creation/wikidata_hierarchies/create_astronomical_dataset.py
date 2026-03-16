import pandas as pd
import json
from pathlib import Path
import random
import itertools

# deterministic shuffling
RANDOM_SEED = 42

def create_statistics(dataset_name, input_table_df, save_path, primary_key_column):
    statistics_dict = {
        "dataset_name": dataset_name,
        "input_table_num_rows": len(input_table_df),
        "input_table_num_cols": len(input_table_df.columns),
        "primary_key_column": primary_key_column,
        "datatypes": input_table_df.dtypes.astype(str).to_dict(),
    }

    num_empty_cells = float((input_table_df.isnull().sum()).sum())
    sparsity = float(num_empty_cells / input_table_df.size)
    statistics_dict["num_empty_cells"] = num_empty_cells
    statistics_dict["sparsity"] = sparsity

    with open(save_path / "dataset_information.json", "w") as file:
        json.dump(statistics_dict, file, indent=2)

import pandas as pd
import json
import random

random.seed(42)
random_gen = random.Random(RANDOM_SEED)

# --- Hardcoded branch mapping ---
BRANCH_MAPPING = {
    "variable_stars": [
        "variable star", "long-period variable star", "Mira variable", 
        "pulsating variable star", "Delta Scuti variable", "RR Lyrae variable",
        "Cepheid variable", "classical Cepheid variable", "eruptive variable star",
        "flare star", "Orion variable", "T Tauri-type star", "rotating variable star",
        "BY Draconis variable", "irregular variable"
    ],
    "galaxies": [
        "galaxy", "elliptical galaxy", "dwarf galaxy", "dwarf spheroidal galaxy", 
        "irregular galaxy", "active galactic nucleus", "Seyfert galaxy", 
        "Seyfert 1 galaxy", "Seyfert 2 galaxy", "quasar", "blazar", 
        "BL Lacertae object", "radio galaxy"
    ],
    "star_clusters": [
        "star cluster", "open cluster", "moving group", "globular cluster",
        "stellar association", "stellar stream"
    ],
    "nebulae": [
        "nebula", "dark nebula", "reflection nebula", "emission nebula",
        "H II region", "planetary nebula", "supernova remnant", 
        "Herbig–Haro object", "molecular cloud"
    ],
    "meteorites": [
        "meteorite", "iron meteorite", "stony meteorite", "chondrite",
        "carbonaceous chondrite", "CK chondrite", "ordinary chondrite"
    ]
}

# Reverse mapping: label -> branch
LABEL_TO_BRANCH = {label.lower(): branch for branch, labels in BRANCH_MAPPING.items() for label in labels}

def create_astronomy_testcases(astro_table, branch_mapping, max_testcases=2000):
    """
    Generate up to max_testcases for astronomy objects, deterministic and diverse.

    Args:
        astro_table (pd.DataFrame): cleaned astronomical table with 'QID', 'instance of___P31', 'label'
        branch_mapping (dict): Mapping of high-confidence branches to allowed class labels
        max_testcases (int): Maximum number of testcases to generate

    Returns:
        list: testcases in format:
              {"similar_pair": {"a": ..., "b": ...},
               "dissimilar_pair": {"a": ..., "c": ...},
               "difficulty": ..., "category": ...}
    """
    testcases = []

    # Map QID -> instance_of label
    qid_to_class = dict(zip(astro_table["QID"], astro_table["instance of___P31"]))

    # Group QIDs by branch
    branch_to_qids = {}
    for branch, labels in branch_mapping.items():
        branch_qids = astro_table[astro_table["instance of___P31"].isin(labels)]["QID"].tolist()
        if len(branch_qids) >= 2:  # only include branches that can make pairs
            branch_to_qids[branch] = sorted(branch_qids)

    print(f"Creating testcases for {len(branch_to_qids)} branches...")

    pairs_generated = 0
    branch_cycle = sorted(branch_to_qids.keys())

    # Deterministic stride generator for diversity
    def deterministic_pairs(qids):
        n = len(qids)
        step = 1
        while True:
            for i in range(n):
                j = (i + step) % n
                if i != j:
                    yield qids[i], qids[j]
            step += 1
            if step >= n:
                step = 1

    branch_generators = {branch: deterministic_pairs(qids) for branch, qids in branch_to_qids.items()}

    while pairs_generated < max_testcases:
        for branch in branch_cycle:
            gen = branch_generators[branch]
            try:
                a, b = next(gen)
            except StopIteration:
                continue

            # Pick a dissimilar branch deterministically (next branch in cycle)
            other_branches = [br for br in branch_cycle if br != branch]
            if not other_branches:
                continue
            opposite_branch = other_branches[pairs_generated % len(other_branches)]
            c_qids = branch_to_qids[opposite_branch]
            c = c_qids[pairs_generated % len(c_qids)]

            testcases.append({
                "similar_pair": {
                    "a": {"qid": a, "instance_of": qid_to_class[a]},
                    "b": {"qid": b, "instance_of": qid_to_class[b]}
                },
                "dissimilar_pair": {
                    "a": {"qid": a, "instance_of": qid_to_class[a]},
                    "c": {"qid": c, "instance_of": qid_to_class[c]}
                },
                "difficulty": "medium",
                "category": "same-branch"
            })

            pairs_generated += 1
            if pairs_generated >= max_testcases:
                break
        if pairs_generated >= max_testcases:
            break

    print(f"Generated {len(testcases)} testcases.")
    return testcases

def save_testcases(testcases, dataset_save_dir):
    (dataset_save_dir / "test_cases").mkdir(exist_ok=True)
    for idx,  testcase in enumerate(testcases):
        with open(dataset_save_dir / "test_cases" / f"{idx}.json", "w") as file:
            json.dump(testcase, file, indent=2)


def remove_numerical_cols(cache_dir):
    save_folder = cache_dir / "@only-text"
    save_folder.mkdir(exist_ok=True)
    # load input_table.csv from cache dir
    input_table_path = cache_dir / "original" / "input_table.csv"
    input_table = pd.read_csv(input_table_path)
    # remove columns with pandas type "float64" or "int64"
    numerical_cols = input_table.select_dtypes(include=["float64", "int64"]).columns
    input_table = input_table.drop(columns=numerical_cols)

    # also remove cols "orbital period___P2146", "apparent magnitude___P1215", "apoapsis___P2243"
    input_table = input_table.drop(columns=["orbital period___P2146", "apparent magnitude___P1215", "apoapsis___P2243"], errors='ignore')

    # save the modified table
    input_table.to_csv(save_folder / "input_table.csv", index=False)
    print(f"Removed numerical columns from input_table, saved new table to {save_folder / 'input_table.csv'}")

    create_statistics(dataset_name='only-text', 
            input_table_df=input_table, 
            save_path=save_folder,
            primary_key_column="QID"
            )
    
    print(f"Saved variation with removed numerical columns to {save_folder}")




def create_astronomy_dataset():
    """
    Prepare astronomy dataset for test case generation.
    
    Steps:
    1. Load cleaned table and hierarchy.
    2. Reorder columns: QID, label, class, description first.
    3. Remove rows missing essential data (QID, label, instance/class).
    4. Shuffle deterministically.
    5. Save prepared table and statistics to disk.
    """

    cache_dir = Path("./cache/dataset_creation_resources/wikidata_astronomical_objects")
    input_csv_path = cache_dir / "astronomical_table_cleaned.csv"
    hierarchy_json_path = cache_dir / "old" / "astronomical_hierarchy_manual.json"
    dataset_save_dir = Path("./cache/datasets/more_similar_than/astronomical_objects")
    dataset_save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Load table and hierarchy
    # ---------------------------
    table = pd.read_csv(input_csv_path, low_memory=False)

    with open(hierarchy_json_path, "r") as f:
        hierarchy = json.load(f)

    print(f"Loaded table with {len(table)} rows and {len(table.columns)} columns")

    # ---------------------------
    # Determine essential columns
    # ---------------------------
    class_col = next((col for col in table.columns if "instance of" in col.lower() or "class" in col.lower()), None)
    qid_col = "QID"
    label_col = "label"
    desc_col = next((col for col in table.columns if "description" in col.lower()), None)

    if class_col is None:
        raise ValueError("No instance/class column found in input CSV")

    first_columns = [qid_col, label_col, class_col, desc_col]
    first_columns = [c for c in first_columns if c is not None]
    remaining_columns = [c for c in table.columns if c not in first_columns]

    table = table[first_columns + remaining_columns]

    # ---------------------------
    # Remove rows with missing essential info
    # ---------------------------
    essential_cols = [qid_col, label_col, class_col]
    table = table.dropna(subset=essential_cols)
    table = table[table[qid_col] != ""]  # in case empty string
    table = table[table[label_col] != ""]
    table = table[table[class_col] != ""]

    print(f"Table after cleaning: {len(table)} rows remain")

    # ---------------------------
    # Deterministic shuffle
    # ---------------------------
    table = table.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # ---------------------------
    # Save prepared table
    # ---------------------------
    # save dataset as "original" variation
    original_data_dir = dataset_save_dir / 'original'
    original_data_dir.mkdir(exist_ok=True)
    table.to_csv(original_data_dir / "input_table.csv", index=False)
    print(f"Saved prepared table to {original_data_dir}")

    # ---------------------------
    # Create dataset statistics
    # ---------------------------
    create_statistics(
        dataset_name="astronomy_dataset",
        input_table_df=table,
        save_path=original_data_dir,
        primary_key_column=qid_col
    )
    print(f"Saved dataset statistics to {original_data_dir / 'dataset_information.json'}")

    # ---------------------------
    # testcases
    # ---------------------------
    testcases = create_astronomy_testcases(astro_table=table, branch_mapping=BRANCH_MAPPING)

    # save the testcases
    save_testcases(testcases, dataset_save_dir)

    print(f"Saved {len(testcases)} testcases")

    return table, hierarchy



if __name__ == "__main__":
    create_astronomy_dataset()