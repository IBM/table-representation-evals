from pathlib import Path
import json
import time
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

from benchmark_src.dataset_creation.wikidata_hierarchies import hierarchy_utils, wikidata_utils

SAVE_DIR = Path("./cache/dataset_creation_resources/wikidata_astronomical_objects")
RESOURCES_DIR = Path("cache/dataset_creation_resources/resources")

creation_random = 48573948

def prune_sparse_columns(dataframe, null_threshold: float):
    """
    Prunes the dataframe in the following ways:
        - removes too sparse columns where more than null_threshold percent values are missing

        Args:
            dataframe (Dataframe): The dataframe to be filtered
            null_threshold (float): Percentage of null values in a column, columns with a higher percentage get deleted

        Returns:
            dataframe: The filtered dataframe
    """
    min_num_values = int(len(dataframe) * (1 - null_threshold))
    print(f"Columns must have at least {min_num_values} values to be kept")
    dataframe = dataframe.dropna(axis="columns", thresh=min_num_values)
    return dataframe


def get_properties_for_instances(entity_qids):
    sparql_query = f"""
    SELECT ?s ?p ?propLabel ?o ?oLabel 
    WHERE {{
    ?s ?p ?o .
    OPTIONAL {{
    ?prop wikibase:directClaim ?p
    }}
    MINUS {{?o wikibase:rank ?r}}
    FILTER(!isLiteral(?o) || lang(?o) = "" || langMatches(lang(?o), "EN"))
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    VALUES ?s {{ {' '.join(['wd:'+qid for qid in entity_qids])} }}
    }}
    """
    res_get = requests.get("https://query.wikidata.org/sparql", params={'format': 'json',
                                                                        'query': sparql_query,
                                                                        },
                            headers={'User-Agent': 'CollectIdProperties/0.0 (liane.vogel@ibm.com)'})
    try:
        data = res_get.json()
    except:
        print("Got exception: .json() not working", res_get)
        print(sparql_query)
        data = None

    result = data['results']['bindings']
    if len(result) == 0:
        print("Empty result")

    instance_information = {}
    for row in result:
        # get the instance the information is about
        instance_id = str(row['s']['value']).split("/")[-1]
        if not instance_id in instance_information:
            instance_information[instance_id] = {}

        # get property id
        pid = str(row['p']['value']).split("/")[-1]
        p_label = None
        if "propLabel" in row:
            p_label = str(row["propLabel"]["value"])
        else:
            if pid in ["description", "version"]:
                p_label = pid
                pid = "None"
            else:
                # do not keep information without a label
                continue

        # get object id and value
        obj_id = str(row['o']['value']).split("/")[-1]
        if not wikidata_utils.is_wikidata_id(obj_id):
            obj_id = None
        obj_label = str(row['oLabel']['value'])
        if wikidata_utils.is_wikidata_id(obj_label):
            # do not save values without an english label
            continue

        property_info = wikidata_utils.construct_id_label_string(id=pid, label=p_label)
        #obj_info = wikidata_utils.construct_id_label_string(id=obj_id, label=obj_label)
        
        instance_information[instance_id][property_info] = obj_label

    return instance_information


def get_instance_information(instance_qid_list):
    n_batch = 20

    all_instance_information = {}
    print(f"Processing {len(instance_qid_list)} entities, {n_batch} at a time")

    for i in tqdm(range(0, len(instance_qid_list), n_batch)):
        # force the query to wait a bit to avoid hitting query limits
        time.sleep(0.25)
        instance_info = get_properties_for_instances(instance_qid_list[i:i + n_batch])
        all_instance_information |= instance_info

    return all_instance_information

def build_initial_table(hierarchy, items_per_subclass, item_label_lookup):
    # find leaves
    leaf_nodes = hierarchy_utils.get_leaves(hierarchy, hierarchy, [])
    print(f"Found {len(leaf_nodes)} leaf classes in the hierarchy")

    # collect the items for each of the leaf classes
    entities = {}
    for item_class in leaf_nodes:
        class_label, class_id = wikidata_utils.deconstruct_label_id_string(item_class)
        try:
            entities[item_class] = items_per_subclass[class_id]
        except:
            print(f"Could not find concrete entities for {item_class}")

    all_entities = []
    for entity_list in entities.values():
        all_entities.extend(entity_list)
    num_entities_total = sum([len(x) for x in entities.values()])
    print(f"Found {num_entities_total} entities in total, same as {len(all_entities)}")

    # collect all properties for each entity
    entity_information = get_instance_information(all_entities)
    entity_information_list = []
    for entity_id, entity_info in entity_information.items():
        entity_info["QID"] = entity_id
        entity_title = item_label_lookup[entity_id]
        if not wikidata_utils.is_wikidata_id(entity_title):
            entity_info["label"] = entity_title
            entity_information_list.append(entity_info)

    full_table = pd.DataFrame(entity_information_list, index=["QID"])
    print(full_table)

    return full_table

# TODO: refactor code with other build_initial_table function
def build_table(all_classes, items_per_subclass, item_label_lookup):
    created_table = pd.DataFrame()

    # collect the entities for each of the leaf classes
    entities = {}
    for entity_class in all_classes:
        class_label, class_id = wikidata_utils.deconstruct_label_id_string(entity_class)
        try:
            entity_list = items_per_subclass[class_id]
            if len(entity_list) > 0:
                entities[entity_class] = entity_list
            #print(f"Found {len(entities[genre])} entities for genre {genre_label}")
        except:
            print(f"Could not find entities for {entity_class}")
            pass

    print(f"Total found entities: {sum([len(x) for x in entities.values()])} with {len(entities.keys())} classes")
    
    # maybe filter later?
    all_entities = []
    for entity_list in entities.values():
        # TODO: parameterize how many entities? Take 200 for now to have a manageable table size
        random.Random(creation_random).shuffle(entity_list)
        chosen_entities = entity_list[:200]
        all_entities.extend(chosen_entities)
    print(f"Chose {len(all_entities)} instances in total")

     # collect all properties for each entity (#TODO: cache the information)
    entity_information = get_instance_information(all_entities)
    entity_information_list = []
    for entity_id, entity_info in entity_information.items():
        entity_info["QID"] = entity_id
        entity_title = item_label_lookup[entity_id]
        if not wikidata_utils.is_wikidata_id(entity_title):
            entity_info["label"] = entity_title
            entity_information_list.append(entity_info)

    created_table = pd.DataFrame(entity_information_list)
    created_table = created_table.set_index("QID")
    
    print(created_table)

    return created_table

if __name__ == "__main__":

    # load the created hierarchy from disk
    print(SAVE_DIR.exists())
    with open(SAVE_DIR / "astronomical_hierarchy.json", "r") as file:
        hierarchy = json.load(file)

    # get all classes
    all_classes = hierarchy_utils.get_all_keys(hierarchy)

    # remove root class from the list
    all_classes.remove("astronomical object___Q6999")
    print(f"Found {len(all_classes)} classes in the hierarchy")

    # load instance information from disk
    with open(SAVE_DIR / "old" / "items_per_subclass.json", "r") as file:
        items_per_subclass = json.load(file)
    with open(SAVE_DIR / "old" /"items_labels.json", "r") as file:
        item_label_lookup = json.load(file)

    rebuild_table = True

    if rebuild_table:
        created_table = build_table(all_classes, items_per_subclass, item_label_lookup)
        created_table.to_csv(SAVE_DIR / "astronomical_table_initial.csv")
    else:
        created_table = pd.read_csv(SAVE_DIR / "astronomical_table_initial.csv", low_memory=False)
    

    print(f"Have columns: {created_table.columns}")

    if "Unnamed: 0" in created_table.columns:
        print(f"Have to set new index")
        created_table = created_table.set_index("QID", drop=True)
        created_table = created_table.drop("Unnamed: 0", axis=1)
    
    #first_columns = ["QID", "label", "description___None"]
    #new_column_order = first_columns + [col for col in created_table.columns if col not in first_columns]
    #created_table = created_table[new_column_order]

    print(f"Initial table has {len(created_table.columns)} columns")

    # filter properties (based on information from Sola)
    with open(RESOURCES_DIR / "wikimedia_related_properties.json", "r") as file:
        wikimedia_properties = json.load(file)
    with open(RESOURCES_DIR / "unique_identifiers.json", "r") as file:
        identifier_properties = json.load(file)

    to_remove = []
    for property in created_table.columns:
        try:
            label, id = wikidata_utils.deconstruct_label_id_string(label_id_string=property)
        except:
            label, id = property, property
        if id in wikimedia_properties.keys() or id in identifier_properties.keys():
            #print(f"Removing {id, label} column")
            to_remove.append(property)
    created_table.drop(to_remove, axis=1, inplace=True)
    print(f"Have {len(created_table.columns)} columns after removing wikimedia and id related properties")
    

    ### Prune the table:
    # first remove extremely sparse columns (>=98% empty)
    # then remove sparse rows (<20% filled)
    # remove columns that are >=95% empty

    # remove cols where more than 98% of the values are empty
    created_table = prune_sparse_columns(created_table, 0.98)
    
    # keep only rows with at least 20% columns filled
    created_table = created_table.loc[created_table.notna().mean(axis=1) >= 0.20]

    # remove cols where more than 95% of the values are empty
    created_table = prune_sparse_columns(created_table, 0.95)
    print(f"Have {len(created_table.columns)} columns after removing too sparse columns")

    with open(SAVE_DIR / "table_removed_cols_for_debugging.json", "w") as file:
        json.dump(to_remove, file, indent=2)

    # shuffle the table
    created_table =  created_table.sample(frac=1)

    print(created_table)

    # create pandas dataframe and save to disk
    created_table.to_csv(SAVE_DIR / "astronomical_table.csv", index=True)