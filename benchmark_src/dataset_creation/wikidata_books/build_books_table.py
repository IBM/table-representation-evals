from pathlib import Path
import json
import time
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

from benchmark_src.dataset_creation.wikidata_books import hierarchy_utils, wikidata_utils

SAVE_DIR = Path("data/wikidata_genres/")
RESOURCES_DIR = Path("resources")

creation_random = 48573948

def get_properties_for_books(entity_qids):
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

    book_information = {}
    for row in result:
        # get book the information is about
        book_id = str(row['s']['value']).split("/")[-1]
        if not book_id in book_information:
            book_information[book_id] = {}

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
        
        book_information[book_id][property_info] = obj_label
        #print(book_id, "--", pid, p_label, "--", obj_id, obj_label)

    return book_information


def get_books_information(book_qid_list):
    n_batch = 20

    all_book_information = {}
    print(f"Processing {len(book_qid_list)} entities, {n_batch} at a time")

    for i in tqdm(range(0, len(book_qid_list), n_batch)):
        # force the query to wait a bit to avoid hitting query limits
        time.sleep(0.25)
        book_info = get_properties_for_books(book_qid_list[i:i + n_batch])
        all_book_information |= book_info

    return all_book_information

def build_initial_table(hierarchy, books_per_genre, book_label_lookup):
    # find leaves
    leaf_nodes = hierarchy_utils.get_leaves(hierarchy, hierarchy, [])
    print(f"Found {len(leaf_nodes)} leaf genres")

    # collect the books for each of the leaf genres
    books = {}
    for genre in leaf_nodes:
        genre_label, genre_id = wikidata_utils.deconstruct_label_id_string(genre)
        try:
            books[genre] = books_per_genre[genre_id]
        except:
            print(f"Could not find books for {genre}")

    all_books = []
    for book_list in books.values():
        all_books.extend(book_list)
    num_books_total = sum([len(x) for x in books.values()])
    print(f"Found {num_books_total} books in total, same as {len(all_books)}")

    # collect all properties for each book
    book_information = get_books_information(all_books)
    book_information_list = []
    for book_id, book_info in book_information.items():
        book_info["QID"] = book_id
        book_title = book_label_lookup[book_id]
        if not wikidata_utils.is_wikidata_id(book_title):
            book_info["label"] = book_title
            book_information_list.append(book_info)

    book_table = pd.DataFrame(book_information_list, index=["QID"])
    print(book_table)

    return book_table

# TODO: refactor code with other build_initial_table function
def build_books_table(all_genres, books_per_genre, book_label_lookup):
    book_table = pd.DataFrame()

    # collect the books for each of the leaf genres
    books = {}
    for genre in all_genres:
        genre_label, genre_id = wikidata_utils.deconstruct_label_id_string(genre)
        try:
            book_list = books_per_genre[genre_id]
            if len(book_list) > 0:
                books[genre] = book_list
            #print(f"Found {len(books[genre])} books for genre {genre_label}")
        except:
            print(f"Could not find books for {genre}")
            pass

    print(f"Total found books: {sum([len(x) for x in books.values()])} with {len(books.keys())} genres")
    
    ############ TODO: this filtering should be done later.... ########################
    all_books = []
    for book_list in books.values():
        # TODO: parameterize how many books 
        random.Random(creation_random).shuffle(book_list)
        chosen_books = book_list[:20]
        all_books.extend(chosen_books)
    print(f"Chose {len(all_books)} books in total")
    ####################################################################################

     # collect all properties for each book (#TODO: cache the information)
    book_information = get_books_information(all_books)
    book_information_list = []
    for book_id, book_info in book_information.items():
        book_info["QID"] = book_id
        book_title = book_label_lookup[book_id]
        if not wikidata_utils.is_wikidata_id(book_title):
            book_info["label"] = book_title
            book_information_list.append(book_info)

    book_table = pd.DataFrame(book_information_list)
    book_table = book_table.set_index("QID")
    
    print(book_table)

    return book_table

if __name__ == "__main__":

    # load the created hierarchy from disk
    with open(SAVE_DIR / "hierarchy_for_more_similar_than.json", "r") as file:
        hierarchy = json.load(file)

    # get all genres
    all_genres = hierarchy_utils.get_all_keys(hierarchy)

    # remove genre from the list
    all_genres.remove("genre___Q483394")
    print(f"Found {len(all_genres)} genres")

    # load book information from disk
    with open(SAVE_DIR / "books_per_genre.json", "r") as file:
        books_per_genre = json.load(file)
    with open(SAVE_DIR / "book_label_lookup.json", "r") as file:
        book_label_lookup = json.load(file)

    # TODO: need to add replacements to leaf nodes 
    ## TODO: need to save them to disk in create_hierarchy first

    rebuild_table = False

    if rebuild_table:
        book_table = build_books_table(all_genres, books_per_genre, book_label_lookup)
        book_table.to_csv(SAVE_DIR / "books_table_more_similar_than.csv")
    else:
        book_table = pd.read_csv(SAVE_DIR / "books_table_more_similar_than.csv", low_memory=False)
    if "Unnamed: 0" in book_table.columns:
        book_table = book_table.set_index("QID", drop=True)
        book_table = book_table.drop("Unnamed: 0", axis=1)
    
    first_columns = ["QID", "label", "author___P50", "description___None"]
    new_column_order = first_columns + [col for col in book_table.columns if col not in first_columns]
    book_table = book_table[new_column_order]

    print(f"Initial table has {len(book_table.columns)} columns")

    # filter properties (based on information from Sola)
    with open(RESOURCES_DIR / "wikimedia_related_properties.json", "r") as file:
        wikimedia_properties = json.load(file)
    with open(RESOURCES_DIR / "unique_identifiers.json", "r") as file:
        identifier_properties = json.load(file)

    to_remove = []
    for property in book_table.columns:
        try:
            label, id = wikidata_utils.deconstruct_label_id_string(label_id_string=property)
        except:
            label, id = property, property
        if id in wikimedia_properties.keys() or id in identifier_properties.keys():
            #print(f"Removing {id, label} column")
            to_remove.append(property)
    book_table.drop(to_remove, axis=1, inplace=True)
    print(f"Have {len(book_table.columns)} columns after removing wikimedia and id related properties")
    
    book_table = prune_sparse_columns(book_table, PERCENTAGE_NULL_ACCEPTED)
    print(f"Have {len(book_table.columns)} columns after removing too sparse columns")

    with open(SAVE_DIR / "books_removed_cols.json", "w") as file:
        json.dump(to_remove, file, indent=2)

    # shuffle the table
    book_table =  book_table.sample(frac=1)

    print(book_table)

    # create pandas dataframe and save to disk
    book_table.to_csv(SAVE_DIR / "books_table_more_similar_than_cleaned.csv", index=False)

