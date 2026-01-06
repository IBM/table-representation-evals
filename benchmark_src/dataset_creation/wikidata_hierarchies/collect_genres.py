from collections import defaultdict
import json
from time import time
import requests
from tqdm import tqdm

from benchmark_src.dataset_creation.wikidata_hierarchies import wikidata_utils

def get_books_and_genres(wikidata_url):
    """
    Retrieve all qids that are used together with the "genre" (P136) property in LiteraryWork items.
    Filters out genres that do not have an English label.

    Returns:
        dict:   Found genres and their labels {genre_qid: genre_label, ...}
    """

    sparql_str = f"""
        SELECT distinct ?s ?sLabel ?g ?gLabel
        WHERE {{
            ?s wdt:P136 ?g ;    # P136=genre (property)
            wdt:P31 wd:Q7725634 .   # P31=instanceOf Q7725634=LiteraryWork
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
    """

    res_get = requests.get(wikidata_url, params={'format': 'json',
                                        'query': sparql_str,
                                        },
                            headers={'User-Agent': 'CollectIdProperties/0.0 (liane.vogel@ibm.com)'})
    try:
        output_data = res_get.json()
    except:
        print("--- Error! --- res_get.json() failed --- res_get is: ")
        print(res_get)
        print("Query: ", sparql_str)
        output_data = {"results": {"bindings": []}}

    genres_per_book = {}
    book_label_lookup = {}
    genre_label_lookup = {}
    print(f"Got {len(output_data["results"]["bindings"])} possible genres")
    for genre in output_data["results"]["bindings"]:
        genre_id = genre["g"]["value"].split("/")[-1]
        genre_label = genre["gLabel"]["value"].lower()
        book_id = genre["s"]["value"].split("/")[-1]
        book_label = genre["sLabel"]["value"].lower()
        # filter out those that don't have an english label
        if not wikidata_utils.is_wikidata_id(genre_label):
            if book_id not in genres_per_book:
                genres_per_book[book_id] = []
            genres_per_book[book_id].append(genre_id)
            genre_label_lookup[genre_id] = genre_label
            book_label_lookup[book_id] = book_label
    print(f"Found: {len(genre_label_lookup.keys())} genres and {len(genres_per_book.keys())} books")
    return genres_per_book, book_label_lookup, genre_label_lookup

def get_superclasses(qid):
    """
    Return a list with all superclasses for the given qid.

    """
    sparql_query = f"""    
    SELECT distinct ?o ?oLabel
        WHERE {{
            wd:{qid} wdt:P279 ?o
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}"""

    res_get = requests.get("https://query.wikidata.org/sparql", params={'format': 'json',
                                                                        'query': sparql_query,
                                                                        },
                            headers={'User-Agent': 'CollectIdProperties/0.0 (liane.vogel@ibm.com)'})
    try:
        data = res_get.json()
    except:
        print(res_get)
        print(sparql_query)
        data = None

    if len(data['results']['bindings']) > 0:
        superclasses = []
        for row in data['results']['bindings']:
            supergenre_qid = str(row['o']['value']).split("/")[-1]
            supergenre_label = str(row['oLabel']['value']).lower()
            superclasses.append([supergenre_label, supergenre_qid])
        return superclasses
    else:
        #print(f"Nothing found for {i, collected_genres[i]}")
        return None

if __name__ == "__main__":
    wikidata_url = "https://query.wikidata.org/sparql"

    # get books and their genres from wikidata
    print(f"Retrieving all genres from wikidata")
    genres_per_book, book_label_lookup, genre_label_lookup = get_books_and_genres(wikidata_url)
    
    # group books by genre, keep only those with exactly one genre
    books_per_genre = {}
    single_genre = 0
    multiple_genres = 0
    for lit_work, genres in genres_per_book.items():
        for g in genres:
            if g not in books_per_genre.keys():
                books_per_genre[g] = []
        # do not take items with multiple genres into account for now
        if len(genres) > 1:
            multiple_genres += 1
        else:
            books_per_genre[genres[0]].append(lit_work)
            single_genre += 1

    # get genre classes that have at least 3 books
    min_used = 3
    genres_to_get_superclasses_for = []
    for genre, book_list in books_per_genre.items():
        if len(book_list) >= min_used:
            genres_to_get_superclasses_for.append(genre)

    # get superclasses for all the genres (also the superclasses of the superclasses and so on)
    genre_info = {}
    all_genre_labels = genre_label_lookup.copy()
    QIDs_to_process = genres_to_get_superclasses_for
    while len(QIDs_to_process) > 0:
        print(f"Getting superclasses for {len(QIDs_to_process)} genres")
        genres_newly_found = set()
        for i in tqdm(QIDs_to_process):
            superclasses = get_superclasses(i)
            if superclasses:
                for label, id in superclasses:
                    # check for english label
                    if not wikidata_utils.is_wikidata_id(label):
                        all_genre_labels[id] = label
                        if id not in genres_to_get_superclasses_for and id not in genre_info.keys():
                            genres_newly_found.add(id)
                genre_info[i] = {"label": all_genre_labels[i], "superclasses": superclasses}
            else:
                genre_info[i] = {"label": all_genre_labels[i], "superclasses": []}
        QIDs_to_process = genres_newly_found

    print(f"Now have {len(genre_info)} genres")

    for qid in genre_info.keys(): 
        if qid not in books_per_genre.keys():
            #print(qid, all_genre_labels[qid])
            genre_info[qid]["used_count"] = 0
        else:
            genre_info[qid]["used_count"] = len(books_per_genre[qid])

    with open("books_per_genre.json", "w") as file:
        json.dump(books_per_genre, file)

    with open("book_label_lookup.json", "w") as file:
        json.dump(book_label_lookup, file) 

    with open("lit_work_used_genres_info.json", "w") as file:
        json.dump(genre_info, file)