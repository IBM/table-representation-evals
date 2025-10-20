import requests
from collections import defaultdict, Counter
import time
import json
import argparse
from tqdm import tqdm
import copy
import re
import jsonlines

IS_WIKIDATA_ID = r"Q\d+"
def is_wikidata_id(text):
    pattern = re.compile(IS_WIKIDATA_ID)
    return bool(pattern.search(text))

def collect_lit_works_by_genre(entity_qids, url):
    sparql_str = f"""
    SELECT distinct ?s ?sLabel ?g
    WHERE {{
        ?s wdt:P136 ?g ;    # P136=genre (property)
          wdt:P31 wd:Q7725634 .   # P31=instanceOf Q7725634=LiteraryWork
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        VALUES ?g {{ {' '.join(['wd:'+qid for qid in entity_qids])} }}
    }}
    """
    # print(sparql_str)
    res_get = requests.get(url, params={'format': 'json',
                                        'query': sparql_str,
                                        },
                           headers={'User-Agent': 'CollectIdProperties/0.0 (liane.vogel@ibm.com)'})
    try:
        output_data = res_get.json()
    except:
        print("--- Error! --- res_get.json() failed --- res_get is: ")
        print(res_get)
        output_data = {"results": {"bindings": []}}
    literary_works = defaultdict(lambda: [])
    literary_works_labels = {}
    for row in output_data['results']['bindings']:
        item_q_id = str(row['s']['value']).split("/")[-1]
        item_genre = str(row['g']['value']).split("/")[-1]
        item_label = str(row['sLabel']['value'])
        # remove literary works that don't have english labels
        if not is_wikidata_id(item_label):
            literary_works[item_q_id].append(item_genre)
            literary_works_labels[item_q_id] = item_label 

    return literary_works, literary_works_labels


def collect_lit_works(ent_list):
    wikidata_url = "https://query.wikidata.org/sparql"
    finished_entities = set(ent_list)
    n_batch = 50

    labels = {}
    all_output = {}
    while ent_list:
        print(f"Processing {len(ent_list)} entities, {n_batch} at a time")
        new_entries = 0
        new_entities = []
        for i in tqdm(range(0, len(ent_list), n_batch)):
            # force the query to wait a bit to avoid hitting query limits
            time.sleep(0.25)
            output, out_labels = collect_lit_works_by_genre(entity_qids=ent_list[i:i + n_batch], url=wikidata_url)
            all_output |= output
            labels |= out_labels
            for k, vlist in output.items():
                for v in vlist:
                    if v not in finished_entities:
                        finished_entities.add(v)
                        new_entities.append(v)
                        new_entries += 1

        print(f"{new_entries} new entries added to process.")
        ent_list = list(set(new_entities))
    with open('lit_work_genres.json', 'w') as f:
        json.dump(all_output, f)
    with open('lit_work_labels.json', 'w') as f:
        json.dump(labels, f)

def collect_genre_data():
    """
    Collect information about all genres:
        - first get all subclasses of the "genre" class
        - 
    

    """

    # get genre subclasses first because trying to get everything in one query gets a bit messy
    sparql_str = """
    SELECT distinct ?genre ?genreLabel
    WHERE {
        ?genre wdt:P279* wd:Q483394 .   # P270=subclassOf Q483394=genre
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    res_get = requests.get("https://query.wikidata.org/sparql", params={'format': 'json',
                                        'query': sparql_str,
                                        },
                           headers={'User-Agent': 'CollectIdProperties/0.0 (liane.vogel@ibm.com)'})
    try:
        data = res_get.json()
    except:
        print(res_get)
    genres = set()
    labels = {}
    for row in data['results']['bindings']:
        genres.add(str(row['genre']['value']).split("/")[-1])
        labels[str(row['genre']['value']).split("/")[-1]] = str(row['genreLabel']['value'])

    # also get all genres that are instanceOf any genre sublcass (instanceOf and subclassOf are not consistently used in Wikidata)
    if True:
        query_2 = """
        SELECT distinct ?genre ?genreLabel
        WHERE {
            ?genreClasses wdt:P279* wd:Q483394 . # any subclass of genre
            ?genre wdt:P31 ?genreClasses . # P31=instanceOf (any instance of the genre class)
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }

        }"""
        res_get = requests.get("https://query.wikidata.org/sparql", params={'format': 'json',
                                            'query': query_2,
                                            },
                            headers={'User-Agent': 'CollectIdProperties/0.0 (liane.vogel@ibm.com)'})
        try:
            data = res_get.json()
        except:
            print(res_get)
        for row in data['results']['bindings']:
            genres.add(str(row['genre']['value']).split("/")[-1])
            labels[str(row['genre']['value']).split("/")[-1]] = str(row['genreLabel']['value'])

    # remove genres without an english label
    bad_genres = set()
    for q_id, label in labels.items():
        if label.split("/")[-1] == q_id:
            bad_genres.add(q_id)
    print(f"removing {len(bad_genres)} genres that don't have an english label")
    genres -= bad_genres
    for bg in bad_genres:
        labels.pop(bg)
    print(f"{len(genres)} genres to process")

    genre_subclass_info = defaultdict(lambda: [])

    # get subclass/instance relations of genre entities
    QIDs = list(labels.keys())
    for i in range(0, len(labels), 300):
        # batch to avoid long queries
        query_3 = f"""    
        SELECT distinct ?s ?o
            WHERE {{
                ?s ?p ?o .
                VALUES ?p {{ wdt:P279 wdt:P31 }}

            VALUES ?s {{ {' '.join(['wd:'+qid for qid in QIDs[i:i+300]])} }}
            }}"""
        res_get = requests.get("https://query.wikidata.org/sparql", params={'format': 'json',
                                                                            'query': query_3,
                                                                            },
                               headers={'User-Agent': 'CollectIdProperties/0.0 (liane.vogel@ibm.com)'})
        try:
            data = res_get.json()
        except:
            print(res_get)

        for row in data['results']['bindings']:
            if str(row['o']['value']).split("/")[-1] in labels:
                genre_subclass_info[str(row['s']['value']).split("/")[-1]].append(str(row['o']['value']).split("/")[-1])
    with open('genre_subclass_info.json', 'w') as f:
        json.dump(dict(genre_subclass_info), f)
    with open('genre_labels.json', 'w') as f:
        json.dump(labels, f)

    genres = list(labels.keys())
    return genres

def clean_and_expand():
    with open('genre_subclass_info.json', 'r') as f:
        subclass_info_dict = json.load(f)
    with open('genre_labels.json', 'r') as f:
        genre_labels = json.load(f)
    with open('lit_work_genres.json', 'r') as f:
        lit_work_genres = json.load(f)
    with open("lit_work_labels.json", "r") as file:
        lit_work_labels = json.load(file)

    # First clean subclass info dict for these that are actually used by literature
    all_genres_used_in_lit_work_ids = []

    for lit_work_id, used_genres in lit_work_genres.items():
        lit_work_name = lit_work_labels[lit_work_id]
        if not is_wikidata_id(lit_work_name):
            for used_genre in used_genres:
                all_genres_used_in_lit_work_ids.append(used_genre)
        else:
            print("Missed one")

    print(f"Have {len(all_genres_used_in_lit_work_ids)} that are actually used by literary work items")

    ### also add all superclasses of the genres that are actually used in literary work!
    superclass_genres_to_add = set()
    for genre_id in all_genres_used_in_lit_work_ids:
        if genre_id in subclass_info_dict.keys():
            superclasses = subclass_info_dict[genre_id]
            for superclass in superclasses:
                if superclass not in all_genres_used_in_lit_work_ids:
                    superclass_genres_to_add.add(genre_id)

    print(f"Adding {len(superclass_genres_to_add)} additional genres that are used as superclasses")
    all_genres_used_in_lit_work_ids += list(superclass_genres_to_add)
            

    used_genre_counter = Counter(all_genres_used_in_lit_work_ids)
    used_genre_info_dict = {}
    seen_genre_labels = []
    for genre_id, used_count in dict(used_genre_counter).items():
        try:
            subclass_ids = subclass_info_dict[genre_id]
        except KeyError:
            print(f"{genre_id} {genre_labels[genre_id]} not found")
            continue
        superclasses_with_labels = [(genre_labels[class_id], class_id) for class_id in subclass_ids]
        if genre_labels[genre_id] in seen_genre_labels:
            print(f"have duplicate label {genre_labels[genre_id], genre_id} ")
        if genre_id in superclass_genres_to_add:
            used_count = 0
        seen_genre_labels.append(genre_labels[genre_id])
        used_genre_info_dict[genre_id] = {"label": genre_labels[genre_id],
                                            "QID": genre_id, 
                                            "url": f"https://www.wikidata.org/wiki/{genre_id}", 
                                            "used_count": used_count,
                                            "superclasses": superclasses_with_labels }

    used_genre_info_dict = dict(sorted(used_genre_info_dict.items(), key=lambda item: item[1]["used_count"], reverse=True))

    with jsonlines.open("lit_work_used_genres_info.jsonl", "w") as file:
        file.write_all(used_genre_info_dict.items())

    with open("lit_work_used_genres_info.json", "w") as file:
        json.dump(used_genre_info_dict, file)
    

    return
# TODO: continue to adapt! 

    # expand subclass hierarchy
    changed = True 
    while changed:
        prev_len = sum([len(subclass_values) for subclass_values in subclass_info_dict.values()])
        for genre_id, subclass_value_list in subclass_info_dict.items():
            new_super = []
            for class_value in subclass_value_list:
                new_super += subclass_info_dict.get(class_value, [])
            subclass_info_dict[genre_id] = list(set(subclass_value_list + new_super))
        if prev_len != sum([len(vlist) for vlist in subclass_info_dict.values()]):
            changed = True
        else:
            changed = False

    # not actually using genre counts for anything in this code
    genre_counts = defaultdict(lambda: 0)
    # keep track of what genres are actually used by any literary works
    valid_genres = []
    for genre_id, class_value in works.items():
        all_genres = []
        for vg in class_value:
            try:
                all_genres += subclass_info_dict[vg]
                all_genres.append(vg)
            except:
                pass
        all_genres = list(set(all_genres))
        valid_genres += all_genres
        for vg in all_genres:
            genre_counts[vg] += 1

    print(genre_counts)

    used_genre_info = {}
    used_genre_labels = {}
    valid_genres = set(valid_genres)
    for valid_genre in valid_genres:
        if valid_genre == 'Q483394': # "genre" entity
            continue
        genre_id = valid_genre
        class_value = subclass_info_dict[valid_genre]
        used_genre_info[genre_id] = class_value
        used_genre_labels[genre_id] = labels[genre_id]
        for vv in class_value:
            used_genre_labels[vv] = labels[vv]
    with open('cleaned_genre_subclass_info.json', 'w') as f:
        json.dump(used_genre_info, f)
    with open('cleaned_genre_labels.json', 'w') as f:
        json.dump(used_genre_labels, f)


def main():
    # query genres, save subclass/instance hierarchy and labels
    #genres = collect_genre_data()

    #collect_lit_works(list(genres))

    # re-process to do some cleanup
    clean_and_expand()

if __name__ == '__main__':
    main()