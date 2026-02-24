
import re
import requests
from time import time
from tqdm import tqdm
from collections import defaultdict
import json


IS_WIKIDATA_ID = r"Q\d+"
IS_WIKIDATA_PID = r"P\d+"

def is_wikidata_id(text):
    pattern = re.compile(IS_WIKIDATA_ID)
    return bool(pattern.search(text))

def is_wikidata_property_id(text):
    pattern = re.compile(IS_WIKIDATA_PID)
    return bool(pattern.search(text))

def construct_id_label_string(id, label):
    return str(label)  + "___" + str(id)

def deconstruct_label_id_string(label_id_string):
    """
    Returns the label and the id.

    Returns:
        string: label 
        string: id
    """
    splitted = label_id_string.split("___")
    assert len(splitted) == 2, f"Do not have exactly two values: {splitted}"
    label = splitted[0]
    id = splitted [1]
    return label, id


def get_subclasses(superclass):
    """
    Gets all subclasses of a given superclass from Wikidata.
    """
    
    # WIKIDATA, get all items that are "subclass" of items of a given class
    sparql_query = f"""
        SELECT ?s ?sLabel 
        WHERE {{
        ?s wdt:P279 wd:{superclass} .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """

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

    subclasses = []
    result = data['results']['bindings']
    #if len(result) == 0:
    ##    print("Empty result")
    #    print(sparql_query)
    for row in result:
        label = row["sLabel"]["value"]
        url = row["s"]["value"]
        qid = row["s"]["value"].split("/")[-1]
        if not is_wikidata_id(label):
            subclasses.append((qid, label))

    return subclasses

from collections import defaultdict
import requests
import time

def collect_items_of_subclass(entity_qids, url="https://query.wikidata.org/sparql"):
    """
    Collect all items whose P31 is one of the subclasses in entity_qids.
    If the query fails (JSON parsing error), retry each subclass individually.
    """
    def query_wikidata(subclass_list, limit=None):
        """Helper: send SPARQL query for a list of subclasses"""
        values_str = ' '.join(['wd:' + qid for qid in subclass_list])
        sparql_str = f"""
        SELECT DISTINCT ?s ?sLabel ?subclass
        WHERE {{
            ?s wdt:P31 ?subclass .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            VALUES ?subclass {{ {values_str} }}
        }}
        """
        if limit is not None:
            sparql_str += f" LIMIT {limit}"
        try:
            res = requests.get(
                url,
                params={'format': 'json', 'query': sparql_str},
                headers={
                    'User-Agent': 'CollectIdProperties/0.0',
                    'Accept': 'application/sparql-results+json'
                },
                timeout=300
            )
            res.raise_for_status()  # Raises for 4xx/5xx
            data = res.json()
            return data
        except (requests.RequestException, ValueError) as e:
            print(f"Query failed for subclasses {subclass_list}: {e}")
            if isinstance(e, ValueError):
                # Print a snippet of the response for debugging
                # print("Response snippet:", res.text[:1000])
                pass
            return None

    # First try querying all at once
    data = query_wikidata(entity_qids)

    # If it fails, try each subclass individually
    if data is None:
        print("Retrying each subclass individually...")
        items_by_subclass = defaultdict(list)
        items_labels = {}
        for qid in entity_qids:
            time.sleep(0.1)  # small delay to avoid hammering the server
            single_data = query_wikidata([qid], limit=20000)
            if single_data is None:
                print(f"Failed again for subclass {qid}, skipping.")
                continue  # skip this qid if it still fails
            for row in single_data['results']['bindings']:
                item_qid = row['s']['value'].split("/")[-1]
                subclass_qid = row['subclass']['value'].split("/")[-1]
                item_label = row['sLabel']['value']
                if not is_wikidata_id(item_label):
                    items_by_subclass[item_qid].append(subclass_qid)
                    items_labels[item_qid] = item_label
        print(f"Collected {len(items_labels)} items for {len(entity_qids)} subclasses.")
        return items_by_subclass, items_labels

    # If the big query worked, parse normally
    items_by_subclass = defaultdict(list)
    items_labels = {}
    for row in data['results']['bindings']:
        item_qid = row['s']['value'].split("/")[-1]
        subclass_qid = row['subclass']['value'].split("/")[-1]
        item_label = row['sLabel']['value']
        if not is_wikidata_id(item_label):
            items_by_subclass[item_qid].append(subclass_qid)
            items_labels[item_qid] = item_label

    print(f"Collected {len(items_labels)} items for {len(entity_qids)} subclasses.")
    return items_by_subclass, items_labels


def collect_all_instances(ent_list):
    wikidata_url = "https://query.wikidata.org/sparql"
    finished_entities = set(ent_list)
    n_batch = 25

    labels = {}
    all_output = {}
    while ent_list:
        print(f"Processing {len(ent_list)} entities, {n_batch} at a time")
        new_entries = 0
        new_entities = []
        for i in tqdm(range(0, len(ent_list), n_batch)):
            # force the query to wait a bit to avoid hitting query limits
            time.sleep(0.25)
            output, out_labels = collect_items_of_subclass(entity_qids=ent_list[i:i + n_batch], url=wikidata_url)
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

    # Save results
    #with open('items_by_subclass.json', 'w') as f:
    #    json.dump(all_output, f)

    items_per_subclass = {}
    for key, values in all_output.items():
        if len(values) == 1:
            if values[0] not in items_per_subclass:
                items_per_subclass[values[0]] = []
            items_per_subclass[values[0]].append(key)

    with open("items_per_subclass.json", "w") as file:
        json.dump(items_per_subclass, file, indent=2)

    with open('items_labels.json', 'w') as f:
        json.dump(labels, f)
    return all_output, labels
