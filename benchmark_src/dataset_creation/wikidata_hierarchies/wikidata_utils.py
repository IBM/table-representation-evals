
import re
import requests
from time import time
from tqdm import tqdm
from collections import defaultdict


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