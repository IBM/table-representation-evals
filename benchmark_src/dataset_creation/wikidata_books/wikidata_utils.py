
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