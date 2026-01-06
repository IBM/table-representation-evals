import requests
from benchmark_src.dataset_creation.wikidata_hierarchies import wikidata_utils
from tqdm import tqdm
import json


if __name__ == "__main__":

    #################################################################################
    # First get all subclasses of "astronomical object" from Wikidata recursively
    #################################################################################
    initial_superclass = ("Q6999", "Astronomical Object")

    # BFS to get all subclasses of the initial superclass
    all_subclasses = set()
    to_process = [initial_superclass]

    while len(to_process) > 0:
        new_classes = set()
        for x in tqdm(to_process):
            found_classes = wikidata_utils.get_subclasses(x[0])
            for y in found_classes:
                if y not in all_subclasses:
                    new_classes.add(y)
        print(f"Found {len(new_classes)} new subclasses: {new_classes}")
        all_subclasses.update(new_classes)
        to_process = new_classes
    print(f"Found {len(all_subclasses)} subclasses of {initial_superclass}")


    #################################################################################
    # Next, get all instance of items for each of the subclasses
    #################################################################################
