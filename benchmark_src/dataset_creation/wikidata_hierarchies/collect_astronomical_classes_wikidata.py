"""
build_and_filter_astronomical_hierarchy.py

Builds an astronomy-specific hierarchy and filters out observational/event-only classes.

Pipeline:

1. Recursively collect all subclasses of "astronomical object" (Q6999).
2. Build subclass → superclass DAG using BFS.
3. Collect all Wikidata items instantiating these classes.
4. Compute usage statistics.
5. Compute class types: physical, hybrid, observational, event, other.
6. Include all physical/hybrid classes and their superclasses.
7. Filter out observational/event-only classes.
8. Save filtered hierarchy and DAG.
"""

from benchmark_src.dataset_creation.wikidata_hierarchies import wikidata_utils
from tqdm import tqdm
import json
from collections import Counter
from functools import lru_cache

# ----------------------------
# Root sets for classification
# ----------------------------
PHYSICAL_ROOTS = {
    "Q523",      # star
    "Q318",      # galaxy
    "Q42372",    # nebula
    "Q168845",   # star cluster
    "Q1022867",  # minor planet
    "Q101600",   # brown dwarf
    "Q60186",    # meteorite
    "Q1054444",  # interstellar cloud
}

OBSERVATIONAL_ROOTS = {
    "Q1931185",   # astronomical radio source
    "Q67206691",  # infrared source
    "Q71962386",  # gamma-ray source
    "Q472483",    # X-ray source
}

EVENT_ROOTS = {
    "Q1656682",  # astronomical event
    "Q3937",     # supernova
    "Q22247",    # gamma-ray burst
}

ALLOWED_TYPES = {"physical", "hybrid"}  # only retain these


if __name__ == "__main__":

    use_cached = True

    if use_cached:
        
        have_cached = True
    


    ###########################################################################
    # Step 1: Collect subclasses and build DAG
    ###########################################################################
    initial_superclass = ("Q6999", "Astronomical Object")

    all_subclasses = set()
    to_process = [initial_superclass]

    # subclass_id -> [superclass_ids]
    class_parent_map = {}

    while to_process:
        new_classes = set()
        for superclass in tqdm(to_process):
            found_classes = wikidata_utils.get_subclasses(superclass[0])
            for subclass in found_classes:
                qid = subclass[0]

                if qid not in class_parent_map:
                    class_parent_map[qid] = []
                if superclass[0] not in class_parent_map[qid]:
                    class_parent_map[qid].append(superclass[0])

                # Queue for BFS
                if subclass not in all_subclasses:
                    new_classes.add(subclass)

        all_subclasses.update(new_classes)
        to_process = new_classes

    print(f"Found {len(all_subclasses)} subclasses of {initial_superclass[1]}")

    ###########################################################################
    # Step 2: Collect Wikidata items for subclasses
    ###########################################################################
    subclass_qids = [qid for qid, label in all_subclasses]
    print(f"{len(subclass_qids)} subclasses to process.")

    items, labels = wikidata_utils.collect_all_instances(subclass_qids)
    print(f"Collected {len(items)} items across {len(subclass_qids)} subclasses.")

    # save items and labels to disk, add parameter to script to re-use them if wanted instead of querying wikidata again (skip steps 1/2 upon request)


    ###########################################################################
    # Step 3: Identify used classes
    ###########################################################################
    all_classes_used_in_items = []
    for item_qid, subclass_list in items.items():
        item_name = labels[item_qid]
        if not wikidata_utils.is_wikidata_id(item_name):
            all_classes_used_in_items.extend(subclass_list)

    print(f"Total class usages in items: {len(all_classes_used_in_items)}")

    ###########################################################################
    # Step 4: Add missing superclasses
    ###########################################################################
    superclass_to_add = set()
    for qid in all_classes_used_in_items:
        for parent in class_parent_map.get(qid, []):
            if parent not in all_classes_used_in_items:
                superclass_to_add.add(parent)

    all_classes_used_in_items += list(superclass_to_add)
    print(f"{len(all_classes_used_in_items)} total subclasses after adding superclasses")

    ###########################################################################
    # Step 5: Build QID -> label lookup
    ###########################################################################
    subclass_id_to_label_lookup = {qid: label for qid, label in all_subclasses}
    subclass_id_to_label_lookup["Q6999"] = "astronomical object"

    ###########################################################################
    # Step 6: Compute class types for all classes
    ###########################################################################
    @lru_cache(maxsize=None)
    def get_all_ancestors(qid):
        ancestors = set()
        for parent in class_parent_map.get(qid, []):
            ancestors.add(parent)
            ancestors |= get_all_ancestors(parent)
        return ancestors

    all_qids = set(class_parent_map.keys()) | set(subclass_id_to_label_lookup.keys())
    class_types = {}
    for qid in all_qids:
        ancestors = get_all_ancestors(qid)
        has_physical = bool(ancestors & PHYSICAL_ROOTS)
        has_obs = bool(ancestors & OBSERVATIONAL_ROOTS)
        has_event = bool(ancestors & EVENT_ROOTS)

        if has_event:
            cls_type = "event"
        elif has_physical and has_obs:
            cls_type = "hybrid"
        elif has_physical:
            cls_type = "physical"
        elif has_obs:
            cls_type = "observational"
        else:
            cls_type = "other"
        class_types[qid] = cls_type

    ###########################################################################
    # Step 7: Build metadata and filter observational/event classes
    ###########################################################################
    filtered_classes_info = {}
    seen_labels = set()

    def filter_superclasses(superclasses):
        """Keep only physical/hybrid superclasses."""
        filtered = []
        for label, qid in superclasses:
            if class_types.get(qid) in ALLOWED_TYPES:
                filtered.append((label, qid))
        return filtered

    for qid in all_qids:
        if class_types[qid] in ALLOWED_TYPES:
            info = {
                "label": subclass_id_to_label_lookup.get(qid, qid),
                "QID": qid,
                "url": f"https://www.wikidata.org/wiki/{qid}",
                "used_count": all_classes_used_in_items.count(qid),
                "superclasses": filter_superclasses([(subclass_id_to_label_lookup.get(pid, pid), pid)
                                                     for pid in class_parent_map.get(qid, [])]),
                "class_type": class_types[qid]
            }
            if info["label"] not in seen_labels:
                filtered_classes_info[qid] = info
                seen_labels.add(info["label"])

    ###########################################################################
    # Step 8: Sort and save
    ###########################################################################
    filtered_classes_info = dict(sorted(filtered_classes_info.items(),
                                        key=lambda item: item[1]["used_count"],
                                        reverse=True))

    with open("astronomical_objects_filtered_classes_info.json", "w", encoding="utf-8") as f:
        json.dump(filtered_classes_info, f, indent=2, ensure_ascii=False)

    with open("astronomical_class_parent_map.json", "w", encoding="utf-8") as f:
        json.dump(class_parent_map, f, indent=2, ensure_ascii=False)

    print("Filtered hierarchy and DAG saved successfully.")
