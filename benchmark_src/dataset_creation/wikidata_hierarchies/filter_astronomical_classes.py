"""
filter_astronomical_classes.py

Filter observational and event-only classes from a precomputed
astronomical class hierarchy grounded in Wikidata instances.

This script:

1. Reads a hierarchy of astronomical classes that were used
   by real Wikidata instances (from `astronomical_objects_instances_used_classes_info.json`)
   and the full subclass hierarchy (`subclass_hierarchy.json`).
2. Computes the full ancestor closure for each class.
3. Classifies each class as:
   - physical
   - hybrid (both physical + observational ancestry)
   - observational
   - event
   - other
4. Retains only physical + hybrid classes.
5. Updates superclass references to remove filtered parents while keeping DAG connectivity.
6. Saves:
   - filtered hierarchy for downstream similarity benchmarks
   - rejected classes for debugging purposes
"""

import json
from functools import lru_cache
from collections import Counter

# ----------------------------
# Anchors for classification
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

ALLOWED_TYPES = {"physical", "hybrid"}


def main():
    """
    Main function to filter astronomical classes.
    """

    # ----------------------------
    # Step 0: Load existing data
    # ----------------------------
    with open("astronomical_objects_instances_used_classes_info.json", "r", encoding="utf-8") as f:
        used_classes_info_dict = json.load(f)

    with open("astronomical_class_parent_map.json", "r", encoding="utf-8") as f:
        subclass_hierarchy = json.load(f)

    # ----------------------------
    # Step 1: Compute transitive ancestors
    # ----------------------------
    @lru_cache(None)
    def get_all_ancestors(qid):
        """
        Recursively return all superclasses of a given class.
        """
        ancestors = set()
        for parent in subclass_hierarchy.get(qid, []):
            ancestors.add(parent)
            ancestors |= get_all_ancestors(parent)
        return ancestors

    # ----------------------------
    # Step 2: Compute class types for all classes
    # ----------------------------
    all_qids = set(subclass_hierarchy.keys()) | set(used_classes_info_dict.keys())
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

    print("Class type distribution:", Counter(class_types.values()))

    # ----------------------------
    # Step 3: Attach class_type to used classes
    # ----------------------------
    for qid, info in used_classes_info_dict.items():
        info["class_type"] = class_types.get(qid, "unknown")

    # ----------------------------
    # Step 4: Filter by allowed types
    # ----------------------------
    filtered_classes_info = {
        qid: info
        for qid, info in used_classes_info_dict.items()
        if info["class_type"] in ALLOWED_TYPES
    }

    print(f"Kept {len(filtered_classes_info)} classes out of {len(used_classes_info_dict)}")

    # ----------------------------
    # Step 5: Filter superclasses while preserving DAG
    # ----------------------------
    def filter_superclasses(superclasses):
        """
        Keep only parent classes that are physical or hybrid.
        Even if they have no instances, retain them in the hierarchy.
        """
        filtered = []
        for label, qid in superclasses:
            if class_types.get(qid) in ALLOWED_TYPES:
                filtered.append((label, qid))
        return filtered

    for qid, info in filtered_classes_info.items():
        info["superclasses"] = filter_superclasses(info.get("superclasses", []))

    # ----------------------------
    # Step 6: Save outputs
    # ----------------------------
    with open("astronomical_objects_filtered_physical_hierarchy.json", "w", encoding="utf-8") as f:
        json.dump(filtered_classes_info, f, indent=2, ensure_ascii=False)

    kept_labels = [y["label"] for x, y in filtered_classes_info.items()]

    print(kept_labels)

    # Save rejected classes for debugging
    rejected_classes = {
        qid: info
        for qid, info in used_classes_info_dict.items()
        if qid not in filtered_classes_info
    }
    
    print("################# removed ###################")

    removed_labels = [y["label"] for x, y in rejected_classes.items()]

    print(removed_labels)


    with open("astronomical_objects_rejected_classes.json", "w", encoding="utf-8") as f:
        json.dump(rejected_classes, f, indent=2, ensure_ascii=False)

    print("Filtered hierarchy saved as 'astronomical_objects_filtered_physical_hierarchy.json'")
    print("Rejected classes saved as 'astronomical_objects_rejected_classes.json'")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
