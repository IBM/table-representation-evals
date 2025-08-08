import json
from collections import Counter
import jsonlines
from pathlib import Path

import hierarchy_utils, visualize, wikidata_utils

REPLACEMENTS = {
    # from: to
    ('art genre', 'Q1792379'): ('genre', 'Q483394'),
    ('literary genre','Q223393'): ('genre', 'Q483394'),
    #('literary work', 'Q7725634'): ('genre', 'Q483394'),
    ("non-fiction literature", "Q27801"):("non-fiction", "Q213051"),
    ("fiction literature", "Q38072107"): ("fiction", "Q8253"),
    # TODO: speculative fiction, speculative fiction literature
    #('creative work', 'Q17537576'): ('genre', 'Q483394'),
    # TODO: check which genres get included due to replacements
}   

# TODO: check for classes that are directly instances of "literary genre"

SAVE_DIR = Path("data/wikidata_genres/")
MIN_USED_COUNT = 25

def get_genres_that_appear_as_superclasses(genre_info_dict):
    """
    Collect all genres that appear as superclasses of another genre
    """
    genres_that_appear_as_superclasses = set()
    for genre_id, genre_info in genre_info_dict.items():
        class_list = genre_info["superclasses"]
        for class_item in class_list:
            genres_that_appear_as_superclasses.add(class_item[1])

    return genres_that_appear_as_superclasses

def prune_genres(genre_info_dict, min_used_count=5):
    """
    Will remove all leaf classes that are used by literary work items as genre less than min_used_count times.
    A leaf class is never used as a superclass by any other genre.
    """
    print(f"############# Start pruning #################")
    print(f"Have {len(genre_info_dict)} genres")
    # Collect all genres that appear as superclasses of another genre
    initial_superclasses = get_genres_that_appear_as_superclasses(genre_info_dict)
    print(f"Found {len(initial_superclasses)} genres as superclasses, not all of them might be in this 'used genres' list")

    # Collect all leaf classes (keys that never appear in any superclass list)
    leaf_classes = [class_id for class_id in genre_info_dict.keys() if class_id not in initial_superclasses]
    print(f"Found {len(leaf_classes)} initial leaf classes")

    # Remove leaf classes that were used as genres in literary works less than min_used_count times
    leaf_classes_to_remove = []
    leaf_classes_keep = []
    for class_id in leaf_classes:
        class_info = genre_info_dict[class_id]
        if class_info ["used_count"] < min_used_count:
            leaf_classes_to_remove.append(class_id)
        else:
            leaf_classes_keep.append(class_id)
    
    print(f"Removing {len(leaf_classes_to_remove)} leaf classes that are less than {min_used_count} used")
    print(f"Keeping {len(leaf_classes_keep)} of the original leaves")

    for class_id in leaf_classes_to_remove:
        genre_info_dict.pop(class_id)

    print(f"{len(genre_info_dict)} total genres left after cleaning")

    superclasses = get_genres_that_appear_as_superclasses(genre_info_dict)
    leaf_classes = [class_id for class_id in genre_info_dict.keys() if class_id not in superclasses]

    print(f"Now have {len(leaf_classes)} leaf classes")
    # list which genres in the subclass list do not appear as keys
    genres_that_appear_as_superclasses = set()
    genres_not_as_keys = []
    genres_have_keys = []
    for genre_id, genre_info in genre_info_dict.items():
        class_list = genre_info["superclasses"]
        for class_item in class_list:
            if not class_item[1] in genre_info_dict.keys():
                genres_not_as_keys.append(tuple(class_item))
            else:
                genres_have_keys.append(tuple(class_item))    
            genres_that_appear_as_superclasses.add(class_item[1])

    #print(Counter(genres_not_as_keys))
    print(f"{len(Counter(genres_not_as_keys))} genres are not keys (they are never used as 'genre' property)")
    print(f"{len(Counter(genres_have_keys))} genres are keys (they are used as 'genre' property)")

    # assert that all classes are now either leaves or superclasses
    classes_that_are_neither_leaf_nor_superclass = []
    for class_id, genre_info in genre_info_dict.items():
        if not class_id in genres_that_appear_as_superclasses and not class_id in leaf_classes:
            classes_that_are_neither_leaf_nor_superclass.append(class_id)

    assert len(classes_that_are_neither_leaf_nor_superclass) == 0
    return genre_info_dict

def insert_value_at_path(data, path, insert_key, insert_given_value=None):
    """
    Inserts a value at the specified path in the nested dictionary.

    Args:
        data: The nested dictionary to modify.
        path: The list of keys representing the path.
        value: The value to insert.
    """
    if not path:
        return  # Empty path, nothing to insert

    current_level = data
    for i in range(len(path) - 1):  # Traverse to the second to last key in the path
        key = path[i]
        if not isinstance(current_level, dict):
            raise TypeError("Invalid path or data structure")  # Ensure we are traversing a dictionary
        current_level = current_level.setdefault(key, {})  # Create nested dictionary if needed

    # Insert the value at the last key in the path
    last_key = path[-1]
    if not insert_given_value:
        insert_val = {}
    else:
        insert_val = insert_given_value
    if isinstance(current_level, dict):
        current_level[last_key][insert_key] = insert_val  # Direct assignment to insert/update the value
    else:
        raise TypeError("Invalid path or data structure")  # Ensure we are at a dictionary level

def insert_into_hierarchy(genre_info_dict, classes_left_to_insert, hierarchy, replacements):
    # 
    to_remove = []
    for class_id, genre_info in genre_info_dict.items():
        superclasses = genre_info["superclasses"]
        label = genre_info["label"]
        
        id_label_string = wikidata_utils.construct_id_label_string(class_id, label)

        # do not insert classes that should be replaced
        if tuple([label, class_id]) in replacements.keys():
            #print("Should not insert", class_id, label)
            to_remove.append(id_label_string)
            continue

        # check if there is at least one insertion path, take longest if there are multiple
        if id_label_string in classes_left_to_insert:
            # find longest path
            longest_path = hierarchy_utils.find_longest_insertion_path(hierarchy=hierarchy, genre_superclasses=superclasses)
            if longest_path:
                #print(f"can insert {genre_info["label"]} at {insertion_path} (other superclasses are: {superclasses})")
                insert_value_at_path(data=hierarchy, path=longest_path, insert_key=id_label_string)
                to_remove.append(id_label_string)

    #print(f"Plan to remove: {to_remove}")
    for x in to_remove:  
        try:          
            classes_left_to_insert.remove(x)
        except:
            #print(f"Could not find {x}")
            pass

    return classes_left_to_insert, hierarchy

def traverse_hierarchy(hierarchy_to_search, original_hierarchy, genre_info_dict):
    for genre, superclass_list in hierarchy_to_search.items():
        # compare longest path with current path
        genre_label, genre_id = hierarchy_utils.deconstruct_label_id_string(genre)
        try:
            genre_info = genre_info_dict[genre_id]
            longest_path = hierarchy_utils.find_longest_insertion_path(hierarchy=original_hierarchy, genre_superclasses=genre_info["superclasses"])
            current_path = hierarchy_utils.find_current_path(hierarchy=original_hierarchy, target_key=genre)

            if not longest_path:
                if genre_info["label"] in ["genre", "fiction", "non-fiction"]:
                    longest_path = []
                else:
                    print(f"No longest path to find {genre_info}")

            if not genre in longest_path:
                if len(longest_path) > (len(current_path) - 1): 
                    return (longest_path, current_path)
                else:
                    pass
                    # TODO: revisit
                    #print("---")
                    #print(f"{genre_id, genre_label} is fine, superclasses: {genre_info["superclasses"]}")
                    #print(longest_path)
                    #print(current_path)
            else:
                pass
                #print(f"Circular dependency of {genre}, ignoring: {longest_path}, currently: {current_path}")
        except KeyError:
            #print(f"{genre_label, genre_id} is not used, no further information on superclasses")
            pass
        if isinstance(superclass_list, dict):
            result = traverse_hierarchy(superclass_list, original_hierarchy, genre_info_dict)
            if result is not None:
                return result

def clean_hierarchy(hierarchy, genre_info_dict):
    """ 
    some items are now inserted at higher levels than they should be

    first find out for which that is the case (try to find a path for every superclass, take the longest path)

    """
    done = False
    updated_paths = 0
    while not done:
        traversion_result = traverse_hierarchy(hierarchy, hierarchy, genre_info_dict)

        if traversion_result:
            longest_path, current_path = traversion_result
            # get everything below the current path (might not be a leaf)
            current_values = hierarchy_utils.get_nested_value(hierarchy, current_path)
            
            #print("----")
            #print("Currently at:", current_path)
            #print("Move to:", longest_path)
            #print("Values to move", current_values)

            # delete from curent path
            hierarchy_utils.delete_nested_item(hierarchy, current_path)

            # insert everything at longest path
            insert_value_at_path(data=hierarchy, path=longest_path, insert_key=current_path[-1], insert_given_value=current_values)
            updated_paths += 1
        else:
            done = True

    print(f"Updated the path of {updated_paths} genres")

def find_leaves_with_too_low_usage(hierarchy_to_search, original_hierarchy, genre_info_dict, leaves=[]):
    for genre, sub_hierarchy in hierarchy_to_search.items():
        genre_label, genre_id = hierarchy_utils.deconstruct_label_id_string(genre)

        # is leaf?
        if len(sub_hierarchy) == 0:
            # check num_used_count
            if genre_info_dict[genre_id]["used_count"] < MIN_USED_COUNT:
                # add to leaves list
                print("Pls remove", genre_id, genre_label)
                leaves.append(genre)
            
        elif isinstance(sub_hierarchy, dict):
            find_leaves_with_too_low_usage(sub_hierarchy, original_hierarchy, genre_info_dict, leaves)
    return leaves


def prune_leaves_in_hierarchy(hierarchy, genre_info_dict):
    done = False
    removed_leaves = 0
    while not done:
        print(f"########## Hierarchy len at beginning of iteration is:", len(get_all_keys(hierarchy)))
        # get a path of a leaf that should be removed
        leaves_to_remove = find_leaves_with_too_low_usage(hierarchy, hierarchy, genre_info_dict, leaves=[])
        print(f"Want to remove: {leaves_to_remove}")

        if len(leaves_to_remove) > 0:
            for leaf in leaves_to_remove:
                print(f"Now plan to remove {leaf}")
                # delete from curent path
                leaf_path = hierarchy_utils.find_current_path(hierarchy=hierarchy, target_key=leaf)
                print("Path to delete:", leaf_path)
                hierarchy_utils.delete_nested_item(hierarchy, leaf_path)
                print(f"Hierarchy len is:", len(get_all_keys(hierarchy)))
                removed_leaves += 1
                leaves_to_remove = []
        else:
            done = True

def get_all_keys(dictionary):
    """
    Recursively extracts all keys from a nested dictionary.

    Args:
        dictionary (dict): The dictionary to traverse.

    Returns:
        list: A list of all keys found in the dictionary.
    """
    keys_list = []  # Initialize an empty list to store the keys
    for key, value in dictionary.items():  # Iterate through the dictionary's items
        keys_list.append(key)  # Add the current key to the list
        if isinstance(value, dict):  # If the value is a dictionary
            keys_list.extend(get_all_keys(value))  # Recursively call the function and extend the list
    return keys_list  # Return the list of keys

if __name__ == "__main__":
    print(Path.cwd())
    print(SAVE_DIR.exists())

    pruning = True
    min_used_count = MIN_USED_COUNT
    hierarchy = {"genre___Q483394": {"fiction___Q8253": {}, "non-fiction___Q213051":{}},}

    with open(SAVE_DIR / "lit_work_used_genres_info.json", "r") as file:
        genre_info_dict = json.load(file)

    if pruning:
        converted = False
        prev_len = len(genre_info_dict)
        while not converted:
            genre_info_dict = prune_genres(genre_info_dict=genre_info_dict, min_used_count=min_used_count)
            if len(genre_info_dict) == prev_len:
                converted = True
            else:
                prev_len = len(genre_info_dict)       

    with jsonlines.open(SAVE_DIR / "classes_after_pruning.jsonl", "w") as file:
        file.write_all(genre_info_dict.items())

    # find candidates for replacements
    all_genre_labels = []
    all_genre_ids = []
    for genre_id, genre_inf in genre_info_dict.items():
        all_genre_ids.append(genre_id)
        all_genre_labels.append(genre_inf["label"])
    
    to_replace = []
    for term in ["literature", "fiction", "novel"]:
        for idx, x in enumerate(all_genre_labels):
            x_stripped = x.replace(term, "").strip()
            if x != "" and x != x_stripped:
               if x_stripped in all_genre_labels or x_stripped.lower() in all_genre_labels:
                    print(f"Replace {x} by {x_stripped}")
                    replace_id = [id for id, i in genre_info_dict.items() if i["label"] in [x_stripped, x_stripped.lower]]
                    assert len(replace_id) == 1
                    REPLACEMENTS[(x, all_genre_ids[idx])] = (x_stripped, replace_id[0])

    for idx, x in enumerate(all_genre_labels):
        if x != "literature":
            x_stripped = x.replace("literature", "fiction").strip()
            if x != x_stripped:
                if x_stripped in all_genre_labels or x_stripped.lower() in all_genre_labels:
                    print(f"Replace {x} by {x_stripped}")
                    replace_id = [id for id, i in genre_info_dict.items() if i["label"] in [x_stripped, x_stripped.lower]]
                    assert len(replace_id) == 1
                    REPLACEMENTS[(x, all_genre_ids[idx])] = (x_stripped, replace_id[0])



    print("Doing replacements")
    for replace, replace_with in REPLACEMENTS.items():
        try:
            num_count_replace = genre_info_dict[replace[1]]["used_count"]
            genre_info_dict[replace_with[1]]["used_count"] += num_count_replace
        except KeyError:
            print(repr(replace[1]), replace, "not found")

    for genre_id, genre_inf in genre_info_dict.items():
        superclasses_replaced = []
        original_superclasses = genre_inf["superclasses"]
        for superclass in original_superclasses:
            superclass = tuple(superclass)
            if superclass in REPLACEMENTS.keys():
                superclasses_replaced.append(REPLACEMENTS[superclass])
            else:
                superclasses_replaced.append(superclass)
        genre_info_dict[genre_id]["superclasses"] = superclasses_replaced

    print("*"*100 )
    print("Start hierarchy")

    # Sort genre_info_dict by superclass
    genre_info_dict_sorted = dict(sorted(genre_info_dict.items(), key=lambda item: len(item[1]["superclasses"])))

    print(REPLACEMENTS)
    
    classes_left_to_insert = [wikidata_utils.construct_id_label_string(id, info["label"]) for id, info in genre_info_dict.items()]

    print(f"Initially plan to instert {len(classes_left_to_insert)} classes into the hierarchy")

    changed = True
    prev_num_classes = len(classes_left_to_insert)
    while changed:
        classes_left_to_insert, hierarchy = insert_into_hierarchy(genre_info_dict_sorted, classes_left_to_insert, hierarchy, REPLACEMENTS)
        print(f"Have {len(classes_left_to_insert)} classes left to insert")
        if len(classes_left_to_insert) == prev_num_classes:
            changed = False
        else:
            prev_num_classes = len(classes_left_to_insert)

    genres_before_cleaning = get_all_keys(hierarchy)
    print(f"Have {len(genres_before_cleaning)} genres in hierarchy")

    with open(SAVE_DIR / "temp_uncleaned_hierarchy.json", "w") as file:
        json.dump(hierarchy, file, indent=1)

    print(f"Now re-ordering the hierarchy")
    clean_hierarchy(hierarchy=hierarchy, genre_info_dict=genre_info_dict)
    genres_after_reordering = get_all_keys(hierarchy)
    print(f"Have {len(genres_after_reordering)} genres in hierarchy after re-ordering")
    
    if not set(genres_after_reordering) == set(genres_before_cleaning):
        print("lost genres:")
        print(set(genres_before_cleaning).difference(set(genres_after_reordering)))

    print(f"Need to prune again if leaf classes are used less than {min_used_count} times")
    prune_leaves_in_hierarchy(hierarchy=hierarchy, genre_info_dict=genre_info_dict)

    genres_after_pruning = get_all_keys(hierarchy)    
    print(f"Have {len(genres_after_pruning)} genres in hierarchy after pruning leaves")

    with open(SAVE_DIR / "hierarchy_info.json", "w") as file:
        hierarchy_info = {"num_included": len(genres_after_reordering),
                          "num_not_included": len(classes_left_to_insert),
                          "included": genres_after_reordering,
                          "not_included": classes_left_to_insert}
        
        json.dump(hierarchy_info, file, indent=1)

    with open(SAVE_DIR / "cleaned_hierarchy.json", "w") as file:
        json.dump(hierarchy, file, indent=1)

    visualize.visualize_hierarchy(hierarchy)