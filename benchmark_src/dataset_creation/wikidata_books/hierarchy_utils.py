from benchmark_src.dataset_creation.wikidata_books import wikidata_utils

def get_leaves(hierarchy_to_search, original_hierarchy, leaves=[]):
    for genre, sub_hierarchy in hierarchy_to_search.items():
        # is leaf?
        if len(sub_hierarchy) == 0:
            leaves.append(genre)
        elif isinstance(sub_hierarchy, dict):
            get_leaves(sub_hierarchy, original_hierarchy, leaves)
    return leaves

def find_insertion_path(hierarchy, superclass, path=None):
    if path is None:
        path = []

    for key, value in hierarchy.items():
        if key == superclass:
            return path + [key]
        if isinstance(value, dict):
            result = find_insertion_path(value, superclass, path+[key])
            if result:
                return result
    return None

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

def find_longest_insertion_path(hierarchy, genre_superclasses):
    # find all currently possible insertion paths for this genre
    possible_insertion_paths = []
    for superclass in genre_superclasses:
        superclass = wikidata_utils.construct_id_label_string(superclass[1], superclass[0])
        insertion_path = find_insertion_path(hierarchy, superclass)
        if insertion_path is not None:
            possible_insertion_paths.append(insertion_path)
    # select the longest of the paths
    if len(possible_insertion_paths) > 0:
        longest_path = max(possible_insertion_paths, key=len)
    else:
        longest_path = None

    return longest_path

def find_current_path(hierarchy, target_key, current_path=[]):
    """
    Recursively finds the path (sequence of keys) to a target_key in a nested dictionary.

    Args:
        nested_dict (dict): The nested dictionary to search.
        target_key: The key to find the path for.
        current_path (list, optional): The path built so far (used in recursive calls). 
                                     Defaults to an empty list.

    Returns:
        list: A list of keys representing the path to the target_key, or None if not found.
    """
    for key, value in hierarchy.items():
        new_path = current_path + [key]  # Append current key to the path

        if key == target_key:
            return new_path  # Found the target key, return the path

        if isinstance(value, dict):
            # If the value is a dictionary, recursively search within it
            path_found = find_current_path(value, target_key, new_path)
            if path_found:
                return path_found  # If path found in the nested dict, return it

    return None  # Target key not found in this branch

def get_nested_value(data, path, default=None):
    """Safely gets a value from a nested dictionary using a list of keys."""
    current = data
    for key in path:
        if isinstance(current, dict):
            current = current.get(key, default)  # Use get() for safety
            if current is default and key != path[-1]: # If default is returned before the last key
                return default # The path is invalid, return default
        else:
            return default  # Path is invalid, return default
    return current

def delete_nested_item(dictionary, path):
    """
    Deletes an item from a nested dictionary at the specified path.

    Args:
        dictionary (dict): The nested dictionary.
        path (list): A list of keys representing the path to the item.
                     The last element in the list is the key to be deleted.

    Returns:
        bool: True if the item was successfully deleted, False otherwise.
    """
    if not path:  # Empty path, nothing to delete
        return False

    current_dict = dictionary
    # Traverse down the hierarchy using keys in the path (excluding the last key)
    for key in path[:-1]:
        if key in current_dict and isinstance(current_dict[key], dict):
            current_dict = current_dict[key]
        else:
            return False  # Invalid path, or item not found

    # Delete the item at the last key in the path
    last_key = path[-1]
    if last_key in current_dict:
        del current_dict[last_key]
        return True
    else:
        return False  # Item not found at the specified path
    

def find_siblings(nested_dict, target_key):
    """
    Finds the direct siblings of a given key in a nested dictionary.

    Args:
        nested_dict (dict): The nested dictionary representing the hierarchy.
        target_key: The key for which to find siblings.

    Returns:
        list or None: A list of sibling keys (excluding the target_key itself)
                      if the key is found and has siblings, otherwise None.
    """

    for key, value in nested_dict.items():
        if isinstance(value, dict):
            if target_key in value:
                # Target key found in the current dictionary (value).
                # The siblings are the other keys in this dictionary.
                siblings = [k for k in value.keys() if k != target_key]
                return siblings if siblings else None  # Return siblings if they exist, otherwise None
            else:
                # Recursively search in nested dictionaries.
                result = find_siblings(value, target_key)
                if result is not None:
                    return result  # Return result if siblings are found in nested dictionary
        elif isinstance(value, list): # Check if the value is a list
            for item in value: # Iterate over the items in the list
                if isinstance(item, dict): # If an item is a dictionary, recursively search it
                    result = find_siblings(item, target_key)
                    if result is not None:
                        return result # Return result if siblings are found in nested dictionary
    return None  # Key not found or no direct siblings
