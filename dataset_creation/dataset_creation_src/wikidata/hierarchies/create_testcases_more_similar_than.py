import pandas as pd
from pathlib import Path
import json
import random 
import itertools

import hierarchy_utils

DATASET_DIR = Path("created_datasets/row_similarity_data_full/wikidata_books/")
SAVE_DIR = Path("created_datasets/wikidata_genres/")

creation_random = random.Random(34234)



def select_book_from_opposite_highlevel_genre(all_genres_in_table, grouped_books, opposite_qids):
    """
    Select a fiction or non-fiction book from the opposite high-level genre.
    Always returns a valid (qid, genre) pair.
    """
    # Filter opposite_qids to those present in grouped_books
    valid_opposite_genres = [g for g in sorted(opposite_qids) if g in grouped_books.groups]
    if not valid_opposite_genres:
        print(f'No valid opposite genres..')
        raise NotImplementedError("No valid opposite genres found in the book table.")

    selected_qid_opposite_genre = creation_random.choice(valid_opposite_genres)
    books_in_opposite_genre = list(grouped_books.get_group(selected_qid_opposite_genre)["QID"])
    selected_qid_opposite_book = creation_random.choice(sorted(books_in_opposite_genre))

    return selected_qid_opposite_book, selected_qid_opposite_genre


def create_easy_same_genre_testcases(group, genre, genre_col, group_ids, all_genres_in_table, grouped_books, opposite_qids):
    all_pairs_in_genre = list(itertools.combinations(group_ids, 2))

    testcases = []

    # most simple testcases: per row, pick another row from the same group (pop from list), then select one from opposite high-level genre
    weight = len(group_ids) // 2
    for _ in range(weight):
        if all_pairs_in_genre:
            # Select a random pair from the unique pairs
            selected_pair = creation_random.choice(sorted(all_pairs_in_genre))

            selected_qid_opposite_book, selected_qid_opposite_genre = select_book_from_opposite_highlevel_genre(all_genres_in_table=all_genres_in_table, grouped_books=grouped_books, opposite_qids=opposite_qids)

            if selected_qid_opposite_book is not None:                         
                testcases.append({
                "similar_pair": { 
                    "a": {"qid": selected_pair[0], "genre": group[group["QID"]==selected_pair[0]][genre_col].item()},
                    "b": {"qid": selected_pair[1], "genre": group[group["QID"]==selected_pair[1]][genre_col].item()}
                },
                "dissimilar_pair": {
                    "a": {"qid": selected_pair[0], "genre": group[group["QID"]==selected_pair[0]][genre_col].item()},
                    "c": {"qid": selected_qid_opposite_book, "genre": selected_qid_opposite_genre.split("___")[0].lower()}
                },
                "difficulty": "easy",
                "category": "same-genre",
                })

            else:
                print(f"Warning: No opposite QIDs available for genre {genre}")
        
        else:
            print(f"Warning: No pairs available in genre {genre} to generate test cases.")

    return testcases


def create_medium_sibling_genre_testcases(group, genre, genre_col, group_ids, all_genres_in_table, grouped_books, opposite_qids, genre_siblings):
    testcases = []

    # most simple testcases: per row, pick another row from the same group (pop from list), then select one from opposite high-level genre
    weight = len(group_ids) // 2
    for _ in range(weight):
        
        # Select a random book id from the group
        selected_book = creation_random.choice(sorted(group_ids))

        # select a sibling genre
        selected_sibling_genre = creation_random.choice(sorted(genre_siblings))
        #print(f"Selected sibling: {selected_sibling_genre}")

        # Due to casing a few genres are no longer found (TODO: do not change casing when creating hierarchy)
        try:
            books_from_sibling_genre = list(grouped_books.get_group(selected_sibling_genre)["QID"])
        except:
            continue
        #print(f"Sibling books: {books_from_sibling_genre}")
        sibling_book = creation_random.choice(sorted(books_from_sibling_genre))
        #print(f"Selected sibling book: {sibling_book}")

        selected_qid_opposite_book, selected_qid_opposite_genre = select_book_from_opposite_highlevel_genre(all_genres_in_table=all_genres_in_table, grouped_books=grouped_books, opposite_qids=opposite_qids)

        if selected_qid_opposite_book is not None:                         
            testcases.append({
            "similar_pair": { 
                "a": {"qid": selected_book, "genre": group[group["QID"]==selected_book][genre_col].item()},
                "b": {"qid": sibling_book, "genre": selected_sibling_genre}
            },
            "dissimilar_pair": {
                "a": {"qid": selected_book, "genre": group[group["QID"]==selected_book][genre_col].item()},
                "c": {"qid": selected_qid_opposite_book, "genre": selected_qid_opposite_genre.split("___")[0].lower()}
            },
            "difficulty": "medium",
            "category": "sibling-genre",
            })

        else:
            print(f"Warning medium sibling testcases: No opposite QIDs available for genre {genre}")
        

    return testcases
    
def create_testcases(book_table: pd.DataFrame, hierarchy: dict):
    print("#######################################")
    print(f'Creating testcases for book_table {len(book_table)} rows')
    # get all genres
    all_genres = hierarchy_utils.get_all_keys(hierarchy)
    all_fiction = hierarchy_utils.get_all_keys(hierarchy['genre___Q483394']['fiction___Q8253'])
    all_nonfiction = hierarchy_utils.get_all_keys(hierarchy['genre___Q483394']["non-fiction___Q213051"])
    #print(all_fiction)
    all_fiction = [x.split("___")[0] for x in all_fiction]
    all_nonfiction = [x.split("___")[0] for x in all_nonfiction]

    print(f"Found {len(all_genres)} genres, {len(all_fiction)} fiction, {len(all_nonfiction)} non-fiction")

    # find genre and author columns dynamically
    genre_col = next((col for col in book_table.columns if "genre" in col.lower()), None)
    author_col = next((col for col in book_table.columns if "author" in col.lower()), None)
    description_col = next((col for col in book_table.columns if "description" in col.lower()), None)

    if genre_col is None or author_col is None or description_col is None:
        raise ValueError(f"Could not find description, genre or author column in input_table.csv, {book_table.columns}")

    # sort by qid to have a consistent order
    book_table = book_table.sort_values("QID").reset_index(drop=True)

    book_table = book_table[
    (book_table[author_col].notna()) & (book_table[author_col] != "") &
    (book_table[description_col].notna()) & (book_table[description_col] != "")
    ]

    # Build a deterministic genre-to-group mapping
    genre_to_group = {}
    unique_genres = sorted(set(book_table[genre_col].dropna().unique()), key=lambda x: str(x).lower())
    for genre in unique_genres:
        genre_to_group[genre] = book_table[book_table[genre_col] == genre].sort_values("QID").reset_index(drop=True)

    # Use sorted_genres for all further processing
    sorted_genres = unique_genres


    grouped_books = book_table.groupby([genre_col])
    #sorted_genres = sorted(grouped_books.groups.keys(), key=lambda x: str(x).lower())

    # filter all_fiction and all_nonfiction for the genres actually included in the books table
    all_genres_in_table = sorted([x.lower() for x in grouped_books.groups])
    all_fiction = sorted([x for x in all_fiction if x.lower() in all_genres_in_table])
    all_nonfiction = sorted([x for x in all_nonfiction if x.lower() in all_genres_in_table])

    print(f"{len(grouped_books)} genres")
    testcases = []

    for genre in sorted_genres:
        group = genre_to_group[genre]
        group_ids = sorted(list(group["QID"]))

        # Determine the opposite high-level genre
        current_genre_is_fiction = (genre.lower() in all_fiction)
        current_genre_is_nonfiction = (genre.lower() in all_nonfiction)
        if current_genre_is_fiction:
            opposite_qids = all_nonfiction
        elif current_genre_is_nonfiction:
            opposite_qids = all_fiction
        else:
            print(f"Did not find {genre} in fiction or nonfiction")

        if len(group) >= 2:
            ###############################################################################
            # Create easy testcases where the similar books are from the exact same genre
            ###############################################################################
            testcases += create_easy_same_genre_testcases(group=group, genre=genre, genre_col=genre_col, group_ids=group_ids, all_genres_in_table=all_genres_in_table, grouped_books=grouped_books, opposite_qids=opposite_qids)

            ###############################################################################
            # Create medium testcases where the similar books are from sibling genres
            ###############################################################################
            # see if the genre has any siblings (TODO: split by ___ search then in the genre names)
            genre_name_in_hierarchy = [x for x in all_genres if genre.lower()+"_" in x.lower()]

            if not genre_name_in_hierarchy:
                print(f"Got {genre_name_in_hierarchy} while looking for {genre}, skipping for now")
                continue

            # Use the shortest match (most specific), or just the first
            target_genre_for_siblings = min(genre_name_in_hierarchy, key=len)
            #print(f'using {target_genre_for_siblings} as target for siblings search for genre {genre}')
            genre_siblings = hierarchy_utils.find_siblings(nested_dict=hierarchy, target_key=target_genre_for_siblings)
            #print(f"Have the following siblings: {genre_siblings} to {genre} ")
            if genre_siblings:
                genre_siblings = [x.split("___")[0] for x in genre_siblings]
                testcases += create_medium_sibling_genre_testcases(group=group, genre=genre, genre_col=genre_col, group_ids=group_ids,  all_genres_in_table=all_genres_in_table, grouped_books=grouped_books, opposite_qids=opposite_qids, genre_siblings=genre_siblings)

    print(f"\nGenerated {len(testcases)} test cases.")

    return testcases

def save_testcases(testcases, dataset_save_dir):
    (dataset_save_dir / "test_cases").mkdir(exist_ok=True)
    for idx,  testcase in enumerate(testcases):
        with open(dataset_save_dir / "test_cases" / f"{idx}.json", "w") as file:
            json.dump(testcase, file, indent=2)

if __name__ == "__main__":
    book_table = pd.read_csv(DATASET_DIR / "input_table.csv", low_memory=False)
    #print(book_table)

    # load the created hierarchy from disk
    with open(SAVE_DIR / "hierarchy_for_more_similar_than.json", "r") as file:
        hierarchy = json.load(file)

    testcases = create_testcases(book_table, hierarchy, DATASET_DIR)

    # Save the generated test cases
    save_testcases(testcases, DATASET_DIR)

