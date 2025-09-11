import pandas as pd
from pathlib import Path
import json
import random 
import itertools

import hierarchy_utils

DATASET_DIR = Path("created_datasets/row_similarity_data_full/wikidata_books/")
SAVE_DIR = Path("created_datasets/wikidata_genres/")

random.seed(34234)

## TEMP, not here
def create_statistics(dataset_name, input_table_df, num_testcases, save_path, primary_key_column):
    statistics_dict = {"dataset_name": dataset_name,
                       "input_table_num_rows": len(input_table_df),
                       "input_table_num_cols": len(input_table_df.columns),
                       "primary_key_column": primary_key_column
                       }
    
    # get datatypes of columns
    statistics_dict["datatypes"] = input_table_df.dtypes.astype("str").to_dict()

    # compute table sparsity
    num_empty_cells = float((input_table_df.isnull().sum()).sum())
    sparsity = float(num_empty_cells / input_table_df.size)
    statistics_dict["num_empty_cells"] = num_empty_cells
    statistics_dict["sparsity"] = sparsity
    statistics_dict["num_testcases"] = num_testcases


    with open(save_path / "dataset_information.json", "w") as file:
        json.dump(statistics_dict, file, indent=2)


def select_book_from_opposite_highlevel_genre(all_genres_in_table, grouped_books, opposite_qids):
    """
    Select a fiction or non-fiction book 
    """
    
    # Select one random QID from the opposite high-level genre
    selected_qid_opposite_genre = random.choice(opposite_qids)
    # get a qid
    if selected_qid_opposite_genre in all_genres_in_table:
        try:
            books_in_opposite_genre = list(grouped_books.get_group(selected_qid_opposite_genre)["QID"])
            if books_in_opposite_genre:
                # 3. Select a random book (QID) from that genre's group
                selected_qid_opposite_book = random.choice(books_in_opposite_genre)
        except:
            selected_qid_opposite_book = None
    else:
        print(f"{selected_qid_opposite_genre} not in {sorted(all_genres_in_table)}")
        pass # for these no books were retrieved from wikidata
        selected_qid_opposite_book = None
        selected_qid_opposite_genre = None
            

    return selected_qid_opposite_book, selected_qid_opposite_genre


def create_easy_same_genre_testcases(group, genre, genre_col, group_ids, all_genres_in_table, grouped_books, opposite_qids):
    all_pairs_in_genre = list(itertools.combinations(group_ids, 2))

    testcases = []

    # most simple testcases: per row, pick another row from the same group (pop from list), then select one from opposite high-level genre
    weight = len(group_ids) // 2
    for _ in range(weight):
        if all_pairs_in_genre:
            # Select a random pair from the unique pairs
            selected_pair = random.choice(all_pairs_in_genre)

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
        selected_book = random.choice(group_ids)

        # select a sibling genre
        selected_sibling_genre = random.choice(genre_siblings)
        #print(f"Selected sibling: {selected_sibling_genre}")

        # Due to casing a few genres are no longer found (TODO: do not change casing when creating hierarchy)
        try:
            books_from_sibling_genre = list(grouped_books.get_group(selected_sibling_genre)["QID"])
        except:
            continue
        #print(f"Sibling books: {books_from_sibling_genre}")
        sibling_book = random.choice(books_from_sibling_genre)
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
    
def create_testcases(book_table: pd.DataFrame, hierarchy: dict, dataset_save_dir: Path):
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

    book_table = book_table[
    (book_table[author_col].notna()) & (book_table[author_col] != "") &
    (book_table[description_col].notna()) & (book_table[description_col] != "")
    ]

    grouped_books = book_table.groupby([genre_col])

    # filter all_fiction and all_nonfiction for the genres actually included in the books table
    all_genres_in_table = [x.lower() for x in grouped_books.groups]
    all_fiction = [x for x in all_fiction if x.lower() in all_genres_in_table]
    all_nonfiction = [x for x in all_nonfiction if x.lower() in all_genres_in_table]

    print(f"{len(grouped_books)} genres")
    testcases = []

    for genre, group in grouped_books:
        genre = genre[0]
        #print(f"Genre is {genre}")
        group_ids = list(group["QID"])

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
            if not len(genre_name_in_hierarchy) == 1:
                print(f"Got {genre_name_in_hierarchy} while looking for {genre}, skipping for now")
                continue
            genre_siblings = hierarchy_utils.find_siblings(nested_dict=hierarchy, target_key=genre_name_in_hierarchy[0])
            #print(f"Have the following siblings: {genre_siblings} to {genre} ")
            if genre_siblings:
                genre_siblings = [x.split("___")[0] for x in genre_siblings]
                testcases += create_medium_sibling_genre_testcases(group=group, genre=genre, genre_col=genre_col, group_ids=group_ids,  all_genres_in_table=all_genres_in_table, grouped_books=grouped_books, opposite_qids=opposite_qids, genre_siblings=genre_siblings)

    # Save the generated test cases
    (dataset_save_dir / "test_cases").mkdir(exist_ok=True)
    for idx,  testcase in enumerate(testcases):
       with open(dataset_save_dir / "test_cases" / f"{idx}.json", "w") as file:
           json.dump(testcase, file, indent=2)

    print(f"\nGenerated {len(testcases)} test cases.")

    book_table = pd.read_csv(dataset_save_dir / "input_table.csv", low_memory=False)
    create_statistics(dataset_name="wikidata_books", 
                      input_table_df=book_table, 
                      num_testcases=len(testcases),
                      save_path=dataset_save_dir,
                      primary_key_column="QID"
                      )


if __name__ == "__main__":
    book_table = pd.read_csv(DATASET_DIR / "input_table.csv", low_memory=False)
    #print(book_table)

    # load the created hierarchy from disk
    with open(SAVE_DIR / "hierarchy_for_more_similar_than.json", "r") as file:
        hierarchy = json.load(file)

    create_testcases(book_table, hierarchy, DATASET_DIR)


