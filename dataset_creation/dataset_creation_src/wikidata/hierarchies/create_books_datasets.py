import json
import pandas as pd
from pathlib import Path

import hierarchy_utils, wikidata_utils, create_testcases_more_similar_than

DATASET_DIR = Path("created_datasets/row_similarity_data_full/")
SAVE_DIR = Path("created_datasets/wikidata_genres/")
RESOURCES_DIR = Path("created_datasets/resources")

creation_random = 48573948

PERCENTAGE_NULL_ACCEPTED = 0.95 # it's okay if x% of cells in the column are empty, remove cols with even higher sparsity

def check_authors_genres(books_table: pd.DataFrame):
    # load hierarchy
    with open(SAVE_DIR / "hierarchy_for_more_similar_than.json", "r") as file:
        hierarchy = json.load(file)
    
    # build fiction and non-fiction key lists
    fiction_genres = hierarchy_utils.get_all_keys(hierarchy['genre___Q483394']['fiction___Q8253'])
    fiction_genres.append('fiction___Q8253')
    nonfiction_genres = hierarchy_utils.get_all_keys(hierarchy['genre___Q483394']["non-fiction___Q213051"])
    nonfiction_genres.append("non-fiction___Q213051")

    fiction_genres = [x.split("___")[0] for x in fiction_genres]
    nonfiction_genres = [x.split("___")[0] for x in nonfiction_genres]

    #print(fiction_genres)

    print(f'Found {len(fiction_genres)} fiction genres and {len(nonfiction_genres)} nonfiction genres')
    print(f'Have books table with {len(book_table)} rows')

    # find genre and author columns dynamically
    genre_col = next((col for col in book_table.columns if "genre" in col.lower()), None)
    author_col = next((col for col in book_table.columns if "author" in col.lower()), None)

    if genre_col is None or author_col is None:
        raise ValueError("Could not find genre or author column in input_table.csv")

    print(f"Using columns: author = '{author_col}', genre = '{genre_col}'")

    # classify each book as fiction / non-fiction / unknown
    def classify_genre(g):
        if str(g).lower() in fiction_genres:
            return "fiction"
        elif str(g).lower() in nonfiction_genres:
            return "non-fiction"
        else:
            print(f'{g} not found')
            return "unknown"

    book_table["__category"] = book_table[genre_col].map(classify_genre)

    # group by author and resolve conflicts
    cleaned_groups = []
    for author, group in book_table.groupby(author_col, dropna=False, group_keys=False):
        counts = group["__category"].value_counts()
        
        if "fiction" in counts and "non-fiction" in counts:
            # keep the majority category
            keep_cat = counts.idxmax()
            group = group[group["__category"] == keep_cat]
        
        cleaned_groups.append(group)

    cleaned_table = pd.concat(cleaned_groups, ignore_index=True)
    cleaned_table.drop('__category', axis=1, inplace=True)

    print(f'Cleaned table now has {len(cleaned_table)} rows (from {len(book_table)} originally)')

    return cleaned_table

def prune_sparse_columns(dataframe, null_threshold: float):
    """
    Prunes the dataframe in the following ways:
        - removes too sparse columns where more than null_threshold percent values are missing

        Args:
            dataframe (Dataframe): The dataframe to be filtered
            null_threshold (float): Percentage of null values in a column, columns with a higher percentage get deleted

        Returns:
            dataframe: The filtered dataframe
    """
    min_num_values = int(len(dataframe) * (1 - null_threshold))
    print(f"Columns must have at least {min_num_values} values to be kept")
    dataframe = dataframe.dropna(axis="columns", thresh=min_num_values)
    return dataframe

def filter_table_columns(book_table: pd.DataFrame):
    # filter properties (based on information from Sola)
    with open(RESOURCES_DIR / "wikimedia_related_properties.json", "r") as file:
        wikimedia_properties = json.load(file)
    with open(RESOURCES_DIR / "unique_identifiers.json", "r") as file:
        identifier_properties = json.load(file)

    to_remove = []
    for property in book_table.columns:
        try:
            label, id = wikidata_utils.deconstruct_label_id_string(label_id_string=property)
        except:
            label, id = property, property
        if id in wikimedia_properties.keys() or id in identifier_properties.keys():
            #print(f"Removing {id, label} column")
            to_remove.append(property)
    book_table.drop(to_remove, axis=1, inplace=True)
    with open(SAVE_DIR / "books_removed_cols.json", "w") as file:
        json.dump(to_remove, file, indent=2)

    print(f"Have {len(book_table.columns)} columns after removing wikimedia and id related properties")

    return book_table


def create_column_name_variations(initial_books_table, name_prefix, hierarchy):
    ################## combined names ########################################
    books_table = initial_books_table.copy()
    combined_names_folder = DATASET_DIR / (name_prefix + '_combined_names')
    combined_names_folder.mkdir(exist_ok=True)
    # save input_table
    books_table.to_csv(combined_names_folder / 'input_table.csv', index=False)
    # create testcases
    create_testcases_more_similar_than.create_testcases(book_table=book_table, hierarchy=hierarchy, dataset_save_dir=combined_names_folder)

    ################## simple names ########################################
    books_table = initial_books_table.copy()
    simple_names_folder = DATASET_DIR / (name_prefix + '_simple_names')
    simple_names_folder.mkdir(exist_ok=True)
    # simplify names 
    simple_col_names = [col.split('___')[0] for col in books_table.columns]
    books_table.columns = simple_col_names
    # save input_table
    books_table.to_csv(simple_names_folder / 'input_table.csv', index=False)
    # create testcases
    create_testcases_more_similar_than.create_testcases(book_table=book_table, hierarchy=hierarchy, dataset_save_dir=simple_names_folder)

    ################## complex names ########################################
    books_table = initial_books_table.copy()
    complex_names_folder = DATASET_DIR / (name_prefix + '_complex_names')
    complex_names_folder.mkdir(exist_ok=True)
    # complex names 
    complex_col_names = []
    for col in books_table.columns:
        split = col.split('___')
        if len(split) == 2:
            complex_col_names.append(split[1])
        else:
            complex_col_names.append(split[0])
    books_table.columns = complex_col_names

    # save input_table
    books_table.to_csv(complex_names_folder / 'input_table.csv', index=False)
    # create testcases
    create_testcases_more_similar_than.create_testcases(book_table=book_table, hierarchy=hierarchy, dataset_save_dir=complex_names_folder)




if __name__ == "__main__":
    #create_variations()

    # load main books table
    book_table = pd.read_csv(SAVE_DIR / "books_table_more_similar_than.csv", low_memory=False)
    
    # re-order the first columns
    first_columns = ["QID", "label", "author___P50", "description___None"]
    new_column_order = first_columns + [col for col in book_table.columns if col not in first_columns]
    book_table = book_table[new_column_order]
    print(f"Initial table has {len(book_table.columns)} columns and {len(book_table)} rows")

    # delete rows of books where authors wrote fiction and non-fiction books
    book_table = check_authors_genres(book_table)

    # filter table columns (TODO: later also create other versions)
    book_table = filter_table_columns(book_table)
    
    # prune columns
    book_table = prune_sparse_columns(book_table, 0.95)
    print(f"Have {len(book_table.columns)} columns after removing too sparse columns")

    # shuffle the table
    book_table = book_table.sample(frac=1, random_state=creation_random)
    # create pandas dataframe and save to disk
    book_table.to_csv(SAVE_DIR / "books_table_more_similar_than_cleaned.csv", index=False)

    #print(book_table)

    # load the created hierarchy from disk
    with open(SAVE_DIR / "hierarchy_for_more_similar_than.json", "r") as file:
        hierarchy = json.load(file)

    #################### table with 29 columns ############################################

    # create variations of the table with 29 columns
    create_column_name_variations(initial_books_table=book_table, name_prefix='wikidata_books', hierarchy=hierarchy)

    #################### table with just 10 columns ############################################

    # prune columns
    smaller_book_table = prune_sparse_columns(book_table, 0.25)
    print(f"Have {len(smaller_book_table.columns)} columns after removing too sparse columns")
    print(smaller_book_table)
    print(smaller_book_table.columns)

    # create variations of the table with 29 columns
    create_column_name_variations(initial_books_table=smaller_book_table, name_prefix='wikidata_books_9cols', hierarchy=hierarchy)