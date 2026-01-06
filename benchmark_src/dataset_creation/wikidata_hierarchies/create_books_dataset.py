import json
import pandas as pd
from pathlib import Path
from hydra.utils import get_original_cwd

from benchmark_src.dataset_creation.wikidata_hierarchies import hierarchy_utils, wikidata_utils, create_testcases_more_similar_than

### TODO: clean up dataset creation for the whole books hierarchy and books table (and fully integrate in pipeline if cached files are not yet found!)


SAVE_DIR = Path(get_original_cwd()) / Path("cache/dataset_creation_resources/wikidata_books/wikidata_genres/")
RESOURCES_DIR = Path(get_original_cwd()) / Path("cache/dataset_creation_resources/resources")

creation_random = 48573948

PERCENTAGE_NULL_ACCEPTED = 0.95 # it's okay if x% of cells in the column are empty, remove cols with even higher sparsity

def check_authors_genres(book_table: pd.DataFrame):
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

def filter_table_columns(book_table: pd.DataFrame, filter_identifiers: bool=False):
    # filter properties (based on information from Sola)
    with open(RESOURCES_DIR / "wikimedia_related_properties.json", "r") as file:
        wikimedia_properties = json.load(file)
    if filter_identifiers:
        with open(RESOURCES_DIR / "unique_identifiers.json", "r") as file:
            identifier_properties = json.load(file)

    to_remove = []
    for property in book_table.columns:
        try:
            label, id = wikidata_utils.deconstruct_label_id_string(label_id_string=property)
        except:
            label, id = property, property
        if id in wikimedia_properties.keys():
            #print(f"Removing {id, label} column")
            to_remove.append(property)
        if filter_identifiers:
            if id in identifier_properties.keys():
                #print(f"Removing {id, label} column")
                to_remove.append(property)
            
    book_table.drop(to_remove, axis=1, inplace=True)
    with open(SAVE_DIR / "books_removed_cols.json", "w") as file:
        json.dump(to_remove, file, indent=2)

    print(f"Have {len(book_table.columns)} columns after removing wikimedia and id related properties")

    return book_table

def save_dataset_variation(dataset_save_dir: Path, books_table: pd.DataFrame, name_prefix: str, variation_name: str, testcases: list):
    dataset_name = name_prefix + variation_name
    dataset_save_dir = dataset_save_dir / dataset_name
    dataset_save_dir.mkdir(exist_ok=True)
    # save input_table
    books_table.to_csv(dataset_save_dir / 'input_table.csv', index=False)
    # save testcases
    create_testcases_more_similar_than.save_testcases(testcases, dataset_save_dir)
    create_statistics(dataset_name=dataset_name, 
                    input_table_df=books_table, 
                    testcases=testcases,
                    save_path=dataset_save_dir,
                    primary_key_column="QID"
                    )

def create_column_name_variations(initial_books_table, name_prefix, hierarchy, testcases):
    ################## original names ########################################
    variation_name = '_original_names'
    save_dataset_variation(initial_books_table, name_prefix, variation_name, testcases)
    
    ################## numbered column names ########################################
    variation_name = '_numbered_names'
    books_table = initial_books_table.copy()
    # number the columns names 
    new_col_names = ['QID'] + [f'column_{i}' for i in range(len(books_table.columns))[1:]]
    books_table.columns = new_col_names
    save_dataset_variation(books_table, name_prefix, variation_name, testcases)


### TODO: put under helpers to work for other datasets as well
def create_statistics(dataset_name, input_table_df, save_path, primary_key_column, testcases):
    statistics_dict = {
        "dataset_name": dataset_name,
        "input_table_num_rows": len(input_table_df),
        "input_table_num_cols": len(input_table_df.columns),
        "primary_key_column": primary_key_column,
        "datatypes": input_table_df.dtypes.astype("str").to_dict(),
    }

    num_empty_cells = float((input_table_df.isnull().sum()).sum())
    sparsity = float(num_empty_cells / input_table_df.size)
    statistics_dict["num_empty_cells"] = num_empty_cells
    statistics_dict["sparsity"] = sparsity

    if testcases is not None:
        statistics_dict["num_testcases"] = len(testcases)

        # Count easy and medium testcases
        num_easy = sum(1 for tc in testcases if tc.get("difficulty") == "easy")
        num_medium = sum(1 for tc in testcases if tc.get("difficulty") == "medium")
        statistics_dict["num_easy_testcases"] = num_easy
        statistics_dict["num_medium_testcases"] = num_medium

    with open(save_path / "dataset_information.json", "w") as file:
        json.dump(statistics_dict, file, indent=2)

def create_books_dataset(cfg, dataset_save_dir: Path):
    # load main books table
    book_table = pd.read_csv(SAVE_DIR / "books_table_more_similar_than.csv", low_memory=False)
    # load the created hierarchy from disk
    with open(SAVE_DIR / "hierarchy_for_more_similar_than.json", "r") as file:
        hierarchy = json.load(file)
    
    # re-order the first columns
    first_columns = ["QID", "label", "author___P50", "description___None"]
    new_column_order = first_columns + [col for col in book_table.columns if col not in first_columns]
    book_table = book_table[new_column_order]
    print(f"Initial table has {len(book_table.columns)} columns and {len(book_table)} rows")

    # delete rows of books where authors wrote fiction and non-fiction books
    book_table = check_authors_genres(book_table)

    # shuffle the table
    book_table = book_table.sample(frac=1, random_state=creation_random)

    # create testcases only once
    testcases = create_testcases_more_similar_than.create_testcases(
        book_table=book_table,
        hierarchy=hierarchy
    )
    book_table.to_csv(SAVE_DIR / "books_table_more_similar_than_prepared.csv", index=False)
    
    dataset_save_dir.mkdir(parents=True, exist_ok=True)
    
    # save the testcases
    create_testcases_more_similar_than.save_testcases(testcases, dataset_save_dir)

    # save dataset as "original" variation
    original_data_dir = dataset_save_dir / 'original'
    original_data_dir.mkdir(exist_ok=True)
    book_table.to_csv(original_data_dir / "input_table.csv", index=False)

    create_statistics(dataset_name='original', 
                input_table_df=book_table, 
                testcases=testcases,
                save_path=original_data_dir,
                primary_key_column="QID"
                )
    




if __name__ == "__main__":
    create_books_dataset()

    # #################### full table (without filtering columns) ############################################
    # # create variations of the table before filtering 
    # create_column_name_variations(initial_books_table=book_table, name_prefix=f'wikidata_books_{len(book_table.columns)}cols', hierarchy=hierarchy, testcases=testcases)

    # #################### full table (after filtering columns) ############################################
    # # filter table columns (TODO: later also create other versions)
    # book_table = filter_table_columns(book_table)
    # print(f"Have {len(book_table.columns)} columns after filtering")
    # # create variations of the table after filtering
    # create_column_name_variations(initial_books_table=book_table, name_prefix=f'wikidata_books_{len(book_table.columns)}cols', hierarchy=hierarchy, testcases=testcases)


    # #################### table with 29 columns ############################################
    # # prune columns
    # book_table = prune_sparse_columns(book_table, 0.95)
    # print(f"Have {len(book_table.columns)} columns after removing too sparse columns")

    # # create variations of the table with 29 columns
    # create_column_name_variations(initial_books_table=book_table, name_prefix=f'wikidata_books_{len(book_table.columns)}cols', hierarchy=hierarchy, testcases=testcases)

    # #################### table with just 9  (only 25% empty cells in a column) ############################################

    # # prune columns
    # smaller_book_table = prune_sparse_columns(book_table, 0.25)
    # print(f"Have {len(smaller_book_table.columns)} columns after removing too sparse columns")
    # print(smaller_book_table)
    # print(smaller_book_table.columns)

    # # create variations of the table with 9 columns
    # create_column_name_variations(initial_books_table=smaller_book_table, name_prefix=f'wikidata_books_{len(smaller_book_table.columns)}cols', hierarchy=hierarchy, testcases=testcases)
    
    # #################### table with just 3 columns ############################################

    # # prune columns
    # columns_to_keep = ['QID', 'label', 'author___P50', 'genre___P136']
    # smallest_book_table = smaller_book_table[columns_to_keep]
    # print(f"Have {len(smallest_book_table.columns)} columns after removing too sparse columns")
    # print(smallest_book_table)
    # print(smallest_book_table.columns)

    # # create variations of the table with 29 columns
    # create_column_name_variations(initial_books_table=smallest_book_table, name_prefix=f'wikidata_books_{len(smallest_book_table.columns)}cols', hierarchy=hierarchy, testcases=testcases)