import pandas as pd
import json
import os
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder
from omegaconf import OmegaConf
from pathlib import Path

from benchmark_src.dataset_creation.wikidata_hierarchies.create_books_dataset import create_statistics

def is_numeric_like(series: pd.Series) -> bool:
    """
    Returns True if a Series is numeric-like: more digits/special chars
    than letters in its first meaningful value.
    Missing values are ignored.
    """
    # Skip non-object dtype numeric columns
    if pd.api.types.is_numeric_dtype(series):
        return True

    if series.dtype != object:
        return False

    for v in series:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s:
            continue

        # Count letters vs digits/special characters
        letters = sum(c.isalpha() for c in s)
        digits_special = sum(not c.isalpha() and not c.isspace() for c in s)

        return digits_special >= letters

    return False

def compute_missing_rates(df: pd.DataFrame):
    return df.isna().mean()


def transform_non_numeric_to_numeric(df: pd.DataFrame, cols: List[str]
                                     ) -> Tuple[pd.DataFrame, List[str]]:
    """Transform selected non-numeric columns into numeric using label encoding."""
    transformed_cols = []
    for col in cols:
        if df[col].dtype == "object":
            df[col + "_encoded"] = LabelEncoder().fit_transform(df[col].astype(str).fillna("missing"))
            transformed_cols.append(col + "_encoded")
    return df, transformed_cols

# TODO: rework to support numbered naming, need to keep primary key column!
def rename_columns(df: pd.DataFrame, naming_strategy: str):
    if naming_strategy == "original":
        return df
    elif naming_strategy == "numbered":
        renamed = {}
        for i, col in enumerate(df.columns):
            renamed[col] = f"col_{i}"
        df = df.rename(columns=renamed)
        return df
    else: 
        raise NotImplementedError(f"Column naming strategy {naming_strategy} not implemented.")

def create_dataset_variations(variation_parameters: dict, dataset_df: pd.DataFrame, dataset_name: str, save_dir, variation_name):

    # extract parameters and set defaults
    number_of_cols = variation_parameters.get("number_of_cols", len(dataset_df.columns))
    number_of_cols = int(number_of_cols)
    percentage_numerical = variation_parameters.get("percentage_numerical", None)
    percentage_numerical = float(percentage_numerical) if percentage_numerical is not None else None
    print(f"percentage numerical is: {percentage_numerical}")
    column_naming = variation_parameters.get("column_naming", "original")

    # check that there are no additional parameters, otherwise throw not implemented error
    for param in variation_parameters.keys():
        if param not in ["number_of_cols", "percentage_numerical", "column_naming"]:
            raise NotImplementedError(f"Variation parameter {param} not implemented.")

    # load dataset specific config
    dataset_config_path = Path("./benchmark_src/config/dataset") / f"{dataset_name}.yaml"
    if not dataset_config_path.exists():
        print(f"Could not find dataset config path: {dataset_config_path}")
    try:
        dataset_cfg = OmegaConf.load(str(dataset_config_path))
    except Exception as e:
        print(f"Could not load dataset config: {e}")
        

    # check if there are columns to always keep, they need to be in the final df
    keep_columns = dataset_cfg.get("columns_to_always_keep", [])


    if percentage_numerical is not None:
        if not (0 <= percentage_numerical <= 1):
            raise ValueError("perc_numerical must be between 0 and 1")
        if not (1 <= number_of_cols <= dataset_df.shape[1]):
            raise ValueError(f"total_cols must be between 1 and number of columns in df ({dataset_df.shape[1]})")

        # 1) detect numeric-like columns
        numeric_like = set(dataset_df.select_dtypes(include="number").columns.tolist())
        print(f"Numeric like cols are {len(numeric_like)}")

        # include Wikidata-style date strings as numeric-like (without converting)
        for col in dataset_df.columns:
            if col not in numeric_like and is_numeric_like(dataset_df[col]):
                numeric_like.add(col)

        numeric_like = list(numeric_like)

        # 2) compute how many numeric columns to pick (floor behaviour)
        num_numeric_wanted = int(number_of_cols * percentage_numerical)
        # ensure we don't ask for more numeric columns than exist
        num_numeric = min(num_numeric_wanted, len(numeric_like))

        # 3) choose numeric columns by fewest missing values (if any to pick)
        chosen_numeric_cols = []
        if num_numeric > 0:
            missing_rates = dataset_df[numeric_like].isna().mean()
            numeric_sorted = missing_rates.sort_values().index.tolist()
            chosen_numeric_cols = numeric_sorted[:num_numeric]

        print(f"Chosen numeric cols: {chosen_numeric_cols}")

        # 4) choose remaining columns from the rest (random sample), make sure that keep_columns are included
        for c in keep_columns:
            if c not in chosen_numeric_cols and c in dataset_df.columns:
                chosen_numeric_cols.append(c)
        remaining_needed = number_of_cols - len(chosen_numeric_cols)
        other_pool = [c for c in dataset_df.columns if c not in numeric_like]

        # TODO: ask LLM about other pool to see which of the columns can be turned into categorical numerical columns (and get functions to do so..)

        if remaining_needed > len(other_pool):
            # this only happens if total_cols > available columns (should be prevented earlier),
            # but keep a clear message
            raise ValueError(f"Not enough columns to select {number_of_cols} unique columns "
                            f"(available: {len(dataset_df.columns)}).")

        if remaining_needed == 0:
            chosen_other_cols = []
        else:
            # sample remaining_needed columns from other_pool
            chosen_other_cols = other_pool[:remaining_needed]

        # optional informative warnings
        if num_numeric_wanted > num_numeric:
            print(
                f"Requested {num_numeric_wanted} numeric-like columns but only {len(numeric_like)} "
                f"numeric-like columns exist; selected {num_numeric} numeric-like columns."
            )
        selected_cols = chosen_other_cols + chosen_numeric_cols 
    else:
        # Simple truncation if no numeric proportion variation requested
        selected_cols = dataset_df.columns[:number_of_cols].tolist()

    variant_df = dataset_df[selected_cols]
    variant_df = rename_columns(variant_df, column_naming)

    # save variant_df
    save_dir.mkdir(exist_ok=True)
    variant_df.to_csv(save_dir / "input_table.csv", index=False)

    create_statistics(dataset_name=variation_name, 
                input_table_df=variant_df, 
                testcases=None,
                save_path=save_dir,
                primary_key_column="QID" # TODO: get from dataset_information
                )
    