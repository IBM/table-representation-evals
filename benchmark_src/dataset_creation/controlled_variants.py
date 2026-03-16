import pandas as pd
import json
import os
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder


TOTAL_COLUMNS = 10
PROPORTIONS = [0, 0.25, 0.5, 0.75, 1.0]
OUTPUT_DIR = "variants"


def compute_missing_rates(df: pd.DataFrame) -> pd.Series:
    return df.isnull().mean()


def build_column_pools(df: pd.DataFrame, missing_rates: pd.Series
                       ) -> Tuple[List[str], List[str], pd.Series, pd.Series]:
    numeric_cols = missing_rates[df.dtypes.isin(["float64", "int64"])].sort_values().index.tolist()
    non_numeric_cols = missing_rates[df.dtypes == "object"].sort_values().index.tolist()
    return numeric_cols, non_numeric_cols, missing_rates[numeric_cols], missing_rates[non_numeric_cols]


def match_missingness(numeric_rates: pd.Series, non_numeric_rates: pd.Series) -> Dict[str, str]:
    """Match each numeric column missingness rate to closest non-numeric column missingness rate."""
    matches = {}
    for num_col, num_rate in numeric_rates.items():
        closest_col = min(non_numeric_rates.index, key=lambda col: abs(non_numeric_rates[col] - num_rate))
        matches[num_col] = closest_col
    return matches


def transform_non_numeric_to_numeric(df: pd.DataFrame, cols: List[str]
                                     ) -> Tuple[pd.DataFrame, List[str]]:
    """Transform selected non-numeric columns into numeric using label encoding."""
    transformed_cols = []
    for col in cols:
        if df[col].dtype == "object":
            df[col + "_encoded"] = LabelEncoder().fit_transform(df[col].astype(str).fillna("missing"))
            transformed_cols.append(col + "_encoded")
    return df, transformed_cols


def controlled_column_selection_with_matching(
        df: pd.DataFrame,
        numeric_cols: List[str],
        non_numeric_cols: List[str],
        numeric_missing_rates: pd.Series,
        non_numeric_missing_rates: pd.Series,
        proportion_numeric: float,
        total_columns: int
    ) -> Tuple[List[str], List[str], List[str]]:
    """Select columns controlling numeric proportion and matching missingness patterns."""
    num_numeric = int(total_columns * proportion_numeric)
    num_non_numeric = total_columns - num_numeric

    selected_numeric = numeric_cols[:num_numeric]
    selected_non_numeric = non_numeric_cols[:num_non_numeric]

    transformed_cols = []
    if len(selected_numeric) < num_numeric:
        needed = num_numeric - len(selected_numeric)
        # Match missingness
        matches = match_missingness(numeric_missing_rates, non_numeric_missing_rates)
        fallback_candidates = [matches[col] for col in numeric_cols[:needed] if matches[col] in non_numeric_cols]
        df, transformed_cols = transform_non_numeric_to_numeric(df, fallback_candidates)
        selected_numeric.extend(transformed_cols)
        selected_non_numeric = [col for col in selected_non_numeric if col not in fallback_candidates]

    return selected_numeric, selected_non_numeric, transformed_cols


def generate_variants(df: pd.DataFrame,
                      proportions: List[float],
                      total_columns: int,
                      output_dir: str) -> List[dict]:
    missing_rates = compute_missing_rates(df)
    numeric_cols, non_numeric_cols, numeric_missing_rates, non_numeric_missing_rates = \
        build_column_pools(df, missing_rates)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    manifest = []
    for p in proportions:
        numeric_selection, non_numeric_selection, transformed_cols = \
            controlled_column_selection_with_matching(
                df, numeric_cols, non_numeric_cols, numeric_missing_rates,
                non_numeric_missing_rates, p, total_columns
            )

        selected_columns = numeric_selection + non_numeric_selection
        variant_df = df[selected_columns]

        variant_name = f"numeric_{int(p*100)}"
        variant_path = os.path.join(output_dir, f"{variant_name}.csv")
        variant_df.to_csv(variant_path, index=False)

        manifest.append({
            "variant": variant_name,
            "numeric_columns": numeric_selection,
            "non_numeric_columns": non_numeric_selection,
            "transformed_columns": transformed_cols,
            "file": variant_path
        })

    return manifest


if __name__ == "__main__":
    # === Load your dataset here ===
    df = pd.read_csv("wikidata_books_table.csv")  # Replace with your file path

    manifest = generate_variants(df, PROPORTIONS, TOTAL_COLUMNS, OUTPUT_DIR)

    manifest_path = os.path.join(OUTPUT_DIR, "controlled_variant_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Controlled variants generated in '{OUTPUT_DIR}' folder")
    print(f"Manifest saved at: {manifest_path}")
    print(json.dumps(manifest, indent=2))
