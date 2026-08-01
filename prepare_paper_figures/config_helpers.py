"""Shared utilities for paper figure scripts."""

import pandas as pd
import numpy as np
from pathlib import Path


# Row counts for predictive_ml datasets, transcribed from the per-dataset
# comments in configs/global_datasets.yaml (predictive_ml_datasets is fetched
# live via OpenML at run time, so no structured size metadata exists elsewhere).
PREDICTIVE_ML_DATASET_ROWS = {
    "credit-g": 1000,
    "Fitness_Club": 1500,
    "Is-this-a-good-customer": 1723,
    "Marketing_Campaign": 2240,
    "seismic-bumps": 2584,
    "coil2000_insurance_policies": 9822,
    "Bank_Customer_Churn": 10000,
    "E-CommereShippingData": 10999,
    "online_shoppers_intention": 12330,
    "in_vehicle_coupon_recommendation": 12684,
    "HR_Analytics_Job_Change_of_Data_Scientists": 19158,
    "bank-marketing": 45211,
    "kddcup09_appetency": 50000,
    "Diabetes130US": 71518,
    "customer_satisfaction_in_airline": 129880,
    "blood-transfusion-service-center": 748,
    "diabetes": 768,
    "qsar-biodeg": 1054,
    "hazelnut-spread-contaminant-detection": 2400,
    "Bioresponse": 3751,
    "churn": 5000,
    "polish_companies_bankruptcy": 5910,
    "taiwanese_bankruptcy_prediction": 6819,
    "NATICUSdroid": 7491,
    "heloc": 10459,
    "jm1": 10885,
    "credit_card_clients_default": 30000,
    "Amazon_employee_access": 32769,
    "APSFailure": 76000,
    "GiveMeSomeCredit": 150000,
    "anneal": 898,
    "website_phishing": 1353,
    "splice": 3190,
    "students_dropout_and_academic_success": 4424,
    "maternal_health_risk": 1014,
    "MIC": 1699,
    "hiva_agnostic": 3845,
    "SDSS17": 78053,
    "healthcare_insurance_expenses": 1338,
    "Another-Dataset-on-used-Fiat-500": 1538,
    "wine_quality": 6497,
    "Food_Delivery_Time": 45451,
    "diamonds": 53940,
    "QSAR_fish_toxicity": 907,
    "concrete_compressive_strength": 1030,
    "airfoil_self_noise": 1503,
    "QSAR-TID-11": 5742,
    "miami_housing": 13776,
    "houses": 20640,
    "superconductivity": 21263,
    "physiochemical_protein": 45730,
}


# Curated display names for dataset identifiers that would otherwise show up
# raw (often with underscores, which break unescaped in LaTeX tables) in both
# figures and results tables for their task. Shared here so a table and its
# companion figure never drift apart on how a dataset is labeled.
COLUMN_SIMILARITY_SEARCH_DATASET_NAMES = {
    "nextia": "NextiaJD",
    "opendata": "OpenData",
    "valentine": "Valentine",
    "wikijoin_small": "WikiJoin-Small",
    "autojoin": "AutoJoin",
}

VALUE_LINKING_DATASET_NAMES = {
    "bird_cell_exact": "BIRD Exact Match",
    "bird_cell_fuzzy": "BIRD Fuzzy Match",
}

COLUMN_TYPE_ANNOTATION_DATASET_NAMES = {
    "sotab": "SOTAB",
    "gittables_cta": "GitTables",
}

SCHEMA_LINKING_DATASET_NAMES = {
    "bird_column_schema": "BIRD Column Schema",
}

TABLE_SHUFFLING_DATASET_NAMES = {
    "ckan_subset": "CKAN Subset",
    "ecb": "ECB",
    "fetaqa": "FeTaQA",
    "ottqa": "OTT-QA",
    "spider-train": "Spider",
    "tabfact": "TabFact",
}


def chart_name_sort_key(name: str) -> str:
    """Case-insensitive alphabetical sort key for a chart_name; strips a trailing
    '*' (the predictive_ml row-embeddings marker) so an approach's starred and
    unstarred variants sort together."""
    return str(name).rstrip('*').lower()


def sort_columns_by_chart_name(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder a DataFrame's columns alphabetically by chart_name (case-insensitive)."""
    return df[sorted(df.columns, key=chart_name_sort_key)]


def sort_index_by_chart_name(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder a DataFrame's index alphabetically by chart_name (case-insensitive)."""
    return df.reindex(sorted(df.index, key=chart_name_sort_key))


def parse_variation(dataset: str) -> tuple[str, str]:
    """Split 'dataset@@vN' into ('dataset', 'vN'). Returns ('dataset', '') if no variation."""
    if '@@' in dataset:
        base, var = dataset.rsplit('@@', 1)
        return base, var
    return dataset, ''


def extract_row_limit(config: str) -> int | None:
    """Extract table_row_limit=N from config string. Returns None if not found."""
    for part in str(config).split(','):
        part = part.strip()
        if part.startswith('table_row_limit='):
            return int(part.split('=')[1])
    return None


def extract_serialization(config: str) -> str | None:
    """Extract table_serialization_format=X from config string. Returns None if not found."""
    for part in str(config).split(','):
        part = part.strip()
        if part.startswith('table_serialization_format='):
            return part.split('=')[1]
    return None


def filter_by_row_limit(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Filter df to rows where Configuration contains table_row_limit=<limit>."""
    return df[df['Configuration'].str.contains(f'table_row_limit={limit}', na=False)]


def filter_by_serialization(df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """Filter df to rows where Configuration contains table_serialization_format=<fmt>."""
    return df[df['Configuration'].str.contains(f'table_serialization_format={fmt}', na=False)]


def filter_chart_names(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Filter df to rows where chart_name ends with suffix (e.g. '(md)')."""
    return df[df['chart_name'].str.endswith(suffix, na=False)]


def write_latex_table(
    pivoted_df: pd.DataFrame,
    plots_folder: Path,
    filename: str,
    caption: str = '',
    label: str = '',
    index_name: str = 'Dataset',
    bold_best: bool = True,
    underline_second: bool = True,
    higher_better: bool = True,
    axis: str = 'rows',
    float_fmt: str = '.4f',
    nan_str: str = '---',
    table_env: bool = True,
    star: bool = False,
):
    """
    Write a LaTeX table from a pivoted DataFrame.

    Parameters
    ----------
    pivoted_df : rows indexed by dataset (or approach), columns are chart names.
    bold_best : bold the best value.
    underline_second : underline the second-best value.
    higher_better : if True, max is best; if False, min is best (e.g. BCS).
    axis : 'rows' — best per row (e.g. best approach per dataset);
           'columns' — best per column (e.g. best approach per perturbation type).
    """
    df = pivoted_df.copy()
    value_cols = list(df.columns)

    # Add Mean row
    mean_series = df.mean(numeric_only=True)
    df.loc['Mean'] = mean_series

    # Precompute per-cell formatting tuples: (value, bold, underline)
    cell_formats = {}
    if bold_best:
        if axis == 'rows':
            for idx, row in df.iterrows():
                numeric_vals = {c: v for c, v in row.items() if c in value_cols and pd.notna(v)}
                if not numeric_vals:
                    continue
                best_val = max(numeric_vals.values()) if higher_better else min(numeric_vals.values())
                sorted_vals = sorted(set(numeric_vals.values()),
                                     reverse=higher_better)
                second_val = sorted_vals[1] if len(sorted_vals) > 1 and underline_second else None
                for c, v in numeric_vals.items():
                    cell_formats[(idx, c)] = (v, v == best_val, v == second_val)
        else:  # axis == 'columns'
            for c in value_cols:
                col_vals = {idx: df.loc[idx, c] for idx in df.index
                           if pd.notna(df.loc[idx, c])}
                if not col_vals:
                    continue
                best_val = max(col_vals.values()) if higher_better else min(col_vals.values())
                sorted_vals = sorted(set(col_vals.values()),
                                     reverse=higher_better)
                second_val = sorted_vals[1] if len(sorted_vals) > 1 and underline_second else None
                for idx, v in col_vals.items():
                    was_set = cell_formats.get((idx, c))
                    if was_set is None:
                        cell_formats[(idx, c)] = (v, v == best_val, v == second_val)

    with open(plots_folder / filename, 'w') as f:
        n_cols = len(value_cols)

        if table_env:
            env = 'table*' if star else 'table'
            f.write(f'\\begin{{{env}}}[t]\n')
            f.write('\\centering\n')
            if not star:
                f.write('\\resizebox{\\columnwidth}{!}{%\n')

        if star:
            # tabular* with @{\extracolsep{\fill}} stretches to fill \textwidth,
            # spanning both columns (double-column layout)
            f.write(f'\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l {"c " * n_cols}@{{}}}}\n')
        else:
            f.write(f'\\begin{{tabular}}{{l{"c" * n_cols}}}\n')
        f.write('\\toprule\n')

        header = f'{index_name} & ' + ' & '.join(value_cols) + ' \\\\\n'
        f.write(header)
        f.write('\\midrule\n')

        for idx, row in df.iterrows():
            if idx == 'Mean':
                f.write('\\midrule\n')

            formatted = []
            for c in value_cols:
                v = row[c]
                if pd.isna(v):
                    formatted.append(nan_str)
                else:
                    s = f'{v:{float_fmt}}'
                    fmt = cell_formats.get((idx, c))
                    if fmt:
                        _, is_best, is_second = fmt
                        if is_best:
                            s = f'\\textbf{{{s}}}'
                        elif is_second:
                            s = f'\\underline{{{s}}}'
                    formatted.append(s)

            row_label = str(idx).replace('_', '\\_')
            f.write(f'{row_label} & ' + ' & '.join(formatted) + ' \\\\\n')

        f.write('\\bottomrule\n')
        f.write('\\end{tabular*}\n' if star else '\\end{tabular}\n')

        if table_env:
            if not star:
                f.write('}\n')  # close \resizebox
            if caption:
                f.write(f'\\caption{{{caption}}}\n')
            if label:
                f.write(f'\\label{{{label}}}\n')
            f.write(f'\\end{{{env}}}\n')
