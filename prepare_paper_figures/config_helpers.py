"""Shared utilities for paper figure scripts."""

import pandas as pd
import numpy as np
from pathlib import Path


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
    star: bool = True,
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
        if table_env:
            env = 'table*' if star else 'table'
            f.write(f'\\begin{{{env}}}[t]\n')
            f.write('\\centering\n')

        n_cols = len(value_cols)
        if star:
            f.write(f'\\begin{{tabular*}}{{\\textwidth}}{{l{"c" * n_cols}}}\n')
        else:
            f.write(f'\\begin{{tabular}}{{l{"c" * n_cols}}}\n')
        f.write('\\hline\n')

        header = f'{index_name} & ' + ' & '.join(value_cols) + ' \\\\\n'
        f.write(header)
        f.write('\\hline\n')

        for idx, row in df.iterrows():
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

        f.write('\\hline\n')
        f.write('\\end{tabular*}\n' if star else '\\end{tabular}\n')

        if table_env:
            if caption:
                f.write(f'\\caption{{{caption}}}\n')
            if label:
                f.write(f'\\label{{{label}}}\n')
            f.write(f'\\end{{{env}}}\n')
