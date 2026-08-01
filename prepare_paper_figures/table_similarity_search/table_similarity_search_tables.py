# Rows are the different approaches, columns are the different metrics
import pandas as pd
from pathlib import Path

import config_helpers as h

def create_results_table_small(df: pd.DataFrame, plots_folder: Path):
    # Copy relevant columns
    filtered_df = df[
        ['chart_name', 'Configuration', 'MRR', 'MAP', 'Recall']
    ].copy()

    # Clean approach names
    filtered_df['chart_name'] = filtered_df['chart_name'].str.replace('*', '', regex=False)

    # Identify variant
    filtered_df['variant'] = filtered_df['Configuration'].apply(
        lambda x: 'Just Schema' if 'table_row_limit=0' in str(x) else 'Schema + 100 rows'
    )

    # Round values
    metric_cols = ['MRR', 'MAP', 'Recall']
    filtered_df[metric_cols] = filtered_df[metric_cols].round(4)

    # Split variants
    df_schema = filtered_df[filtered_df['variant'] == 'Just Schema']
    df_100 = filtered_df[filtered_df['variant'] == 'Schema + 100 rows']

    # Set index
    df_schema = df_schema.set_index('chart_name')
    df_100 = df_100.set_index('chart_name')

    # Sort
    df_schema = h.sort_index_by_chart_name(df_schema)
    df_100 = h.sort_index_by_chart_name(df_100)

    # Write LaTeX
    with open(plots_folder / "table_similarity_search.tex", "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\resizebox{\\columnwidth}{!}{%\n")
        f.write("\\begin{tabular}{lccc|ccc}\n")
        f.write("\\toprule\n")

        # Header rows
        f.write(
            " & \\multicolumn{3}{c}{Variant: Just Schema} "
            "& \\multicolumn{3}{c}{Variant: Schema + 100 rows} \\\\\n"
        )
        f.write(
            "Approach & MRR & MAP & Recall & MRR & MAP & Recall \\\\\n"
        )
        f.write("\\midrule\n")

        # Iterate over all approaches
        all_approaches = sorted(set(df_schema.index).union(df_100.index), key=h.chart_name_sort_key)

        # per-column best value, so it can be bolded (ties are all bolded)
        best_schema = {col: df_schema[col].max() for col in metric_cols}
        best_100 = {col: df_100[col].max() for col in metric_cols}

        for approach in all_approaches:
            row_schema = df_schema.loc[approach] if approach in df_schema.index else None
            row_100 = df_100.loc[approach] if approach in df_100.index else None

            def fmt(row, col, best):
                if row is None or pd.isna(row[col]):
                    return "-"
                text = f"{row[col]:.4f}"
                if row[col] == best[col]:
                    text = f"\\textbf{{{text}}}"
                return text

            mrr_s = fmt(row_schema, 'MRR', best_schema)
            map_s = fmt(row_schema, 'MAP', best_schema)
            rec_s = fmt(row_schema, 'Recall', best_schema)

            mrr_100 = fmt(row_100, 'MRR', best_100)
            map_100 = fmt(row_100, 'MAP', best_100)
            rec_100 = fmt(row_100, 'Recall', best_100)

            f.write(
                f"{approach} & {mrr_s} & {map_s} & {rec_s} "
                f"& {mrr_100} & {map_100} & {rec_100} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")

        f.write("\\caption{\\revised{Table Similarity Search}: Results on GitTables, retrieving tables from the same git repository. Comparing schema-only vs schema+rows.}\n")
        f.write("\\label{tab:table_similarity_gitTables}\n")
        f.write("\\vspace{-2em}\n")
        f.write("\\end{table}\n")