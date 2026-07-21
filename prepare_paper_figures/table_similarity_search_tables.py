# Rows are the different approaches, columns are the different metrics
import pandas as pd
from pathlib import Path

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
    df_schema = df_schema.sort_index()
    df_100 = df_100.sort_index()

    # Write LaTeX
    with open(plots_folder / "table_similarity_search.tex", "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")

        # Header rows
        f.write(
            " & \\multicolumn{3}{c}{Variant: Just Schema} "
            "& \\multicolumn{3}{c}{Variant: Schema + 100 rows} \\\\\n"
        )
        f.write(
            "Approach & MRR & MAP & Recall & MRR & MAP & Recall \\\\\n"
        )
        f.write("\\hline\n")

        # Iterate over all approaches
        all_approaches = sorted(set(df_schema.index).union(df_100.index))

        for approach in all_approaches:
            row_schema = df_schema.loc[approach] if approach in df_schema.index else None
            row_100 = df_100.loc[approach] if approach in df_100.index else None

            def fmt(row, col):
                if row is None or pd.isna(row[col]):
                    return "-"
                return f"{row[col]:.4f}"

            mrr_s = fmt(row_schema, 'MRR')
            map_s = fmt(row_schema, 'MAP')
            rec_s = fmt(row_schema, 'Recall')

            mrr_100 = fmt(row_100, 'MRR')
            map_100 = fmt(row_100, 'MAP')
            rec_100 = fmt(row_100, 'Recall')

            f.write(
                f"{approach} & {mrr_s} & {map_s} & {rec_s} "
                f"& {mrr_100} & {map_100} & {rec_100} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")

        f.write("\\caption{Table Similarity Search Results comparing schema-only vs schema+rows.}\n")
        f.write("\\label{tab:table_similarity_search}\n")
        f.write("\\end{table}\n")