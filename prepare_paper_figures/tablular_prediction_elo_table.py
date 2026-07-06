import pandas as pd
from pathlib import Path

from benchmark_src.results_processing import ranking

def create_elo_table(df: pd.DataFrame, results_folder: Path):
    """
    Compute ELO scores and write a VLDB-style LaTeX table (single-column) using booktabs.
    """
    # Compute ELO scores
    elo_df, _ = ranking.compute_elo_scores(df)

    ############################################
    print(f"######## ELO df: ################")
    print(elo_df)

    # --- Merge chart_name from original df for Model column ---
    name_map = df[["task", "Approach", "Configuration", "chart_name"]].drop_duplicates()
    elo_df = elo_df.merge(name_map, on=["task", "Approach", "Configuration"], how="left")

    # --- Round scores ---
    elo_df["elo_score_task"] = elo_df["elo_score_task"].round(0).astype(int)
    elo_df["elo_task_delta"] = elo_df["elo_task_delta"].round(0).astype(int)

    # --- Sort by ELO descending ---
    elo_df = elo_df.sort_values("elo_score_task", ascending=False)

    # --- Select columns for table ---
    table_df = elo_df[["chart_name", "elo_score_task", "elo_task_delta", "num_comparisons_task"]]
    table_df = table_df.rename(columns={
        "chart_name": "Model",
        "elo_score_task": "ELO Score",
        "elo_task_delta": r"$\Delta$",
        "num_comparisons_task": "\#Comp"
    })

    # --- Write LaTeX table to file ---
    table_file = results_folder / "tabular_prediction_elo_ranking_table.tex"
    with open(table_file, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\begin{tabular}{p{0.45\\columnwidth}rrr}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(table_df.columns) + " \\\\\n")
        f.write("\\midrule\n")

        # Write table rows
        for _, row in table_df.iterrows():
            row_values = []
            for v in row:
                if isinstance(v, str):
                    row_values.append(v)
                else:
                    row_values.append(str(v))
            f.write(" & ".join(row_values) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

        # Caption with all necessary background
        f.write("\\caption{Tabular Prediction: ELO scores calculated from pairwise comparisons, where higher values indicate better performance. "
                "All models are initialized at 1500 and updated over 20 rounds. "
                "$\Delta$ reports the change from the initial rating, and \#Comp denotes the number of comparisons, which may differ since not all models ran on every dataset."
                "}\n")
        f.write("\\label{tab:tabular_prediction_elo}\n")
        f.write("\\end{table}\n")

    return elo_df