import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

import config_helpers as h

METRIC_LABELS = {
    "MRR": "Mean Reciprocal Rank (MRR)",
    "MAP": "MAP (mean)",
}

def create_barplot(df: pd.DataFrame, plots_folder: Path):
    # drop columns with all nans (result metrics from other tasks will be nan)
    df = df.dropna(axis=1, how="all")

    print(df.columns)

    #########################################################
    # keep only data where we have results for all datasets
    #########################################################
    # count unique (Approach, Configuration) per dataset
    dataset_counts = df.groupby('dataset')[['Approach','Configuration']].nunique().reset_index()
    dataset_counts.rename(columns={'Approach':'num_approaches','Configuration':'num_configs'}, inplace=True)

    # Total number of unique approaches and configs
    n_approaches = df['Approach'].nunique()
    n_configs = df['Configuration'].nunique()

    # Keep datasets where we have all approaches AND configs
    full_datasets = dataset_counts[
        (dataset_counts['num_approaches'] == n_approaches) &
        (dataset_counts['num_configs'] == n_configs)
    ]['dataset'].tolist()

    df_filtered = df[df['dataset'].isin(full_datasets)].copy()

    num_datasets = len(df_filtered['dataset'].unique())

    print("########################################################")
    print(f"Unique datasets (row_similarity search): {num_datasets}")

    # strip * from chart names
    df_filtered['chart_name'] = df_filtered['chart_name'].str.replace('*', '', regex=False)

    for metric in ("MRR", "MAP"):
        if metric not in df_filtered.columns:
            print(f"WARNING: {metric} not found in row_similarity_search data, skipping")
            continue

        # aggregate the data per approach/configuration (= over the datasets)
        agg_df = df_filtered.groupby(['Approach', 'Configuration']).agg(
            metric_mean=(metric, 'mean'),
            metric_std=(metric, 'std'),   # std deviation across datasets
            color=('color', 'first'),
            chart_name=('chart_name', 'first')
        ).reset_index()

        # order bars alphabetically by chart_name (display name), not by score
        # or the default (raw Approach name) groupby order
        agg_df = agg_df.sort_values("chart_name", key=lambda s: s.map(h.chart_name_sort_key)).reset_index(drop=True)

        # ----------------------------
        # Plot
        # ----------------------------

        fig, ax = plt.subplots(figsize=(12,4.2))

        # Barplot with custom colors
        bars = ax.bar(
            x=agg_df['chart_name'], # use chart_name labels
            height=agg_df['metric_mean'],
            color=agg_df['color'], # use custom colors
            width=0.8  # increase bar width (default is 0.8)
        )

        # Set y-axis from 0 to 1
        ax.set_ylim(0, 1)

        # set y axis label
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=16)
        ax.tick_params(axis='both', labelsize=14)

        # ensure numeric positions
        ax.set_xticks(range(len(agg_df)))

        # rotate the labels
        ax.set_xticklabels(agg_df['chart_name'], rotation=25, ha='right', fontsize=16)

        # Remove legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # Add value + std above each bar
        for i, bar in enumerate(bars):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            std = agg_df['metric_std'].iloc[i]
            ax.text(
                x, y + 0.02,  # slightly above the bar
                f"{agg_df['metric_mean'].iloc[i]:.2f}", #(±{std:.2f})",
                ha='center', va='bottom', fontsize=14
            )

        plt.tight_layout()
        plt.show()

        # Save PDF
        plt.savefig(plots_folder / f"row_sim_barchart_aggregated_{metric.lower()}_{num_datasets}_datasets.pdf")
        plt.close()
