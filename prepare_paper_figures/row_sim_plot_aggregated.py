import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

def create_barplot(df: pd.DataFrame, plots_folder: Path):
    metric = "MRR"

    # Exclude some runs from the chart
    # TODO: is this still needed here?
    # exclude_runs = [
    # #("tabicl", "n_estimators=32,predML_based_on=row_embeddings"),
    # #("tabpfn", "device=cuda,predML_based_on=row_embeddings"),
    # #("baseline", "baseline"),
    # #("sap_rpt_oss", "bagging=1,max_context_size=2048,predML_based_on=row_embeddings"),
    # #("tabula_8b", "batch_size=1,device=cuda,max_length=512,model_name=mlfoundations_tabula-8b,n_few_shot_examples=10,predML_based_on=row_embeddings")
    # ]
    # df = df[
    #     ~df.set_index(['Approach', 'Configuration']).index.isin(exclude_runs)
    # ]

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

    df_filtered = df[df['dataset'].isin(full_datasets)]

    num_datasets = len(df_filtered['dataset'].unique())

    print("########################################################")
    print(f"Unique datasets (row_similarity search): {num_datasets}")

    # strip * from chart names
    df_filtered['chart_name'] = df_filtered['chart_name'].str.replace('*', '', regex=False)

    # aggregate the data per approach/configuration (= over the datasets)
    agg_df = df_filtered.groupby(['Approach', 'Configuration']).agg(
        MRR_mean=(metric, 'mean'),
        MRR_std=(metric, 'std'),   # std deviation across datasets
        color=('color', 'first'),
        chart_name=('chart_name', 'first')
    ).reset_index()

    # ----------------------------
    # Plot
    # ----------------------------
    
    # set figure size
    
    fig, ax = plt.subplots(figsize=(12,6))

    # Barplot with custom colors
    bars = ax.bar(
        x=agg_df['chart_name'], # use chart_name labels
        height=agg_df['MRR_mean'], 
        color=agg_df['color'], # use custom colors
        width=0.8  # increase bar width (default is 0.8)
    )

    # Set y-axis from 0 to 1
    ax.set_ylim(0, 1)

    # set y axis label
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    # rotate the labels
    ax.set_xticklabels(agg_df['chart_name'], rotation=45, ha='right', fontsize=16)

    # ensure numeric positions
    ax.set_xticks(range(len(agg_df)))  

    # Remove legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()


    # Add value + std above each bar
    for i, bar in enumerate(bars):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        std = agg_df['MRR_std'].iloc[i]
        ax.text(
            x, y + 0.02,  # slightly above the bar
            f"{agg_df['MRR_mean'].iloc[i]:.2f}", #(±{std:.2f})",
            ha='center', va='bottom', fontsize=14
        )

    plt.tight_layout()
    plt.show()

    # Save PDF
    plt.savefig(plots_folder / f"row_sim_barchart_aggregated_{metric.lower()}_{num_datasets}_datasets.pdf")
    plt.close()