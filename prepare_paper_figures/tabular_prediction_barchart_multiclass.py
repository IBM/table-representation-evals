import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def create_barplot_multiclass(df: pd.DataFrame, plots_folder: Path):

    metric = "XGBoost_log_loss (↓)_ratio_to_baseline"
    df = df.dropna(subset=[metric])

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

    print(f"Unique datasets (multiclass classification): {num_datasets}")

    # Convert metric to percentage before aggregation
    df_filtered = df_filtered.copy()
    df_filtered[metric] *= 100

    # aggregate the data per approach/configuration (= over the datasets)
    agg_df = df_filtered.groupby(['Approach', 'Configuration']).agg(
        metric_mean=(metric, 'mean'),
        metric_std=(metric, 'std'),   # std deviation across datasets
        color=('color', 'first'),
        chart_name=('chart_name', 'first')
    ).reset_index()


    # Define mapping from old chart_name to new labels
    rename_mapping = {
        "Baseline": "XGBoost",
        "GritLM": "GritLM*",
        "HyTrel": "HyTrel*",
        "IBM Granite R2": "IBM Granite R2*",
        "MiniLM": "MiniLM*",
        "SAP-RPT-1": "SAP-RPT-1",
        "SAP-RPT-1_row": "SAP-RPT-1*",
        "TabICL_row": "TabICL*",
        "TabPFN_row": "TabPFN*",
        "TabuLa-8B_row": "TabuLa-8B*"

    }

    # Apply mapping to the chart_name column
    agg_df['chart_name'] = agg_df['chart_name'].replace(rename_mapping)

    # Sort alphabetically by chart_name and have XGBoost first
    xgb_df = agg_df[agg_df['chart_name'] == "XGBoost"]
    rest_df = agg_df[agg_df['chart_name'] != "XGBoost"].sort_values("chart_name")

    # Concatenate with XGBoost first
    agg_df = pd.concat([xgb_df, rest_df]).reset_index(drop=True)



    print(f"Aggregated df for plotting: {agg_df.columns}, {len(agg_df)} rows")

    # ----------------------------
    # Plot
    # ----------------------------
    
    # set figure size
    
    fig, ax = plt.subplots(figsize=(12,6))

    # Barplot with custom colors
    bars = ax.bar(
        x=agg_df['chart_name'], # use chart_name labels
        height=agg_df["metric_mean"], 
        color=agg_df['color'], # use custom colors
        width=0.8  # increase bar width (default is 0.8)
    )

    # Set y-axis from 0 to 1
    ax.set_ylim(0, 140)

    # set y axis label
    ax.set_ylabel("Ratio [%] to XGBoost Log Loss Score", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    # rotate the labels
    ax.set_xticklabels(agg_df['chart_name'], rotation=30, ha='right', fontsize=16)

    # ensure numeric positions
    ax.set_xticks(range(len(agg_df)))  

    # Remove legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Add a horizontal line at y=100
    ax.axhline(
        y=100,            # y-coordinate
        color='gray',      # line color
        linestyle='--',    # dashed line
        linewidth=1        # thickness
    )

    # Add value + std above each bar
    for i, bar in enumerate(bars):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        std = agg_df['metric_std'].iloc[i]
        ax.text(
            x, y + 0.02,  # slightly above the bar
            f"{agg_df["metric_mean"].iloc[i]:.1f}", # (±{std:.1f})",
            ha='center', va='bottom', fontsize=14
        )

    plt.tight_layout()
    plt.show()

    # Save PDF
    plt.savefig(plots_folder / f"tabular_prediction_multiclass_barchart_aggregated_ratio_{num_datasets}_datasets.pdf")
    plt.close()