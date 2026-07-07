import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

def create_barplot(df: pd.DataFrame, plots_folder: Path):
    metric = "accuracy"

    # drop columns with all NaNs
    df = df.dropna(axis=1, how="all")

    # keep only rows where we have results for all datasets
    approach_config_counts = (
        df.groupby(['Approach', 'Configuration'])['dataset']
        .nunique()
        .reset_index()
        .rename(columns={'dataset': 'num_datasets'})
    )

    num_datasets = df['dataset'].nunique()
    full_configs = approach_config_counts[
        approach_config_counts['num_datasets'] == num_datasets
    ][['Approach', 'Configuration']]

    df_filtered = df.merge(full_configs, on=['Approach', 'Configuration'], how='inner')


        # strip * from chart names
    df_filtered['chart_name'] = df_filtered['chart_name'].str.replace('*', '', regex=False)

    plot_df = df_filtered[['dataset', 'Approach', 'Configuration', 'accuracy', 'color', 'chart_name']].copy()

    # ----------------------------
    # Plot (Grouped by Approach/Configuration)
    # ----------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    # Only the two datasets
    datasets_to_plot = ['s2abel@dirty', 's2abel@clean']
    plot_df = plot_df[plot_df['dataset'].isin(datasets_to_plot)]

    # Unique methods (Approach/Config) for x-axis
    methods = plot_df['chart_name'].unique()
    x = np.arange(len(methods))

    # Extract dirty values first
    dirty_df = plot_df[plot_df['dataset'] == 's2abel@dirty']
    dirty_df = dirty_df.set_index('chart_name').reindex(methods).reset_index()
    dirty_vals = dirty_df['accuracy'].values
    dirty_colors = dirty_df['color'].values

    # Plot dirty bars (base)
    ax.bar(
        x,
        dirty_vals,
        width=0.8,
        color=dirty_colors,
        label='s2abel@dirty'
    )

    # Extract clean values
    clean_df = plot_df[plot_df['dataset'] == 's2abel@clean']
    clean_df = clean_df.set_index('chart_name').reindex(methods).reset_index()
    clean_vals = clean_df['accuracy'].values
    clean_colors = clean_df['color'].values

    # Plot clean bars on top (semi-transparent)
    ax.bar(
        x,
        clean_vals,
        width=0.8,
        color=clean_colors,
        alpha=0.5,
        label='s2abel@clean'
    )

    # Add labels for dirty and clean bars
    for xi, dirty, clean in zip(x, dirty_vals, clean_vals):
        # Dirty bar label (inside the bar)
        ax.text(
            xi,
            dirty - 0.03,                # centered inside dirty bar
            f"{dirty:.2f}",
            ha='center',
            va='center',
            fontsize=14,
            color='white' if dirty > 0.1 else 'black'
        )

        # Clean bar label (above the clean bar)
        ax.text(
            xi,
            clean + 0.01,             # just above the clean bar
            f"{clean:.2f}",
            ha='center',
            va='bottom',
            fontsize=14,
            color='black'
        )

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=10, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Axes and grid
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Legend
    ax.legend(title="Dataset", fontsize=12)

    plt.tight_layout()
    plt.show()



    # Save
    plt.savefig(
        plots_folder / f"cell_barchart_{metric.lower()}_stacked.pdf"
    )
    plt.close()
