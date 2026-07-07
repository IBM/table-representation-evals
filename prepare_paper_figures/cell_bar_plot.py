import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

def create_barplot(df: pd.DataFrame, plots_folder: Path):
    metric = "accuracy"

    # # Exclude some runs from the chart
    # exclude_runs = [
    # ("tabicl", "n_estimators=32,predML_based_on=row_embeddings"),
    # ("tabpfn", "device=cuda,predML_based_on=row_embeddings"),
    # ("baseline", "baseline"),
    # ("sap_rpt_oss", "bagging=1,max_context_size=2048,predML_based_on=row_embeddings"),
    # ("tabula_8b", "batch_size=1,device=cuda,max_length=512,model_name=mlfoundations_tabula-8b,n_few_shot_examples=10,predML_based_on=row_embeddings")
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

    print(f"Unique datasets (row_similarity): {num_datasets}")

    plot_df = df_filtered[
        ['dataset', 'accuracy', 'color', 'chart_name']
    ].copy()

    # ----------------------------
    # Plot (Grouped by Dataset)
    # ----------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique datasets and methods
    datasets = plot_df['dataset'].unique()
    methods = plot_df['chart_name'].unique()

    n_datasets = len(datasets)
    n_methods = len(methods)
    gap = 0.025
    bar_width = (0.8 - gap * (n_methods - 1)) / n_methods

    x = np.arange(n_datasets)  # positions for dataset groups
    total_group_width = n_methods * bar_width + (n_methods - 1) * gap
    group_centers = x + total_group_width / 2  # these are where the x-ticks go

    # Plot bars method by method
    for i, method in enumerate(methods):

        method_df = plot_df[plot_df['chart_name'] == method]

        # Ensure same dataset order
        method_df = (
            method_df
            .set_index('dataset')
            .loc[datasets]
            .reset_index()
        )

        positions = x + i * (bar_width + gap)

        bars = ax.bar(
            positions,
            method_df[metric],
            width=bar_width,
            color=method_df['color'].iloc[0],
            label=method
        )

        # ----------------------------
        # Add labels above bars
        # ----------------------------
        for j, bar in enumerate(bars):

            height = bar.get_height()

            label_text = f"{int(height)}" if height == 0 else f"{height:.2f}"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,                  # small offset
                label_text,                # text
                ha='center',
                va='bottom',
                fontsize=11,
                rotation=0
            )


    # Center x-ticks under each group
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        datasets,
        rotation=0,
        ha='center',
        fontsize=14
    )

    # Axes
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_xticklabels():
        x, y = label.get_position()
        label.set_x(x + 0.01)

    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Legend below figure
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=6,
        frameon=False,
        fontsize=13
    )

    #plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.tight_layout()
    plt.show()

    # Save
    plt.savefig(
        plots_folder / f"cell_barchart_{metric.lower()}_per_dataset.pdf"
    )
    plt.close()
