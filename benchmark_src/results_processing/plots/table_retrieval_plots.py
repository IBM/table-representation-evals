import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

from aggregate import aggregate_results
import benchmark_src.results_processing.plots.plot_utils as plot_utils

def create_barplot(df: pd.DataFrame, results_folder: Path):
    print(f"############## Started table retrieval barplot")
    group_cols = ["Approach", "Configuration", "task"]
    df = aggregate_results(df=df, grouping_columns=group_cols)

    print(df)

    for top_k in [1, 3, 5, 10]:

        data = plot_utils.collect_data_for_plotting(df=df, metric=f"Recall@{top_k}", is_aggregated=True)
        
        # Convert from 0-1 range to 0-100 percentage
        data['Performance'] = data['Performance'] * 100

        fig = plt.figure(figsize=(12, 9)) # Set the figure size
        sns.set_theme(font_scale=1.5) 

        plt.ylim(top=100)
        ax = sns.barplot(x='Approach', 
                        y='Performance', 
                        data=data, 
                        hue='Approach', 
                        palette='Set1', 
                        width=0.8) # Using the DataFrame

        # 1. Get the current tick locations (numerical positions)
        tick_locations = ax.get_xticks()

        # 2. Set the tick locations first (fixing them)
        ax.set_xticks(tick_locations)

        # 3. Then set the tick labels with rotation and alignment
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

        #fig.subplots_adjust(top=0.95)
        fig.subplots_adjust(top=0.88, bottom=0.25, left=0.1, right=0.95)

        # Manually add labels using ax.text()
        for i, bar in enumerate(ax.patches):  # Iterate through the bar objects (rectangles)
            x = bar.get_x() + bar.get_width() / 2  # Center the text horizontally
            y = bar.get_height()  # Position at the top of the bar
            
            # Customize the text and its placement as needed
            #text = f'{data["Performance"][i]:.2f} (± {data["std"][i]:.2f})'
            text = f'{data["Performance"][i]:.2f}'
            ax.text(x, y, text, ha='center', va='bottom', color='black', fontsize=14)

        # Adding a title and labels (if not already done within the ax.set() or similar)
        ax.set_title(f"Table Retrieval - Recall@{top_k} (Aggregated over datasets)")
        ax.set_xlabel("Approach")
        ax.set_ylabel("Recall@{top_k} [%]")

        plt.savefig(results_folder / f"aggregated_barchart_recall@{top_k}.png")
    print(f"############## Finished table retrieval barplot")


def create_barplot_datasets(df: pd.DataFrame, results_folder: Path):
    print(f"############## Started table retrieval barplot datasets")

    for top_k in [1, 3, 5, 10]:
        
        data = plot_utils.collect_data_for_plotting(df=df, metric=f"Recall@{top_k}", is_aggregated=False)
        
        # Convert from 0-1 range to 0-100 percentage
        data['Performance'] = data['Performance'] * 100

        fig = plt.figure(figsize=(14, 10)) # Set the figure size
        plt.ylim(top=100)
        ax = sns.barplot(x='Dataset', 
                        y='Performance', 
                        data=data, 
                        hue='Approach', 
                        palette='Set1', 
                        width=0.65) # Using the DataFrame
        
        # Adjust the x-tick labels to be centered and rotated
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

        # Adjust subplots for better layout
        fig.subplots_adjust(top=0.88, bottom=0.25, left=0.1, right=0.95)

        # Adding a title and labels (if not already done within the ax.set() or similar)
        ax.set_title(f"Table Retrieval - Recall@{top_k}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Recall@{top_k} [%]")

        ############################################################
        # Add legend
        ############################################################
        plt.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=2, borderaxespad=0.)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout(rect=[0, 0.001, 1, 1])

        plt.savefig(results_folder / f"barchart_recall@{top_k}.png")
    print(f"############## Finished table retrieval barplot datasets")


def create_recall_growth_curve(df: pd.DataFrame, results_folder: Path):
    """
    Create a line chart showing Recall@k growth for k=[1,3,5,10]
    X-axis: k values
    Y-axis: Recall@k_mean (aggregated over datasets)
    Each line represents an Approach/Configuration
    """
    print(f"############## Started recall growth curve")
    
    # Aggregate over datasets to get mean Recall@k for each approach
    group_cols = ["Approach", "Configuration", "task"]
    aggregated_df = aggregate_results(df=df, grouping_columns=group_cols)
    
    # Collect data for each k value
    k_values = [1, 3, 5, 10]
    plot_data = []
    
    for k in k_values:
        data = plot_utils.collect_data_for_plotting(df=aggregated_df, metric=f"Recall@{k}", is_aggregated=True)
        for _, row in data.iterrows():
            plot_data.append({
                'Approach': row['Approach'],
                'k': k,
                'Recall': row['Performance'] * 100  # Convert to percentage
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the line plot
    fig = plt.figure(figsize=(12, 8))
    sns.set_theme(font_scale=1.3)
    
    # Get unique approaches for colors
    unique_approaches = plot_df['Approach'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_approaches))
    
    # Plot a line for each approach
    for i, approach in enumerate(unique_approaches):
        approach_data = plot_df[plot_df['Approach'] == approach].sort_values('k')
        plt.plot(approach_data['k'], approach_data['Recall'], 
                marker='o', markersize=8, linewidth=2.5,
                label=approach, color=colors(i))
    
    plt.xlabel('k (Top-k)', fontsize=14)
    plt.ylabel('Recall@k [%]', fontsize=14)
    plt.title('Recall Growth Curve - How searchable are correct tables?', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(k_values)
    plt.ylim(bottom=0, top=100)
    
    plt.tight_layout()
    plt.savefig(results_folder / "recall_growth_curve.png", dpi=300, bbox_inches='tight')
    # plt.close()
    
    print(f"############## Finished recall growth curve")


def create_model_leaderboard(df: pd.DataFrame, results_folder: Path):
    """
    Create a grouped bar chart showing MRR@10_mean per dataset
    X-axis: dataset
    Y-axis: MRR@10_mean
    Grouped by: Approach
    """
    print(f"############## Started model leaderboard")
    
    # Check if MRR@10 column exists, if not try MRR@1 or just MRR
    metric_to_use = "MRR@10"
    if "MRR@10_mean" not in df.columns and "MRR@10" not in df.columns:
        if "MRR@1_mean" in df.columns or "MRR@1" in df.columns:
            print("Warning: MRR@10 not found, using MRR@1 instead")
            metric_to_use = "MRR@1"
        elif "MRR_mean" in df.columns or "MRR" in df.columns:
            print("Warning: MRR@10 not found, using MRR instead")
            metric_to_use = "MRR"
        else:
            print("Error: No MRR metric found in data")
            return
    
    # Collect MRR data per dataset (not aggregated)
    data = plot_utils.collect_data_for_plotting(df=df, metric=metric_to_use, is_aggregated=False)
    
    # Convert from 0-1 range to 0-100 percentage
    data['Performance'] = data['Performance'] * 100
    
    fig = plt.figure(figsize=(14, 8))
    sns.set_theme(font_scale=1.3)
    
    # Create grouped bar chart
    ax = sns.barplot(x='Dataset', 
                    y='Performance', 
                    data=data, 
                    hue='Approach', 
                    palette='Set1', 
                    width=0.7)
    
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    
    # Adjust layout
    fig.subplots_adjust(top=0.92, bottom=0.15, left=0.1, right=0.95)
    
    # Add title and labels
    ax.set_title('Model Leaderboard - Ranking Quality Across Datasets', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel(f'{metric_to_use} [%]', fontsize=14)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Approach')
    plt.grid(True, linestyle=':', alpha=0.6, axis='y')
    
    plt.tight_layout()
    filename = f"model_leaderboard_{metric_to_use.lower().replace('@', '@')}.png"
    plt.savefig(results_folder / filename, dpi=300, bbox_inches='tight')
    # plt.close()
    
    print(f"############## Finished model leaderboard")

