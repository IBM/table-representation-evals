import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plots import plot_info

def create_datatype_comparison_plot(task_df, plots_folder):
    """
    For the datasets manually selected for datatype comparison, create a barplot
    """
    ############################################################
    # collect the needed data
    ############################################################  
    
    plotting_data = {'Approach': [],
                    'Legend': [],
                    'Performance': [],
                    'Dataset': []}

    unique_datasets = task_df['dataset'].unique()    
    for dataset_name in unique_datasets:
        # Filter data for the current dataset
        dataset_df = task_df[task_df['dataset'] == dataset_name].copy()
        # drop columns with all nans (result metrics from other tasks will be nan)
        dataset_df = dataset_df.dropna(axis=1, how="all")

        if dataset_name not in plot_info.multiclass_datatype_comparison_datasets:
            continue

        for row in dataset_df.iterrows():
            row_info = row[1]
            legend = row_info["Approach"] + "__" + row_info["Configuration"].replace("approach.", "")
            # add up setup and inference time
            setup_time = row_info["model_setup---execution_time (s)"]
            inference_time = row_info["task_inference---execution_time (s)"]
            time_taken = setup_time + inference_time
            
            # TODO: adapt later for other task types
            # use XGBoost performance 
            if "XGBoost_log_loss (\u2193)_mean" in row_info:
                performance = row_info["XGBoost_log_loss (\u2193)_mean"]
            elif "XGBoost_log_loss_mean" in row_info:
                performance = row_info["XGBoost_log_loss_mean"]
            else:
                print(row_info.keys())
                raise ValueError(f"Did not find XGBoost score")
                
            plotting_data["Approach"].append(row_info["Approach"])
            plotting_data["Legend"].append(legend)
            plotting_data["Performance"].append(performance)
            plotting_data["Dataset"].append(row_info["dataset"])

    df = pd.DataFrame(plotting_data)

    #print(df)

    ############################################################
    # Create the barplot
    ############################################################
    # Create the figure and axes
    plt.figure(figsize=(12, 7))

    # Create the grouped horizontal bar chart using seaborn's barplot
    # y: column for the y-axis (datasets)
    # x: column for the bar lengths (values - log loss)
    # hue: column for grouping and legend (embedding_model)
    ax = sns.barplot(y='Dataset', x='Performance', hue='Legend', data=df, palette='viridis')

    # Add labels and title
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_xlabel('Log Loss (Lower is Better)', fontsize=12) # Emphasize that lower is better
    ax.set_title('Log Loss by Dataset and Embedding Model (Lower is Better)', fontsize=14)
    ax.legend(title='Embedding Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.yticks(fontsize=10) # No rotation needed for y-axis labels
    plt.xticks(fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for the legend
    plt.grid(axis='x', linestyle='--', alpha=0.7) # Add vertical grid for easier comparison

    #plt.savefig(results_folder / "row_similarity_datasets.png")\
    plt.savefig("multi_classification_datatype_comparison.png")