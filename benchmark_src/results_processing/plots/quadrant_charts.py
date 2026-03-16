import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from benchmark_src.results_processing import aggregate


def quadrant_plot_aggregated_over_datasets(filename, task_df, task_type, results_folder: Path, results_column: str, title: str, marker_labels: bool, inference_only: bool=False):
    ############################################################
    # aggregate over all datasets to get one performance value
    ############################################################
    group_cols = ["Approach", "Configuration", "task"]
    aggregated_over_datasets = aggregate.aggregate_results(df=task_df, grouping_columns=group_cols)

    ############################################################
    # collect the needed data
    ############################################################
    plotting_data = {'Approach': [],
                        'Legend': [],
                        'Time': [],
                        'Performance': []}
    
    for row in aggregated_over_datasets.iterrows():
        row_info = row[1]
        legend = row_info["Approach"] + "__" + row_info["Configuration"].replace("approach.", "")
        # add up setup and inference time
        setup_time = row_info["model_setup---execution_time (s)_mean"]
        inference_time = row_info["task_inference---execution_time (s)_mean"]

        if inference_only:
            time_taken = inference_time
        else:
            time_taken = setup_time + inference_time

        performance = row_info[results_column + "_mean"]

        plotting_data["Approach"].append(row_info["Approach"])
        plotting_data["Legend"].append(legend)
        plotting_data["Time"].append(time_taken)
        plotting_data["Performance"].append(performance)

    plot_df = pd.DataFrame(plotting_data)

    ############################################################
    # Set markers and colors
    ############################################################

    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'x', 'h', 'd', 'v']  # Example marker shapes
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'lime', 'pink', 'darkslategray'] # Example colors

    assert len(markers) >= len(plot_df)
    assert len(colors) >= len(plot_df)

    ############################################################
    # Figure size
    ############################################################
    plt.figure(figsize=(14,8)) # Adjust figure size as needed

    ############################################################
    # Create the Scatter Plot
    ############################################################
    for i, row in plot_df.iterrows():
        plt.scatter(row['Time'], row['Performance'],
                    marker=markers[i], # Assign unique marker
                    color=colors[i], # Assign unique color
                    s=150, alpha=0.7,
                    label=row['Legend']) # Use Approach name as label for legend
        
    ############################################################
    # set axis to start at zero
    ############################################################
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    if task_type == "binary":
        plt.ylim(top=1)

    ############################################################
    # Add lines for quadrants
    ############################################################
    time_threshold = plot_df['Time'].mean()
    time_max = plot_df['Time'].max()
    time_min = plot_df['Time'].min()
    performance_threshold = plot_df['Performance'].mean()
    #avg_performance = df['Performance Reached (%)'].mean()

    # Add horizontal and vertical lines for quadrants
    plt.axvline(x=time_max/2, color='black', linestyle='-', linewidth=2)

    if task_type == "binary":
        plt.axhline(y=0.5, color='black', linestyle='-', linewidth=2)
    #plt.axvline(x=time_threshold, color='red', linestyle='--', linewidth=2, label='Average Time Taken')
    #plt.axhline(y=performance_threshold, color='blue', linestyle='--', linewidth=2, label='Average Performance')

    ############################################################
    # Annotate quadrants
    ############################################################

    # Annotate each quadrant dynamically based on thresholds and plot limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    ############################################################
    # Add labels and title
    ############################################################
    plt.xlabel('Time Taken (s)')
    plt.ylabel(f'Performance: {results_column.replace("mean", "")}')

    plt.title(title, fontsize=14, fontweight='bold', pad=30)
    # Add a subtitle using suptitle()
    #plt.suptitle(f'(Mean across {len()} datasets)', fontsize=12, color='black', y=0.915) # Adjust y for position

    ############################################################
    # Add annotations for each data point
    ############################################################
    if marker_labels:
        for i, row in plot_df.iterrows():
            plt.annotate(row['Approach'], (row['Time'] + 1, row['Performance']), fontsize=9)

    ############################################################
    # Add legend
    ############################################################
    plt.legend(bbox_to_anchor=(0.5, -0.20), loc='upper center', ncol=2, borderaxespad=0.)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(results_folder / filename)