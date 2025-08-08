import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

from gather_results import aggregate_results
import benchmark_src.utils.plots.plot_utils as plot_utils

def quadrant_plot_aggregated_time(task_df, results_folder: Path):
    ############################################################
    # aggregate over all datasets to get one performance value
    ############################################################
    group_cols = ["Approach", "Configuration", "task"]
    aggregated_over_datasets = aggregate_results(df=task_df, grouping_columns=group_cols)


    ############################################################
    # collect the needed data
    ############################################################
    plot_df = plot_utils.collect_data_for_plotting(df=aggregated_over_datasets, metric="In top-1 [%]", is_aggregated=True, resources=True) 

    ############################################################
    # Set markers and colors
    ############################################################

    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'h']  # Example marker shapes
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'lime'] # Example colors

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
        plt.scatter(row['InferenceTime'], row['Performance'],
                    marker=markers[i], # Assign unique marker
                    color=colors[i], # Assign unique color
                    s=150, alpha=0.7,
                    label=row['Approach']) # Use Approach name as label for legend
        
    ############################################################
    # set axis to start at zero
    ############################################################
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.ylim(top=100)

    ############################################################
    # Add lines for quadrants
    ############################################################
    time_threshold = plot_df['InferenceTime'].mean()
    time_max = plot_df['InferenceTime'].max()
    time_min = plot_df['InferenceTime'].min()
    performance_threshold = plot_df['Performance'].mean()
    max_performance = plot_df['Performance'].max()
    #avg_performance = df['Performance Reached (%)'].mean()

    # Add horizontal and vertical lines for quadrants
    plt.axvline(x=time_max/2, color='black', linestyle='-', linewidth=2)
    plt.axhline(y=50, color='black', linestyle='-', linewidth=2)
    #plt.axvline(x=time_threshold, color='red', linestyle='--', linewidth=2, label='Average Time Taken')
    #plt.axhline(y=performance_threshold, color='blue', linestyle='--', linewidth=2, label='Average Performance')

    ############################################################
    # Annotate quadrants
    ############################################################

    # Annotate each quadrant dynamically based on thresholds and plot limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # Define text offsets
    text_x_offset = (x_max - x_min) * 0.02 # 2% of total x-range
    text_y_offset = (y_max - y_min) * 0.02 # 2% of total y-range

    # Quadrant 1: High Performance, Lower Time (Top-Left) - Desirable (Green)
    plt.text(x_min + text_x_offset, 90, 'High Performance\nLower Time', color='gray', fontsize=11, ha='left', va='bottom')
    # Quadrant 2: High Performance, Higher Time (Top-Right) - Less Desirable (Orange)
    plt.text(x_max / 2 + text_x_offset, 90, 'High Performance\nHigher Time', color='gray', fontsize=11, ha='left', va='bottom')
    # Quadrant 3: Low Performance, Lower Time (Bottom-Left) - Less Desirable (Orange)
    plt.text(x_min + text_x_offset, y_min + text_y_offset, 'Low Performance\nLower Time', color='gray', fontsize=11, ha='left', va='bottom')
    # Quadrant 4: Low Performance, Higher Time (Bottom-Right) - Undesirable (Red)
    plt.text(x_max / 2 + text_x_offset, y_min + text_y_offset, 'Low Performance\nHigher Time', color='gray', fontsize=11, ha='left', va='bottom')


    ############################################################
    # Add labels and title
    ############################################################
    plt.xlabel('Time Taken (s)')
    plt.ylabel('Performance: In top-1 [%]')

    plt.title('Row Similarity Search: Time Taken (Inference) vs. Performance Reached', fontsize=24, fontweight='bold', pad=30)
    # Add a subtitle using suptitle()
    plt.suptitle('(Mean across 9 datasets)', fontsize=18, color='black', y=0.86) # Adjust y for position

    ############################################################
    # Add annotations for each data point
    ############################################################
    for i, row in plot_df.iterrows():
        # TODO: time + needs to be adapted dynamically (%)
        plt.annotate(row['Approach'], (row['InferenceTime'] + 5, row['Performance']), fontsize=12)

    ############################################################
    # Add legend
    ############################################################
    # plt.legend(bbox_to_anchor=(0.5, -0.18), loc='upper center', ncol=2, borderaxespad=0.)
    # plt.grid(True, linestyle=':', alpha=0.6)
    # plt.tight_layout(rect=[0, 0.02, 1, 1])

    plt.savefig(results_folder / "row_similarity_datasets_aggregated_inference.png")

def quadrant_plot_aggregated_cpu(task_df, results_folder: Path):
    ############################################################
    # aggregate over all datasets to get one performance value
    ############################################################
    group_cols = ["Approach", "Configuration", "task"]
    aggregated_over_datasets = aggregate_results(df=task_df, grouping_columns=group_cols)

    ############################################################
    # collect the needed data
    ############################################################

    plot_df = plot_utils.collect_data_for_plotting(df=aggregated_over_datasets, metric="In top-1 [%]", is_aggregated=True, resources=True) 

    ############################################################
    # Set markers and colors
    ############################################################

    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'h']  # Example marker shapes
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'lime'] # Example colors

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
        plt.scatter(row['InferenceCPU'], row['Performance'],
                    marker=markers[i], # Assign unique marker
                    color=colors[i], # Assign unique color
                    s=150, alpha=0.7,
                    label=row['Approach']) # Use Approach name as label for legend
        
    ############################################################
    # set axis to start at zero
    ############################################################
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.ylim(top=100)

    ############################################################
    # Add lines for quadrants
    ############################################################
    time_threshold = plot_df['InferenceCPU'].mean()
    time_max = plot_df['InferenceCPU'].max()
    time_min = plot_df['InferenceCPU'].min()
    performance_threshold = plot_df['Performance'].mean()
    max_performance = plot_df['Performance'].max()
    #avg_performance = df['Performance Reached (%)'].mean()

    # Add horizontal and vertical lines for quadrants
    plt.axvline(x=time_max/2, color='black', linestyle='-', linewidth=2)
    plt.axhline(y=50, color='black', linestyle='-', linewidth=2)
    #plt.axvline(x=time_threshold, color='red', linestyle='--', linewidth=2, label='Average Time Taken')
    #plt.axhline(y=performance_threshold, color='blue', linestyle='--', linewidth=2, label='Average Performance')

    ############################################################
    # Add labels and title
    ############################################################
    plt.xlabel('CPU (%)')
    plt.ylabel('Performance: In top-1 [%]')

    plt.title('Row Similarity Search:  CPU (Peak during Inference) vs. Performance Reached', fontsize=24, fontweight='bold', pad=30)
    # Add a subtitle using suptitle()
    plt.suptitle('(Mean across 9 datasets)', fontsize=18, color='black', y=0.86) # Adjust y for position

    ############################################################
    # Add annotations for each data point
    ############################################################
    for i, row in plot_df.iterrows():
        # TODO: time + needs to be adapted dynamically (%)
        plt.annotate(row['Approach'], (row['InferenceCPU'] + 20, row['Performance']), fontsize=12)

    ############################################################
    # Add legend
    ############################################################
    # plt.legend(bbox_to_anchor=(0.5, -0.18), loc='upper center', ncol=2, borderaxespad=0.)
    # plt.grid(True, linestyle=':', alpha=0.6)
    # plt.tight_layout(rect=[0, 0.02, 1, 1])

    plt.savefig(results_folder / "row_similarity_datasets_aggregated_cpu_inference.png")

def plot_per_dataset(task_df, results_folder: Path):
    """
    For each dataset, create an individual plot
    """
        
    ############################################################
    # Group per dataset
    ############################################################
    unique_datasets = task_df['dataset'].unique()

    for dataset_name in unique_datasets:
        # Filter data for the current dataset
        dataset_df = task_df[task_df['dataset'] == dataset_name].copy()
        # drop columns with all nans (result metrics from other tasks will be nan)
        dataset_df = dataset_df.dropna(axis=1, how="all")

        ############################################################
        # collect the needed data
        ############################################################
        plotting_data = {'Approach': [],
                        'Legend': [],
                        'Time': [],
                        'Performance': []}
        
        for row in dataset_df.iterrows():
            row_info = row[1]
            legend = row_info["Approach"] + "__" + row_info["Configuration"].replace("approach.", "")
            # add up setup and inference time
            setup_time = row_info["model_setup---execution_time (s)"]
            inference_time = row_info["task_inference---execution_time (s)"]
            #time_taken = setup_time + inference_time
            time_taken = inference_time
            # use top-1 performance 
            performance = row_info["In top-1 [%]_mean"]

            plotting_data["Approach"].append(row_info["Approach"])
            plotting_data["Legend"].append(legend)
            plotting_data["Time"].append(time_taken)
            plotting_data["Performance"].append(performance)

        plot_df = pd.DataFrame(plotting_data)

        ############################################################
        # Set markers and colors
        ############################################################

        markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'h']  # Example marker shapes
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'lime'] # Example colors

        assert len(markers) >= len(plot_df)
        assert len(colors) >= len(plot_df)

        ############################################################
        # Figure size
        ############################################################
        plt.figure(figsize=(14, 8)) # Adjust figure size as needed

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
        plt.ylim(top=100)

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
        plt.axhline(y=50, color='black', linestyle='-', linewidth=2)
        #plt.axvline(x=time_threshold, color='red', linestyle='--', linewidth=2, label='Average Time Taken')
        #plt.axhline(y=performance_threshold, color='blue', linestyle='--', linewidth=2, label='Average Performance')

        ############################################################
        # Annotate quadrants
        ############################################################

        # # Annotate each quadrant dynamically based on thresholds and plot limits
        # x_min, x_max = plt.xlim()
        # y_min, y_max = plt.ylim()

        # # Define text offsets
        # text_x_offset = (x_max - x_min) * 0.02 # 2% of total x-range
        # text_y_offset = (y_max - y_min) * 0.02 # 2% of total y-range

        # # Quadrant 1: High Performance, Lower Time (Top-Left) - Desirable (Green)
        # plt.text(x_min + text_x_offset, performance_threshold + text_y_offset, 'High Performance\nLower Time', color='gray', fontsize=11, ha='left', va='bottom')
        # # Quadrant 2: High Performance, Higher Time (Top-Right) - Less Desirable (Orange)
        # plt.text(x_max / 2 + text_x_offset, performance_threshold + text_y_offset, 'High Performance\nHigher Time', color='gray', fontsize=11, ha='left', va='bottom')
        # # Quadrant 3: Low Performance, Lower Time (Bottom-Left) - Less Desirable (Orange)
        # plt.text(x_min + text_x_offset, y_min + text_y_offset, 'Low Performance\nLower Time', color='gray', fontsize=11, ha='left', va='bottom')
        # # Quadrant 4: Low Performance, Higher Time (Bottom-Right) - Undesirable (Red)
        # plt.text(x_max / 2 + + text_x_offset, y_min + text_y_offset, 'Low Performance - Higher Time', color='gray', fontsize=11, ha='left', va='bottom')


        ############################################################
        # Add labels and title
        ############################################################
        plt.xlabel('Time Taken (s)')
        plt.ylabel('Performance: In top-1 [%]')

        plt.title(f'Row Similarity Search: {dataset_name}', fontsize=14, fontweight='bold', pad=30)
        # Add a subtitle using suptitle()
        plt.suptitle('Time Taken (Inference) vs. Performance Reached)', fontsize=12, color='black', y=0.915) # Adjust y for position

        ############################################################
        # Add annotations for each data point
        ############################################################
        for i, row in plot_df.iterrows():
            plt.annotate(row['Approach'], (row['Time'] + 1, row['Performance']), fontsize=9)

        ############################################################
        # Add legend
        ############################################################
        plt.legend(bbox_to_anchor=(0.5, -0.20), loc='upper center', ncol=2, borderaxespad=0.)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout(rect=[0, 0.1, 1, 1])

        plt.savefig(results_folder / f"row_similarity_{dataset_name}_inference.png")


def plot_point_per_dataset(task_df, results_folder: Path):

    ############################################################
    # collect the needed data
    ############################################################
    plotting_data = {'Approach': [],
                        'Legend': [],
                        'Time': [],
                        'Performance': [],
                        'Dataset': []}

    for row in task_df.iterrows():
        row_info = row[1]
        legend = row_info["Approach"] + "__" + row_info["Configuration"].replace("approach.", "")
        # add up setup and inference time
        setup_time = row_info["model_setup---execution_time (s)"]
        inference_time = row_info["task_inference---execution_time (s)"]
        #time_taken = setup_time + inference_time
        time_taken = inference_time
        # use top-1 performance 
        performance = row_info["In top-1 [%]_mean"]

        plotting_data["Approach"].append(row_info["Approach"])
        plotting_data["Legend"].append(legend)
        plotting_data["Time"].append(time_taken)
        plotting_data["Performance"].append(performance)
        plotting_data["Dataset"].append(row_info["dataset"])

    plot_df = pd.DataFrame(plotting_data)

    ############################################################
    # Set markers and colors
    ############################################################

    # Define unique markers for each dataset
    unique_datasets = plot_df['Dataset'].unique()
    # Example marker shapes (ensure enough unique markers for all approaches)
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'h', 'v', '<', '>', 'p']
    assert len(markers) >= len(unique_datasets)
    marker_map = {x: markers[i] for i, x in enumerate(unique_datasets)}

    # Define colors for each approach
    unique_approaches = plot_df['Legend'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_approaches))  # Use a colormap to get distinct colors
    color_map = {x: colors(i) for i, x in enumerate(unique_approaches)}
    

    ############################################################
    # Figure size
    ############################################################
    plt.figure(figsize=(14, 8)) # Adjust figure size as needed

    ############################################################
    # Create the Scatter Plot
    ############################################################
    # Plot each data point with marker based on Approach, color based on Dataset
    scatter_points = []  # To store scatter plot objects for the dataset legend
    for i, row in plot_df.iterrows():
        approach = row['Legend']
        dataset = row['Dataset']
        point = plt.scatter(row['Time'], row['Performance'],
                            marker=marker_map[dataset],  # Marker for Approach
                            color=color_map[approach],  # Color for Dataset
                            s=150, alpha=0.7,
                            # We will handle legends separately
                        )
        scatter_points.append(point)

    ############################################################
    # set axis to start at zero
    ############################################################
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.ylim(top=100)


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
    plt.axhline(y=50, color='black', linestyle='-', linewidth=2)
    #plt.axvline(x=time_threshold, color='red', linestyle='--', linewidth=2, label='Average Time Taken')
    #plt.axhline(y=performance_threshold, color='blue', linestyle='--', linewidth=2, label='Average Performance')

    ############################################################
    # Annotate quadrants
    ############################################################

    # Annotate each quadrant dynamically based on thresholds and plot limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # Define text offsets
    text_x_offset = (x_max - x_min) * 0.02 # 2% of total x-range
    text_y_offset = (y_max - y_min) * 0.02 # 2% of total y-range

    # Quadrant 1: High Performance, Lower Time (Top-Left) - Desirable (Green)
    plt.text(x_min + text_x_offset, performance_threshold + text_y_offset, 'High Performance\nLower Time', color='gray', fontsize=11, ha='left', va='bottom')
    # Quadrant 2: High Performance, Higher Time (Top-Right) - Less Desirable (Orange)
    plt.text(x_max / 2 + text_x_offset, performance_threshold + text_y_offset, 'High Performance\nHigher Time', color='gray', fontsize=11, ha='left', va='bottom')
    # Quadrant 3: Low Performance, Lower Time (Bottom-Left) - Less Desirable (Orange)
    plt.text(x_min + text_x_offset, y_min + text_y_offset, 'Low Performance\nLower Time', color='gray', fontsize=11, ha='left', va='bottom')
    # Quadrant 4: Low Performance, Higher Time (Bottom-Right) - Undesirable (Red)
    plt.text(x_max / 2 + + text_x_offset, y_min + text_y_offset, 'Low Performance - Higher Time', color='gray', fontsize=11, ha='left', va='bottom')

    ############################################################
    # Add labels and title
    ############################################################
    plt.xlabel('Time Taken (s)')
    plt.ylabel('Performance: In top-1 [%]')

    plt.title('Row Similarity Search: Time Taken (Inference) vs. Performance Reached', fontsize=14, fontweight='bold', pad=30)
    # Add a subtitle using suptitle()
    #plt.suptitle('(Mean across 9 datasets)', fontsize=12, color='black', y=0.915) # Adjust y for position

    ############################################################
    # Add annotations for each data point
    ############################################################
    #for i, row in plot_df.iterrows():
    #    plt.annotate(row['Approach'], (row['Time'] + 0.7, row['Performance']), fontsize=9)



    # --- Create the legends ---

    # Legend 1: For Approaches (based on color)
    # Create dummy handles for the color legend (points with a generic marker)
    color_legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color, markersize=10,
                                        label=x)
                            for x, color in color_map.items()]

    legend1 = plt.legend(handles=color_legend_handles,
                        title="Approach", ncol=1,
                        loc='upper left', 
                        bbox_to_anchor=(0.3, -0.12),
                        borderaxespad=0.)

    # Add the first legend to the axes
    plt.gca().add_artist(legend1)

    # Legend 2: For Datasets (based on marker)
    # Create dummy handles for the marker legend (points with a generic color)
    marker_legend_handles = [plt.Line2D([0], [0], marker=marker, color='w',
                                        markerfacecolor='gray', markersize=10,
                                        label=x)
                            for x, marker in marker_map.items()]

    legend2 = plt.legend(handles=marker_legend_handles,
                        bbox_to_anchor=(0.2, -0.12),
                        title="Dataset",
                        loc='upper right',
                        borderaxespad=0.)

    # Adjust plot area to make space for the legends
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(results_folder / "row_similarity_datasets_inference.png")


def create_barplot(df: pd.DataFrame, results_folder: Path):
    print(f"############## Started row sim barplot")
    group_cols = ["Approach", "Configuration", "task"]
    df = aggregate_results(df=df, grouping_columns=group_cols)

    print(df)

    for top_k in [1, 3, 5, 10]:

        data = plot_utils.collect_data_for_plotting(df=df, metric=f"In top-{top_k} [%]", is_aggregated=True)

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
        ax.set_title(f"Finding the most similar row - among the top-{top_k} retrieved (Aggregated over 9 datasets)")
        ax.set_xlabel("Approach")
        ax.set_ylabel("Accuracy [%]")

        plt.savefig(results_folder / f"aggregated_barchart_top_{top_k}.png")
    print(f"############## Finished row sim barplot")


def create_barplot_datasets(df: pd.DataFrame, results_folder: Path):
    print(f"############## Started row sim barplot datasets")

    for top_k in [1, 3, 5, 10]:
        
        data = plot_utils.collect_data_for_plotting(df=df, metric=f"In top-{top_k} [%]", is_aggregated=False)

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
        ax.set_title(f"Finding the most similar row - among the top-{top_k} retrieved")
        ax.set_xlabel("Approach")
        ax.set_ylabel("Accuracy [%]")

        ############################################################
        # Add legend
        ############################################################
        plt.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=2, borderaxespad=0.)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout(rect=[0, 0.001, 1, 1])

        plt.savefig(results_folder / f"barchart_top_{top_k}.png")
    print(f"############## Finished row sim barplot datasets")
