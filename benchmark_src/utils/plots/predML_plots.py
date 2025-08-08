import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import altair as alt

from benchmark_src.utils.gather_results import aggregate_results
import benchmark_src.utils.plots.plot_utils as plot_utils

def create_binary_barplot_altair(df: pd.DataFrame, results_folder: Path, aggregated: bool, dataset_name, percentage_baseline: bool):
    print("############## Started binary classification barplot (Altair)")

    tooltip = ["Approach", "Performance"]
    if aggregated:
        group_cols = ["Approach", "Configuration", "task"]
        df = aggregate_results(df=df, grouping_columns=group_cols)
        tooltip.append("std")

    data = plot_utils.collect_data_for_plotting(df=df, metric="XGBoost_roc_auc_score (↑)", is_aggregated=aggregated)

    # Base chart
    base = (
        alt.Chart(data)
        .encode(
            x=alt.X(
                "Approach:N",
                #sort=alt.EncodingSortField(
                #    field="Performance", op="mean", order="descending"
                #),
                axis=alt.Axis(
                    title="Approach", labelAngle=45, labelOverlap=False
                ),  # Rotate and manage label overlap
            ),
            y=alt.Y(
                "Performance:Q",
                title="ROC AUC Score",
                scale=alt.Scale(domain=[0, 1]),  # Set the y-axis range from 0 to 1
            ),
            color=alt.Color("Approach:N", legend=None),
            tooltip=tooltip,
        )
        .properties(title=f"Binary Classification Performance - {dataset_name}")
    )

    # Bars
    bars = base.mark_bar()

    # Text labels on top of the bars
    text = base.mark_text(
        align='center',
        baseline='middle',
        dy=-10,  # Offset text slightly above the bar
        color='black'
    ).encode(text=alt.Text("Performance:Q", format=".2f"))

    if aggregated:
        # Error bars using rule mark
        # For error bars representing std, we can use mark_errorbar with extent='stdev' or 'stderr'
        # Alternatively, you can use mark_rule to manually define the error bars
        # In this case, we'll use mark_errorbar for a more automated approach
        error_bars = base.mark_errorbar(extent="stderr").encode(
            yError="std:Q",  # Use the standard deviation column for error bars
            opacity=alt.value(0.7),  # Adjust opacity for error bars
            color=alt.value('black')
        )


        # Combine the bar, text labels, and error bars
        chart = (bars + error_bars + text).properties(
            width=500, height=400
        )
    else:
        # Combine the bar, text labels
        chart = (bars + text).properties(
            width=500, height=400
        )

    # Save the chart to disk as HTML

    filename = f"binary_barplot_{dataset_name}"

    if percentage_baseline:
        filename += "_percent"

    save_path = results_folder / filename + ".html"
    chart.save(str(save_path))
    print(f"############## Saved Altair plot to {save_path}")

def create_binary_barplot(df: pd.DataFrame, results_folder: Path):
    print(f"############## Started binary classification barplot")
    group_cols = ["Approach", "Configuration", "task"]
    df = aggregate_results(df=df, grouping_columns=group_cols)

    data = plot_utils.collect_data_for_plotting(df=df, metric="XGBoost_roc_auc_score (↑)", is_aggregated=True)

    fig = plt.figure(figsize=(10, 8)) # Set the figure size
    sns.set_theme(font_scale=1.5) 
    plt.ylim(top=1)
    ax = sns.barplot(x='Approach', y='Performance', data=data, hue='Approach', palette='viridis', width=0.8) # Using the DataFrame

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
        text = f'{data["Performance"][i]:.2f} (± {data["std"][i]:.2f})'
        ax.text(x, y, text, ha='center', va='bottom', color='black', fontsize=10)

    # Adding a title and labels for clarity
    plt.title('Tabarena - Binary classification')
    plt.xlabel('')
    plt.ylabel('ROC AUC (↑)') 

    plt.savefig(results_folder / "binary_barchart.png")
    print(f"############## Finished binary classification barplot")

    
def create_multiclass_barplot(df: pd.DataFrame, results_folder: Path):
    print(f"############## Started multiclass classification barplot")
    group_cols = ["Approach", "Configuration", "task"]
    df = aggregate_results(df=df, grouping_columns=group_cols)

    data = plot_utils.collect_data_for_plotting(df=df, metric="XGBoost_log_loss (↓)", is_aggregated=True)

    fig = plt.figure(figsize=(13, 8)) # Set the figure size
    sns.set_theme(font_scale=1.5) 
    plt.ylim(top=1)
    ax = sns.barplot(y='Approach', x='Performance', data=data, hue='Approach', palette='viridis', width=0.8) # Using the DataFrame

    # 1. Get the current tick locations (numerical positions)
    tick_locations = ax.get_xticks()

    # 2. Set the tick locations first (fixing them)
    ax.set_xticks(tick_locations)

    # 3. Then set the tick labels with rotation and alignment
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

    # Adjust layout
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.3, right=0.95)

    # Manually add labels using ax.text()
    for i, bar in enumerate(ax.patches):  # Iterate through the bar objects (rectangles)
        # For horizontal bars, x is at the end of the bar, y is at the center
        x = bar.get_width() + 0.01 # Position at the right end of the bar
        y = bar.get_y() + bar.get_height() / 2 # Center the text vertically
        
        # Customize the text and its placement as needed
        text = f'{data["Performance"][i]:.2f}' #(± {data["std"][i]:.2f})'
        ax.text(x, y, text, ha='left', va='center', color='black', fontsize=12) # Adjust alignment for horizontal bars

    # Adding a title and labels (if not already done within the ax.set() or similar)
    ax.set_title("Tabarena - Multiclass classification")
    ax.set_xlabel("XGBoost Log loss (↓ lower is better)")
    ax.set_ylabel("Approach")

    plt.savefig(results_folder / "multiclass_barchart.png")
    print(f"############## Finished multiclass classification barplot")

    


def create_regression_barplot(df: pd.DataFrame, results_folder: Path):
    print(f"############## Started regression classification barplot")
    group_cols = ["Approach", "Configuration", "task"]
    df = aggregate_results(df=df, grouping_columns=group_cols)

    data = {"Approach": [], "Performance": [], "std": []}
    for _, row in df.iterrows():
        approach = row["Approach"]
        if "embedding_model" in row["Configuration"]:
            approach = row["Configuration"].split("=")[-1]
            if "GritLM" in approach:
                approach = "GritLM"
            elif "BAAI" in approach:
                approach = "BAAI/bge-base-en-v1.5"
            elif "granite" in approach:
                approach = "IBM/granite-embedding-30m-english"
        if "set_prios" in row["Configuration"]:
            if "True" in row["Configuration"]:
                approach = "aidb_prio"
            else:
                approach = "aidb_default"
        data["Approach"].append(approach)
        performance = row["XGBoost_rmse (↓)_mean" + "_mean"]
        data["Performance"].append(performance)
        data["std"].append(row["XGBoost_rmse (↓)_mean" + "_std"])
    data = pd.DataFrame(data)

    fig = plt.figure(figsize=(13, 8)) # Set the figure size
    sns.set_theme(font_scale=1.5) 
    plt.ylim(top=1)
    ax = sns.barplot(y='Approach', x='Performance', data=data, hue='Approach', palette='viridis', width=0.8) # Using the DataFrame

    # 1. Get the current tick locations (numerical positions)
    tick_locations = ax.get_xticks()

    # 2. Set the tick locations first (fixing them)
    ax.set_xticks(tick_locations)

    # 3. Then set the tick labels with rotation and alignment
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

    # Adjust layout
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.3, right=0.95)

    # Manually add labels using ax.text()
    for i, bar in enumerate(ax.patches):  # Iterate through the bar objects (rectangles)
        # For horizontal bars, x is at the end of the bar, y is at the center
        x = bar.get_width() + 100 # Position at the right end of the bar
        y = bar.get_y() + bar.get_height() / 2 # Center the text vertically
        
        # Customize the text and its placement as needed
        #text = f'{data["Performance"][i]:.2f}' # (± {data["std"][i]:.2f})'
        text = f'{data["Performance"][i]:.0f}' # (± {data["std"][i]:.2f})'
        ax.text(x, y, text , ha='left', va='center', color='black', fontsize=12) # Adjust alignment for horizontal bars

    # Adding a title and labels (if not already done within the ax.set() or similar)
    ax.set_title("Tabarena - Regression")
    ax.set_xlabel("XGBoost RMSE (↓ lower is better)")
    ax.set_ylabel("Approach")

    plt.savefig(results_folder / "regression_barchart.png")
    print(f"############## Finished regression barplot")


def create_nonnumeric_performance_plot(df: pd.DataFrame, task_type, metric_name, results_folder: Path, filename):
    print(f"############## Started non-numeric vs. performance lineplot")

    ############################################################
    # collect the needed data
    ############################################################
    plotting_data = {'Approach': [],
                        'Legend': [],
                        'Nonnumeric': [],
                        'Performance': [],
                        'Dataset': []}

    for row in df.iterrows():
        row_info= row[1]
        legend = row_info["Approach"] + "__" + row_info["Configuration"].replace("approach.", "")
        # add up setup and inference time
        nonnumeric = row_info["num_nonnumeric"] / row_info["num_cols"]
        # use top-1 performance 
        performance = row_info[metric_name]

        plotting_data["Approach"].append(row_info["Approach"])
        plotting_data["Legend"].append(legend)
        plotting_data["Nonnumeric"].append(nonnumeric)
        plotting_data["Performance"].append(performance)
        plotting_data["Dataset"].append(row_info["dataset"])

    plot_df = pd.DataFrame(plotting_data)

    # Assuming 'df' already contains 'Approach', 'num_nonnumeric', and 'Performance'
    # as aggregated or calculated values from your previous steps.
    # If not, you'd need to adapt the data preparation within this function or earlier.

    fig, ax = plt.subplots(figsize=(12, 7)) # Create a figure and axes

    # Create the line plot
    sns.scatterplot(
        x="Nonnumeric",
        y="Performance",
        hue="Approach",       # Color different approaches
        data=plot_df,
        marker='o',           # Add markers for each data point
        ax=ax,
        palette='tab10'
    )

    ax.set_title(f"{task_type}: Performance vs. Percentage of Non-Numeric Columns per Approach")
    ax.set_xlabel("Percentage of Non-Numeric Columns")
    ax.set_ylabel(f"Performance {metric_name}")

    ax.set_ylim(bottom=0) # Adjust y-axis limits
    if task_type == "binary":
        ax.set_ylim(bottom=0, top=1.0) # Adjust y-axis limits

    plt.grid(True, linestyle='--', alpha=0.7) # Add a grid for better readability
    plt.legend(title="Approach", bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside the plot
    plt.tight_layout()
    
    plt.savefig(results_folder / filename)
    print(f"Saved {results_folder / filename}")
    print(f"############## Finished linechart nonnumeric")


def create_col_performance_plot(df: pd.DataFrame, task_type, metric_name, results_folder: Path, filename):
    print(f"############## Started column chart")

    ############################################################
    # collect the needed data
    ############################################################
    plotting_data = {'Approach': [],
                        'Legend': [],
                        'Columns': [],
                        'Performance': [],
                        'Dataset': []}

    for row in df.iterrows():
        row_info= row[1]
        legend = row_info["Approach"] + "__" + row_info["Configuration"].replace("approach.", "")
        # add up setup and inference time
        Columns =  row_info["num_cols"]
        # use top-1 performance 
        performance = row_info[metric_name]

        plotting_data["Approach"].append(row_info["Approach"])
        plotting_data["Legend"].append(legend)
        plotting_data["Columns"].append(Columns)
        plotting_data["Performance"].append(performance)
        plotting_data["Dataset"].append(row_info["dataset"])

    plot_df = pd.DataFrame(plotting_data)

    # Assuming 'df' already contains 'Approach', 'num_nonnumeric', and 'Performance'
    # as aggregated or calculated values from your previous steps.
    # If not, you'd need to adapt the data preparation within this function or earlier.

    fig, ax = plt.subplots(figsize=(12, 7)) # Create a figure and axes

    # Create the line plot
    sns.scatterplot(
        x="Columns",
        y="Performance",
        hue="Approach",       # Color different approaches
        data=plot_df,
        marker='o',           # Add markers for each data point
        ax=ax,
        palette='tab10'
    )

    ax.set_title(f"{task_type}: Performance vs. Number of Columns per Approach")
    ax.set_xlabel("PNumber of Columns Columns")
    ax.set_ylabel(f"Performance {metric_name}")

    ax.set_ylim(bottom=0) # Adjust y-axis limits
    if task_type == "binary":
        ax.set_ylim(bottom=0, top=1.0) # Adjust y-axis limits

    plt.grid(True, linestyle='--', alpha=0.7) # Add a grid for better readability
    plt.legend(title="Approach", bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside the plot
    plt.tight_layout()
    
    plt.savefig(results_folder / filename)
    print(f"Saved {results_folder / filename}")
    print(f"############## Finished column")