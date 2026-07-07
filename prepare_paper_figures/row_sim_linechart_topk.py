import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def create_lineplot(df: pd.DataFrame, plots_folder: Path):
    # Filter only datasets with full results (optional, as before)
    n_approaches = df['Approach'].nunique()
    n_configs = df['Configuration'].nunique()
    dataset_counts = df.groupby('dataset')[['Approach','Configuration']].nunique().reset_index()
    full_datasets = dataset_counts[
        (dataset_counts['Approach'] == n_approaches) & 
        (dataset_counts['Configuration'] == n_configs)
    ]['dataset'].tolist()
    df_filtered = df[df['dataset'].isin(full_datasets)]

    num_datasets = len(full_datasets)
    print(f"Aggregated over {num_datasets} datasets")

    # strip * from chart names
    df_filtered['chart_name'] = df_filtered['chart_name'].str.replace('*', '', regex=False)

    # Aggregate In top-k [%] across datasets
    agg_df = df_filtered.groupby(['Approach','Configuration']).agg(
        top1_mean=('In top-1 [%]','mean'),
        top1_std=('In top-1 [%]','std'),
        top3_mean=('In top-3 [%]','mean'),
        top3_std=('In top-3 [%]','std'),
        top5_mean=('In top-5 [%]','mean'),
        top5_std=('In top-5 [%]','std'),
        top10_mean=('In top-10 [%]','mean'),
        top10_std=('In top-10 [%]','std'),
        color=('color','first'),
        chart_name=('chart_name','first')
    ).reset_index()


    # ----------------------------
    # Plot
    # ----------------------------

    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes

    # x-values: the Top-k positions to plot
    x = [1, 3, 5, 10]

    # Iterate through each approach/configuration in the aggregated dataframe
    for idx, row in agg_df.iterrows():
        # y-values: mean "In top-k [%]" for this approach/config
        y = [row['top1_mean'], row['top3_mean'], row['top5_mean'], row['top10_mean']]
        # Standard deviation for shading
        y_std = [row['top1_std'], row['top3_std'], row['top5_std'], row['top10_std']]
        
        # Plot the line with markers
        ax.plot(
            x, y, 
            label=row['chart_name'],        # label for legend (or direct annotation)
            color=row['color'],             # use custom color from dataframe
            marker='o',                     # circular markers for data points
            linewidth=3,                    # slightly thicker lines
            markersize=8                    # bigger markers for visibility
        )

    # Customize axes and labels
    ax.set_xlabel("top-k retrieved", fontsize=14)
    ax.set_ylabel("Recall [%]", fontsize=14)
    #ax.set_title("Top-k Accuracy per Approach", fontsize=16, fontweight='bold')
    ax.set_xticks(x)             # ensure x-axis shows 1,3,5,10
    ax.tick_params(axis='x', labelsize=14) # make font larger of xticks
    ax.tick_params(axis='y', labelsize=14) # make font larger of xticks
    ax.set_ylim(0, 100)         # scale y-axis 0 to 100 for percentages
    ax.grid(True, linestyle='--', alpha=0.3)  # subtle grid for readability

    # Annotate last point of each line 
    for idx, row in agg_df.iterrows():
        y_last = row['top10_mean']
        ax.text(
            x[-1] + 0.5,                # slightly right of last point
            y_last, 
            f"{y_last:.1f}%",           # label with one decimal
            color=row['color'], 
            fontsize=12, 
            va='center'
        )

    # Add legend
    ax.legend(
        title="Approach",
        fontsize=14,
        title_fontsize=13,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),  # center below plot
        ncol=4,                      # number of columns (adjust as needed)
        frameon=False                # remove legend box
    )

    plt.tight_layout()  # adjust spacing
    plt.show()

    # Save to PDF
    plt.savefig(plots_folder / "row_sim_linechart_topk.pdf")
    plt.close()