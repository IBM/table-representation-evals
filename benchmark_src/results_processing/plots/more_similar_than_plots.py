import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_barplot(df: pd.DataFrame, results_folder: Path):
    print(f"############## Started similar_than barplot")

    data = {"Approach": [], "Accuracy": []}
    for _, row in df.iterrows():
        approach = row["Approach"]
        if "embedding_model" in row["Configuration"]:
            print(row["Configuration"])
            approach = row["Configuration"].split("=")[-1]
        if "set_prios" in row["Configuration"]:
            if "True" in row["Configuration"]:
                approach = "aidb_prio"
            else:
                approach = "aidb_default"
        data["Approach"].append(approach)
        data["Accuracy"].append(row["accuracy_mean"] * 100)
    data = pd.DataFrame(data)

    print(data["Accuracy"])

    fig = plt.figure(figsize=(10, 8)) # Set the figure size
    ax = sns.barplot(x='Approach', y='Accuracy', data=data, hue='Approach', palette='viridis', width=0.8) # Using the DataFrame

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
        ax.text(x, y, f'{data["Accuracy"][i]:.2f}', ha='center', va='bottom', color='black', fontsize=10)

    # Adding a title and labels for clarity
    plt.title('')
    plt.xlabel('')
    plt.ylabel('Accuracy [%]') 

    plt.savefig(results_folder / "books_bar_plot.png")
    print(f"############## Finished similar_than barplot")

    
