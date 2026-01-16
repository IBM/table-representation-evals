import os
from pathlib import Path
from bs4 import BeautifulSoup

def render_html_page(plots_folder: str, files: list[str], output_file: str, title: str, plots_per_row: int):
    # === START HTML PAGE ===
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        .grid-container {{
            display: grid;
            grid-template-columns: repeat({plots_per_row}, 1fr);
            gap: 20px;
        }}
        .grid-item {{
            width: 100%;
            height: 500px;
            border: 1px solid #ccc;
        }}
        iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
    </style>
    </head>
    <body>
    <h1>{title}</h1>
    <div class="grid-container">
    """

    # === ADD PLOTS USING IFRAME ===
    for file in files:
        full_path = file
        # use relative path for iframe src
        iframe_src = file.replace("\\", "/")  # for Windows
        html += f'<div class="grid-item"><iframe src="{iframe_src}"></iframe></div>\n'

    # === END HTML PAGE ===
    html += """
    </div>
    </body>
    </html>
    """

    # === SAVE OUTPUT ===
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Created {output_file} with {len(files)} plots!")


def create_predml_plots_html_view(results_folder):
    # === CONFIG ===
    assert results_folder.exists(), f"Could not find results folder at {results_folder}"

    predML_plots_folder = results_folder / "plots" / "predictive_ml"

    # write two different HTML files, one for MLP plots and one for XGBoost plots
    configs = {
        "mlp": {
            "plots_folder": Path("MLP_model"),
            "output_file": "all_plots_mlp.html",
            "title": "MLP Plots"
        },
        "xgboost": {
            "plots_folder": Path("XGBoost_model"),
            "output_file": "all_plots_xgboost.html",
            "title": "XGBoost Plots"
        }
    }

    for config_name, config in configs.items():
        plots_folder = config["plots_folder"]
        output_file = config["output_file"]
        title = config["title"]

        #assert plots_folder.exists(), f"Could not find plots folder at {plots_folder}"

        plots_per_row = 2                  # number of plots per row in the grid

        # === GET HTML FILES ===
        files = sorted([f for f in os.listdir(predML_plots_folder / plots_folder) if f.endswith(".html")])

        # === RENDER HTML PAGE ===
        render_html_page(
            plots_folder=plots_folder,
            files=files,        
            output_file=predML_plots_folder / plots_folder / output_file,
            title=title,
            plots_per_row=plots_per_row
        )

    # create an overview with mlp plots on the left and xgboost plots on the right

    # folders
    mlp_plots_folder = predML_plots_folder / configs["mlp"]["plots_folder"]
    xgboost_plots_folder = predML_plots_folder / configs["xgboost"]["plots_folder"]

    assert mlp_plots_folder.exists()
    assert xgboost_plots_folder.exists()

    # list files - exclude aggregated overview files
    mlp_files = sorted([f for f in os.listdir(mlp_plots_folder) 
                        if f.endswith(".html") 
                        and not f.endswith("_percent.html")
                        and not f.startswith("all_plots")])  

    xgboost_files = sorted([f for f in os.listdir(xgboost_plots_folder) 
                            if f.endswith(".html")
                            and not f.endswith("_percent.html")
                            and not f.startswith("all_plots")]) 
    
    # remove prefixes to find common plots
    mlp_file_suffixes = [f[len("MLP_"):] for f in mlp_files]
    xgboost_file_suffixes = [f[len("XGBoost_"):] for f in xgboost_files]

    # find common plots
    common_files = sorted(list(set(mlp_file_suffixes).intersection(set(xgboost_file_suffixes))))

    # build combined list with **full paths**
    combined_files = []
    for suffix in common_files:
        mlp_full_path = os.path.join(configs["mlp"]["plots_folder"], f"MLP_{suffix}")
        xgb_full_path = os.path.join(configs["xgboost"]["plots_folder"], f"XGBoost_{suffix}")
        combined_files.append(mlp_full_path)
        combined_files.append(xgb_full_path)

    # render the combined html page
    output_file = predML_plots_folder / "all_plots_mlp_xgboost_comparison.html"
    title = "MLP vs XGBoost Plots Comparison"
    plots_per_row = 2

    render_html_page(
        plots_folder=".",  # not used
        files=combined_files,
        output_file=output_file,
        title=title,
        plots_per_row=plots_per_row
    )
