import pathlib
import logging
import requests
import io
import os
import zipfile
import tarfile
import json

logger = logging.getLogger(__name__)

def logger_init():
    logging.basicConfig(format='{levelname} - {name} - {asctime} {message}', 
                    level=logging.INFO,
                    style="{",
                    datefmt="%Y-%m-%d %H:%M")
    

def get_filename_from_url(url):
       return os.path.basename(url)

def create_statistics(dataset_name, input_table_df, num_testcases, save_path, primary_key_column):
    statistics_dict = {"dataset_name": dataset_name,
                       "input_table_num_rows": len(input_table_df),
                       "input_talbe_num_cols": len(input_table_df.columns),
                       "primary_key_column": primary_key_column
                       }
    
    # get datatypes of columns
    statistics_dict["datatypes"] = input_table_df.dtypes.astype("str").to_dict()

    # compute table sparsity
    num_empty_cells = float((input_table_df.isnull().sum()).sum())
    sparsity = float(num_empty_cells / input_table_df.size)
    statistics_dict["num_empty_cells"] = num_empty_cells
    statistics_dict["sparsity"] = sparsity
    statistics_dict["num_testcases"] = num_testcases


    with open(save_path / "dataset_information.json", "w") as file:
        json.dump(statistics_dict, file, indent=2)

def download_url(url: str, path: pathlib.Path, *, unzip: bool = False, untar: bool = False) -> None:
    """Download the given URL.

    Args:
        url: The URL to download.
        path: The file or directory path.
        unzip: Whether to unzip the downloaded data.
        untar: Whether to untar the downloaded data.
    """
    if unzip and untar:
        raise AssertionError("cannot unzip and untar at the same time")
    logger.debug(f"Download {url}")
    response = requests.get(url)
    if unzip:
        zip_data = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_data, "r") as zip_ref:
            zip_ref.extractall(path)
    elif untar:
        tar_data = io.BytesIO(response.content)
        with tarfile.open(fileobj=tar_data, mode="r:gz") as tar_ref:
            tar_ref.extractall(path)
    else:
        file_name = get_filename_from_url(url)

        with open(path / file_name, "wb") as file:
            file.write(response.content)