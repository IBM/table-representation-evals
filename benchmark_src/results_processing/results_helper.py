from pathlib import Path
import re

def to_slug(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("/", "-")
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w-]", "", s)
    return s


def get_setup_infos(results_file: Path):
    """
    Extract setup information from result file path

        Args:
            results_file: pathlib.Path  the path to extract from

        Returns:
            str: dataset name
            str: task name
            str: configuration 
    """
    dataset_folder = results_file.parent
    task_folder = dataset_folder.parent
    configuration_folder = task_folder.parent

    return dataset_folder.name, task_folder.name, configuration_folder.name
