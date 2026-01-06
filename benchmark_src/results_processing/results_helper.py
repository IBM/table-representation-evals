from pathlib import Path
import re

performance_cols = {
    # RMSE
    'XGBoost_rmse (↓)': 'lower_is_better',
    'KNeighbors_rmse (↓)': 'lower_is_better',
    'LinearRegression_rmse (↓)': 'lower_is_better',
    "MLP_rmse (↓)": 'lower_is_better',
    # roc_auc
    'XGBoost_roc_auc_score (↑)': 'higher_is_better',
    'KNeighbors_roc_auc_score (↑)': 'higher_is_better',
    "MLP_roc_auc_score (↑)": 'higher_is_better',
    # log_loss
    'XGBoost_log_loss (↓)': 'lower_is_better',
    'KNeighbors_log_loss (↓)': 'lower_is_better',
    "MLP_log_loss (↓)": 'lower_is_better',
    # accuracy
    'accuracy': 'higher_is_better',
    'accuracy_easy': 'higher_is_better',
    'accuracy_medium': 'higher_is_better',
    'accuracy_hard': 'higher_is_better',
    'In top-1 [%]': 'higher_is_better',
    'In top-3 [%]': 'higher_is_better',
    'In top-5 [%]': 'higher_is_better',
    'In top-10 [%]': 'higher_is_better',
    'MRR': 'higher_is_better', # mean reciprocal rank
    'Recall@1': 'higher_is_better'
}


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
