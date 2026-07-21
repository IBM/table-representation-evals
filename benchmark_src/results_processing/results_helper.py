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
    'accuracy (↑)': 'higher_is_better',
    'accuracy_easy': 'higher_is_better',
    'accuracy_medium': 'higher_is_better',
    'accuracy_hard': 'higher_is_better',
    # column type annotation
    'macro_f1 (↑)': 'higher_is_better',
    'In top-1 [%]': 'higher_is_better',
    'In top-3 [%]': 'higher_is_better',
    'In top-5 [%]': 'higher_is_better',
    'In top-10 [%]': 'higher_is_better',
    'MRR': 'higher_is_better',  # mean reciprocal rank
    'MAP': 'higher_is_better',  # mean average precision
    'Recall@1': 'higher_is_better',
    # nl2 schema linking
    'mean_mrr': 'higher_is_better',
    # table retrieval (TARGET-based datasets), one entry per configured top_k
    'MRR@1': 'higher_is_better',
    'MRR@3': 'higher_is_better',
    'MRR@5': 'higher_is_better',
    'MRR@10': 'higher_is_better',
    'MAP@1': 'higher_is_better',
    'MAP@3': 'higher_is_better',
    'MAP@5': 'higher_is_better',
    'MAP@10': 'higher_is_better',
    'Recall@3': 'higher_is_better',
    'Recall@5': 'higher_is_better',
    'Recall@10': 'higher_is_better',
    'Precision@1': 'higher_is_better',
    'Precision@3': 'higher_is_better',
    'Precision@5': 'higher_is_better',
    'Precision@10': 'higher_is_better',
    # table shuffling / triplet metrics
    'TripletAccuracy': 'higher_is_better',
    'Triplet Silhouette Score': 'higher_is_better',
    'Bounded Contrastive Score': 'lower_is_better',
    'TextualBias_pearson': 'lower_is_better',
    # table type detection (TTD) classification metrics
    'XGBoost_accuracy (↑)': 'higher_is_better',
    'KNeighbors_accuracy (↑)': 'higher_is_better',
    "MLP_accuracy (↑)": 'higher_is_better',
    'XGBoost_f1_macro (↑)': 'higher_is_better',
    'KNeighbors_f1_macro (↑)': 'higher_is_better',
    "MLP_f1_macro (↑)": 'higher_is_better',
    'XGBoost_f1_micro (↑)': 'higher_is_better',
    'KNeighbors_f1_micro (↑)': 'higher_is_better',
    "MLP_f1_micro (↑)": 'higher_is_better',
}


def to_slug(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("/", "-")
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w-]", "", s)
    return s


def get_setup_infos(results_file: Path):
    """
    Extract setup information from result file path.

    Handles two path structures:
      approach/[param_slug/]task/dataset/results.json
      approach/[param_slug/]task/task_param_slug/dataset/results.json

    The task_param_slug level (if present) is detected by the presence of '=' in
    the folder name and is appended to configuration so variants remain distinct.

        Args:
            results_file: pathlib.Path  the path to extract from

        Returns:
            str: dataset name
            str: task name
            str: configuration
    """
    dataset_folder = results_file.parent
    task_or_slug_folder = dataset_folder.parent

    if "=" in task_or_slug_folder.name:
        # task_param_slug level is present (e.g. query_mode=full_nl)
        task_param_slug = task_or_slug_folder.name
        task_folder = task_or_slug_folder.parent
        param_slug_folder = task_folder.parent
        if "=" in param_slug_folder.name:
            configuration = f"{param_slug_folder.name},{task_param_slug}"
        else:
            configuration = task_param_slug
    else:
        task_folder = task_or_slug_folder
        configuration_folder = task_folder.parent
        configuration = configuration_folder.name

    return dataset_folder.name, task_folder.name, configuration
