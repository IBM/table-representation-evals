import multiprocessing
import logging
import json
import gc
from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk
from benchmark_src.utils import framework, result_utils, load_benchmark
from benchmark_src.tasks import component_utils
from benchmark_src.dataset_creation.sotab import download_sotab

logger = logging.getLogger(__name__)


def load_benchmark_data(cfg):
    """
    Load the benchmark data from cache, downloading/processing it first if it isn't cached yet.

    Returns:
        label_vocab: list of all possible labels
        splits: dict with keys "train"/"test", each {table_path: (file_format, {column_index: label})}
    """
    dataset_cache_path = Path(cfg.cache_dir) / "datasets" / "column_type_annotation" / cfg.dataset_name

    if not (dataset_cache_path / "valid_data.json").exists():
        if cfg.dataset_name == "sotab":
            logger.info(f"Dataset '{cfg.dataset_name}' not found in cache, downloading/processing it now.")
            raw_datasets_dir = Path(cfg.cache_dir) / "raw_datasets" / "sotab"
            download_sotab.ensure_dataset(raw_datasets_dir=raw_datasets_dir, output_dir=dataset_cache_path)
        else:
            raise ValueError(f"Unknown column_type_annotation dataset_name '{cfg.dataset_name}', only 'sotab' is currently supported.")

    with open(dataset_cache_path / "valid_data.json") as file:
        cached_data = json.load(file)

    label_vocab = cached_data["label_vocab"]

    splits = {}
    for split_name in ["train", "test"]:
        split_data = cached_data["splits"][split_name]
        table_paths = split_data["table_paths"]
        column_labels = split_data["column_labels"]

        per_table = {}
        for table_path, file_format in table_paths.items():
            table_name = Path(table_path).name
            labels_for_table = column_labels.get(table_name, {})
            per_table[table_path] = (file_format, labels_for_table)
        splits[split_name] = per_table

    logger.info(f"Loaded CTA benchmark '{cfg.dataset_name}': "
                f"{len(label_vocab)} labels, "
                f"{len(splits['train'])} train tables, {len(splits['test'])} test tables.")


    return label_vocab, splits


@monitor_resources()
def embed_split(column_embedding_component, split_tables, split_name, project_root):
    """
    Embed every labeled column of every table in a split.

    Returns:
        X: np.ndarray [num_labeled_columns, embedding_dim]
        y: list of str labels, aligned with X
    """
    X_list = []
    y_list = []

    for i, (table_path, (file_format, labels_for_table)) in enumerate(split_tables.items()):
        if i % 50 == 0:
            logger.info(f"[{split_name}] Processing table {i+1}/{len(split_tables)}") 
        if len(labels_for_table) == 0:
            continue

        full_table_path = Path(project_root) / table_path
        df = load_benchmark.load_dataframe(full_table_path, file_format=file_format)

        column_embeddings, column_names = column_embedding_component.create_column_embeddings_for_table(df)

        if hasattr(column_embeddings, "cpu"):
            column_embeddings = column_embeddings.cpu().numpy()
        elif isinstance(column_embeddings, list):
            column_embeddings = np.array([emb.cpu().numpy() if hasattr(emb, "cpu") else emb for emb in column_embeddings])
        column_embeddings = np.asarray(column_embeddings)

        if len(column_embeddings) != len(df.columns):
            logger.error(f"[{split_name}] Skipping table {table_path}: got {len(column_embeddings)} embeddings "
                        f"for {len(df.columns)} columns.")
            continue

        if np.isnan(column_embeddings).any():
            column_embeddings = np.nan_to_num(column_embeddings, nan=0.0)
            logger.warning(f"[{split_name}] Found NaNs in embeddings for table {table_path}, converted to 0s.")

        for column_index_str, label in labels_for_table.items():
            column_index = int(column_index_str)
            if column_index >= len(column_embeddings):
                logger.error(f"[{split_name}] Table {table_path}: column_index {column_index} out of range "
                            f"({len(column_embeddings)} columns embedded), skipping.")
                continue
            X_list.append(column_embeddings[column_index])
            y_list.append(label)

        del df, column_embeddings, column_names
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    X = np.vstack(X_list) if X_list else np.empty((0, 0))
    return X, y_list


def main(cfg: DictConfig):
    logger.debug(f"Started run_column_type_annotation_benchmark")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True)

    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)

    label_vocab, splits = load_benchmark_data(cfg)

    test_case_limit = getattr(cfg.task, "test_case_limit", None)
    if test_case_limit:
        test_case_limit = int(test_case_limit)
        for split_name in splits:
            splits[split_name] = dict(list(splits[split_name].items())[:test_case_limit])
        logger.info(f"test_case_limit={test_case_limit}: using {test_case_limit} tables per split")

    column_embedding_component = embedder._load_component(
        "column_embedding_component",
        "ColumnEmbeddingComponent",
        ColumnEmbeddingInterface
    )

    _, resource_metrics_setup = component_utils.run_model_setup(
        component=column_embedding_component,
        input_table=None,
        dataset_information=None
    )

    logger.info("Embedding train split for column type annotation")
    (X_train, y_train_labels), resource_metrics_train = embed_split(
        column_embedding_component=column_embedding_component,
        split_tables=splits["train"],
        split_name="train",
        project_root=cfg.project_root,
    )

    logger.info("Embedding test split for column type annotation")
    (X_test, y_test_labels), resource_metrics_test = embed_split(
        column_embedding_component=column_embedding_component,
        split_tables=splits["test"],
        split_name="test",
        project_root=cfg.project_root,
    )

    logger.info(f"Train columns: {len(y_train_labels)}, test columns: {len(y_test_labels)}")
    assert len(y_train_labels) > 0, "No labeled training columns were embedded, cannot train classifier."
    assert len(y_test_labels) > 0, "No labeled test columns were embedded, cannot evaluate classifier."

    label_encoder = LabelEncoder()
    label_encoder.fit(label_vocab)
    y_train = label_encoder.transform(y_train_labels)
    y_test = label_encoder.transform(y_test_labels)

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    @monitor_resources()
    def fit_and_predict():
        classifier.fit(X_train, y_train)
        return classifier.predict(X_test)

    y_pred, resource_metrics_classifier = fit_and_predict()

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Column type annotation results -> macro_f1: {macro_f1:.4f}, accuracy: {accuracy:.4f}")

    result_metrics = {
        "macro_f1 (↑)": macro_f1,
        "accuracy (↑)": accuracy,
    }

    # combine the resource metrics from embedding train+test and classifier fit/predict into the "task" bucket
    resource_metrics_task = {
        "function": "column_type_annotation_inference",
        "execution_time (s)": (
            resource_metrics_train["execution_time (s)"]
            + resource_metrics_test["execution_time (s)"]
            + resource_metrics_classifier["execution_time (s)"]
        ),
    }
    for metrics in (resource_metrics_train, resource_metrics_test, resource_metrics_classifier):
        for key, value in metrics.items():
            if key not in ("function", "execution_time (s)"):
                resource_metrics_task[key] = max(resource_metrics_task.get(key, 0), value)

    save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)

    result_utils.save_results(cfg=cfg, metrics=result_metrics)
