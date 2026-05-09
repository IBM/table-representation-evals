import hashlib
import logging
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import sklearn.impute
import sklearn.metrics
import sklearn.neighbors
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import xgboost as xgb
from omegaconf import DictConfig
from tqdm import tqdm

from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface
from benchmark_src.dataset_creation.table_type_detection.load_ttd import load_ttd_split
from benchmark_src.tasks import component_utils
from benchmark_src.utils import result_utils
from benchmark_src.utils.framework import get_approach_class
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk

logger = logging.getLogger(__name__)

TTD_DATASET_NAME = "wdc_schema_org"


def get_embedder(cfg: DictConfig) -> tuple[TableEmbeddingInterface, dict[str, Any]]:
    approach_cls = get_approach_class(cfg)
    embedder = approach_cls(cfg)
    table_component: TableEmbeddingInterface = embedder._load_component(
        "table_embedding_component", "TableEmbeddingComponent", TableEmbeddingInterface
    )
    _, resource_metrics_setup = component_utils.run_model_setup(component=table_component)
    return table_component, resource_metrics_setup


def _cache_path(cfg: DictConfig) -> Path:
    cache_dir = Path(cfg.cache_dir) / "ttd_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{cfg.run_identifier}|limit={cfg.test_case_limit}"
    run_hash = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{run_hash}.npz"


def _normalize_table_embedding(table_embedding: Any) -> np.ndarray:
    embedding = np.asarray(table_embedding)
    component_utils.assert_table_embedding_format(embedding)
    return np.squeeze(embedding, axis=0) if embedding.ndim == 2 else embedding


def _embed_tables(table_component: TableEmbeddingInterface, tables: list) -> np.ndarray:
    embeddings = []
    for table in tqdm(tables, desc="Embedding TTD tables"):
        embeddings.append(_normalize_table_embedding(table_component.create_table_embedding(table)))

    if not embeddings:
        raise ValueError("No TTD embeddings were created.")

    return np.stack(embeddings, axis=0)


def _embed_or_load_cached(
    table_component: TableEmbeddingInterface,
    cfg: DictConfig,
    train_tables: list,
    test_tables: list,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        force_embed = bool(cfg.benchmark_tasks.table_type_detection.task_parameters.force_embed_corpus)
    except Exception:
        force_embed = False

    cache_path = _cache_path(cfg)

    if not force_embed and cache_path.exists():
        logger.info(f"Loading cached TTD embeddings from {cache_path}")
        cached = np.load(cache_path)
        return cached["train"], cached["test"]

    logger.info("Creating TTD table embeddings.")
    train_embeddings = _embed_tables(table_component, train_tables)
    test_embeddings = _embed_tables(table_component, test_tables)

    np.savez(cache_path, train=train_embeddings, test=test_embeddings)
    logger.info(f"Saved TTD embeddings to {cache_path}")

    return train_embeddings, test_embeddings


def _train_classifiers(
    train_embeddings: np.ndarray, train_labels: list[int]
) -> tuple[dict[str, Any], sklearn.preprocessing.LabelEncoder]:
    label_encoder = sklearn.preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    num_classes = len(label_encoder.classes_)
    logger.info(f"TTD label mapping: {list(label_encoder.classes_)}")

    model_xgb = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=100,
    )
    model_xgb.fit(train_embeddings, y_train)

    model_mlp = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy="mean"),
        sklearn.preprocessing.StandardScaler(),
        sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42),
    )
    model_mlp.fit(train_embeddings, y_train)

    model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(train_embeddings, y_train)

    return {
        "XGBoost": model_xgb,
        "MLP": model_mlp,
        "KNeighbors": model_knn,
    }, label_encoder


def _predict_classifiers(
    models: dict[str, Any], test_embeddings: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    hard_preds = {name: model.predict(test_embeddings) for name, model in models.items()}
    proba_preds = {name: model.predict_proba(test_embeddings) for name, model in models.items()}
    return hard_preds, proba_preds


@monitor_resources()
def run_ttd_task(
    table_component: TableEmbeddingInterface,
    cfg: DictConfig,
    train_tables: list,
    train_labels: list[int],
    test_tables: list,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], sklearn.preprocessing.LabelEncoder]:
    train_embeddings, test_embeddings = _embed_or_load_cached(
        table_component=table_component,
        cfg=cfg,
        train_tables=train_tables,
        test_tables=test_tables,
    )
    models, label_encoder = _train_classifiers(train_embeddings, train_labels)
    hard_preds, proba_preds = _predict_classifiers(models, test_embeddings)
    return hard_preds, proba_preds, label_encoder


def main(cfg: DictConfig):
    logger.info("Started table type detection benchmark")
    multiprocessing.set_start_method("spawn", force=True)

    if cfg.dataset_name != TTD_DATASET_NAME:
        logger.error(f"TTD only supports dataset_name='{TTD_DATASET_NAME}', got '{cfg.dataset_name}'.")
        raise ValueError(f"Unsupported TTD dataset_name: {cfg.dataset_name}")

    table_embedding_component, resource_metrics_setup = get_embedder(cfg)

    test_tables, test_labels = load_ttd_split("test", limit=cfg.test_case_limit)
    train_tables, train_labels = load_ttd_split("train", limit=cfg.test_case_limit)

    (y_pred_values, y_proba_values, label_encoder), resource_metrics_task = run_ttd_task(
        table_component=table_embedding_component,
        cfg=cfg,
        train_tables=train_tables,
        train_labels=train_labels,
        test_tables=test_tables,
    )

    result_metrics = {}
    y_test = label_encoder.transform(test_labels)
    for model_name, y_pred in y_pred_values.items():
        accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
        f1_macro = sklearn.metrics.f1_score(y_test, y_pred, average="macro")
        f1_micro = sklearn.metrics.f1_score(y_test, y_pred, average="micro")
        log_loss = sklearn.metrics.log_loss(y_test, y_proba_values[model_name])
        logger.info(f"{model_name} TTD accuracy: {accuracy:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, log_loss: {log_loss:.4f}")
        result_metrics[f"{model_name}_accuracy (↑)"] = float(accuracy)
        result_metrics[f"{model_name}_f1_macro (↑)"] = float(f1_macro)
        result_metrics[f"{model_name}_f1_micro (↑)"] = float(f1_micro)
        result_metrics[f"{model_name}_log_loss (↓)"] = float(log_loss)

    save_resource_metrics_to_disk(
        cfg=cfg,
        resource_metrics_setup=resource_metrics_setup,
        resource_metrics_task=resource_metrics_task,
    )
    result_utils.save_results(cfg=cfg, metrics=result_metrics)
