import sklearn.linear_model
import sklearn.neighbors
import sklearn.neural_network
from omegaconf import DictConfig
import multiprocessing
import logging
import pandas as pd

import sklearn.metrics
import xgboost as xgb
import openml
import sklearn
import sklearn.preprocessing
import numpy as np

from benchmark_src.approach_interfaces.predictive_ml_inferface import PredictiveMLInterface
from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface
from benchmark_src.utils.resource_monitoring import monitor_resources, save_resource_metrics_to_disk
from benchmark_src.utils import framework, result_utils
from benchmark_src.tasks import component_utils, tabarena_datasets


logger = logging.getLogger(__name__)

POSITIVE_LABELS = ["yes", "true", "satisfied", "good", "1", "pos", "positive", "1.0"]



def load_tabarena_data(task_id: int):
    """
    
    """
    dataset_information = {}
    task = openml.tasks.get_task(task_id)

    dataset = task.get_dataset()

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=task.target_name, dataset_format="dataframe"
    )

    dataset_information["categorical_indicator"] = categorical_indicator
    dataset_information["num_classes"] = None
    dataset_information["positive_label"] = None


    assert len(X) > 0



    train_indices, test_indices = task.get_train_test_split_indices(fold=0, repeat=0)
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    if task.task_type_id.name == 'SUPERVISED_CLASSIFICATION':
        task_type = "classification"
        dataset_information["num_classes"] = len(task.class_labels) #return this for predictive ml component
        
        # based on target column, get positive label!
        # Filter y to only include values that are in POSITIVE_LABELS
        y_str = y.astype(str).str.lower()
        positive_values_in_y = list(y_str[y_str.isin(POSITIVE_LABELS)].unique())

        if len(positive_values_in_y) > 0:
            if len(positive_values_in_y) > 1:
                raise ValueError(f"Found more than one positive label: {positive_values_in_y}")
            dataset_information["positive_label"] = positive_values_in_y[0]
            print(f"The determined positive label is: { positive_values_in_y[0]}")
        else:
            print(f"No positive labels found in the dataset based on your list. {y_str.unique()}")

    elif task.task_type_id.name == 'SUPERVISED_REGRESSION':
        task_type = "regression"
    else:
        raise AssertionError(f'Unsupported task type: {task.task_type_id.name}')

    return X, X_train, y_train, X_test, y_test, task_type, dataset_information


@monitor_resources()
def run_training_based_on_row_embeddings(row_embedding_component, task_type, whole_table, train_table, train_labels, dataset_information):
    """
    TODO

        Returns:
            model
            idx_of_positive_label
    """
    logger.debug(f"Called run_task_based_on_row_embeddings")
    logger.info(f"Training phase: whole_table shape={whole_table.shape}, train_table shape={train_table.shape}")
    # setup approach model
    row_embedding_component.setup_model_for_task(input_table=whole_table, dataset_information=dataset_information)

    # get row embeddings and assert they have the correct format and shape
    # For TabICL/TabPFN: pass train_labels but NO train_size during training
    # For other methods: pass train_table without extra parameters
    approach_name = row_embedding_component.approach_instance.cfg.approach.get("name", "").lower()
    logger.info(f"Detected approach name: '{approach_name}'")
    if "tabicl" in approach_name or "tabpfn" in approach_name:
        logger.info(f"TabICL/TabPFN training: calling create_row_embeddings with train_table shape={train_table.shape}, train_labels provided, train_size=None")
        train_row_embeddings = row_embedding_component.create_row_embeddings_for_table(input_table=train_table, train_size=None, train_labels=train_labels)
    else:
        logger.info(f"Standard training: calling create_row_embeddings with train_table shape={train_table.shape}")
        train_row_embeddings = row_embedding_component.create_row_embeddings_for_table(input_table=train_table)
    component_utils.assert_row_embedding_format(row_embeddings=train_row_embeddings, input_table=train_table)

    X_train = train_row_embeddings

    idx_positive_label = -1

    if task_type == "classification":
        # encode train labels
        label_encoder = sklearn.preprocessing.LabelEncoder()

        # Fit and transform the categorical labels to integers
        y_train = label_encoder.fit_transform(train_labels)
        logger.debug(f"Mapping of labels: {list(label_encoder.classes_)}")

        assert dataset_information['num_classes'] > 1

        # train model depending on problem_type
        if dataset_information['num_classes'] == 2:
            for idx, class_label in enumerate(list(label_encoder.classes_)):
                class_label = str(class_label)
                if class_label.lower() in POSITIVE_LABELS:
                    idx_positive_label = idx
                    logger.info(f"Index of positive class in labels: {idx_positive_label}")
            if idx_positive_label == -1:
                raise ValueError(f"Have dataset with a not yet supported positive label: {list(label_encoder.classes_)}")

            # Binary XGBoost
            model_xgb = xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=100,
            )

            model_xgb.fit(X_train, y_train)    

        else:
            assert dataset_information['num_classes'] > 2

            # multiclass XGBoost
            model_xgb = xgb.XGBClassifier(
                objective="multi:softprob",  # Use multi:softprob for multiclass probabilities
                num_class=dataset_information['num_classes'],      # Specify the number of classes
                n_estimators=100,
            )

            model_xgb.fit(X_train, y_train)

        # MLP
        model_mlp = sklearn.pipeline.make_pipeline(
            sklearn.impute.SimpleImputer(strategy="mean"),
            sklearn.preprocessing.StandardScaler(),
            sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42)
        )
        model_mlp.fit(X_train, y_train)

        # KNN
        model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5) # TODO 'n_neighbors'
        model_knn.fit(X_train, y_train)

        model_lin_reg = None

    elif task_type == "regression":
        # do not need to transform labels for regression
        y_train = train_labels

        model_xgb = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        )

        model_xgb.fit(X_train, y_train)

        model_lin_reg = sklearn.linear_model.LinearRegression()
        model_lin_reg.fit(X_train, y_train)

        model_knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5) # Tune as needed
        model_knn.fit(X_train, y_train)

        # MLP regressor
        model_mlp = sklearn.pipeline.make_pipeline(
            sklearn.impute.SimpleImputer(strategy="mean"),
            sklearn.preprocessing.StandardScaler(),
            sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(128,), max_iter=500, random_state=42)
        )
        model_mlp.fit(X_train, y_train)

    models = {"XGBoost": model_xgb,
              "KNeighbors": model_knn,
              "LinearRegression": model_lin_reg,
              "MLP": model_mlp}

    return models, idx_positive_label

@monitor_resources()
def run_inference_based_on_row_embeddings(models, row_embedding_component, test_table, task_type, num_classes=None, idx_positive_label=None, train_size=None, actual_test_table=None, train_labels=None):
    # get row embeddings and assert they have the correct format and shape
    # Pass train_size and train_labels to allow the approach to use them for embedding generation
    test_row_embeddings = row_embedding_component.create_row_embeddings_for_table(input_table=test_table, train_size=train_size, train_labels=train_labels)
    # Use actual_test_table for validation if provided (for TabICL), otherwise use test_table
    validation_table = actual_test_table if actual_test_table is not None else test_table
    component_utils.assert_row_embedding_format(row_embeddings=test_row_embeddings, input_table=validation_table)

    X_test = test_row_embeddings

    y_pred_values = {}

    for model_name, model in models.items():
        if model is None:
            continue

        if task_type == "classification":
            assert num_classes > 1

            # train model depending on problem_type
            if num_classes == 2:
                # Get predicted probabilities for the positive class
                y_pred = model.predict_proba(X_test)[:, idx_positive_label]
            else:
                y_pred = model.predict_proba(X_test)
        elif task_type == "regression":
            y_pred = model.predict(X_test)

        y_pred_values[model_name] = y_pred

    return y_pred_values


@monitor_resources()
def run_inference_based_on_custom_model(predictive_ml_component, test_table, task_type):
    # get predictions for test table rows:
    y_pred = predictive_ml_component.predict_test_cases(test_df=test_table, task_type=task_type)

    return y_pred

def main(cfg: DictConfig):
    logger.debug(f"Started predictive ml benchmark")
    logger.debug(f"Received cfg:")
    logger.debug(cfg)
    multiprocessing.set_start_method("spawn", force=True) 

    # instantiate the embedding approach class
    embedding_approach_class = framework.get_approach_class(cfg)
    embedder = embedding_approach_class(cfg)

    # Load data
    if cfg.dataset_name in tabarena_datasets.TABARENA_NAME_TO_ID.keys():
        task_id = tabarena_datasets.TABARENA_NAME_TO_ID[cfg.dataset_name]
       
        whole_table, train_table, train_labels, test_table, test_labels, task_type, dataset_information = load_tabarena_data(task_id)
    else:
        raise NotImplementedError(f"Received unknown dataset_name: {cfg.dataset_name}")
    
    logger.debug(f"Task type is: {task_type}")

    run_task_based_on = cfg.benchmark_tasks.predictive_ml.task_parameters.run_task_based_on
    
    if run_task_based_on == "row_embeddings":
        logger.info(f"Running task based on row embeddings")
        ## load the needed component
        row_embedding_component = embedder._load_component("row_embedding_component", "RowEmbeddingComponent", RowEmbeddingInterface)

        ## setup embedding model, get embeddings, train predictiveML model
        training_output, resource_metrics_setup = run_training_based_on_row_embeddings(row_embedding_component=row_embedding_component, task_type=task_type, whole_table=whole_table, train_table=train_table, train_labels=train_labels, dataset_information=dataset_information)
        models, idx_positive_label = training_output

        # run inference with model
        # For TabICL/TabPFN: pass whole_table with train_size to get test embeddings with full context
        # For other methods: pass test_table without train_size (backward compatible)
        approach_name = embedder.cfg.approach.get("name", "").lower()
        if "tabicl" in approach_name or "tabpfn" in approach_name:
            logger.info(f"TabICL/TabPFN mode: whole_table shape={whole_table.shape}, train_table shape={train_table.shape}, test_table shape={test_table.shape}")
            # Pass whole table but also pass actual test_table for proper validation
            y_pred_values, resource_metrics_task = run_inference_based_on_row_embeddings(
                models=models,
                row_embedding_component=row_embedding_component,
                test_table=whole_table,  # Pass whole table for TabICL/TabPFN
                task_type=task_type,
                num_classes=dataset_information["num_classes"],
                idx_positive_label=idx_positive_label,
                train_size=len(train_table),  # Mark where training ends
                actual_test_table=test_table,  # For validation
                train_labels=train_labels  # Pass training labels
            )
        else:
            y_pred_values, resource_metrics_task = run_inference_based_on_row_embeddings(
                models=models,
                row_embedding_component=row_embedding_component,
                test_table=test_table,
                task_type=task_type,
                num_classes=dataset_information["num_classes"],
                idx_positive_label=idx_positive_label
            )
    elif run_task_based_on == "custom_predictiveML_model":
        ## load the needed component
        predictive_ml_component = embedder._load_component("predictive_ml_component", "PredictiveMLComponent", PredictiveMLInterface)

        ## setup model for task
        _, resource_metrics_setup = component_utils.run_model_setup(component=predictive_ml_component, train_df=train_table, train_labels=train_labels, task_type=task_type, dataset_information=dataset_information)

        ## run the task
        y_pred, resource_metrics_task = run_inference_based_on_custom_model(predictive_ml_component=predictive_ml_component, test_table=test_table, task_type=task_type)
        
        #keep only the positive class prediction for binary classification to work with au-roc from sklearn
        if not isinstance(y_pred, dict):
            if dataset_information["num_classes"] == 2:
                print(y_pred.shape)
                y_pred = y_pred[:,1]
                print(y_pred.shape)
            
            y_pred_values = {"approach": y_pred}
        else:
            if dataset_information["num_classes"] == 2:
                for model_name, y_pred_val in y_pred.items():
                    print(y_pred_val.shape)
                    y_pred_val = y_pred_val[:,1]
                    print(y_pred_val.shape)
                    y_pred[model_name] = y_pred_val

            y_pred_values = y_pred

    else:
        logger.error(f"Got unsupported value for 'run_task_based_on' parameter: *{run_task_based_on}*")
        raise ValueError

    result_metrics = {}

    for model_name, y_pred in y_pred_values.items():
        if task_type == "classification":
            label_encoder = sklearn.preprocessing.LabelEncoder()
            label_encoder.fit_transform(train_labels)
            y_test = label_encoder.transform(test_labels)

            if dataset_information["num_classes"] == 2:        
                # Calculate the ROC AUC score for binary prediction
                auc_score = sklearn.metrics.roc_auc_score(y_test, y_pred)
                logger.info(f"Final AUC score (↑) on test set: {model_name}: {auc_score:.4f}")
                result_metrics[f"{model_name}_roc_auc_score (↑)"] = auc_score
        
            else:
                assert dataset_information["num_classes"] > 2

                evaluation_score = sklearn.metrics.log_loss(y_test, y_pred)
                logger.info(f"Final log loss evaluation score (↓) on test set: {model_name}: {evaluation_score:.4f}")
                result_metrics[f"{model_name}_log_loss (↓)"] = evaluation_score
                
        elif task_type == "regression":
            y_test = test_labels

            rmse_score = sklearn.metrics.root_mean_squared_error(y_true=y_test, y_pred=y_pred)
            logger.info(f"Root Mean Squared Error (RMSE) (↓) on test set: {model_name}: {rmse_score:.4f}")
            result_metrics[f"{model_name}_rmse (↓)"] = rmse_score

    # save resource metrics to disk
    save_resource_metrics_to_disk(cfg=cfg, resource_metrics_setup=resource_metrics_setup, resource_metrics_task=resource_metrics_task)

    # save results to disk
    result_utils.save_results(cfg=cfg, metrics=result_metrics)
    

