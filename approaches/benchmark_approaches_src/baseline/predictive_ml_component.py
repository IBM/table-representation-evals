import re
import pandas as pd
import logging

import sklearn
import xgboost as xgb
from skrub import TableVectorizer

from benchmark_src.approach_interfaces.predictive_ml_inferface import PredictiveMLInterface

logger = logging.getLogger(__name__)

### Implement this component if your approach does not provide row embeddings but trains the model on the training input table + labels.
### Otherwise, just delete this file.

class PredictiveMLComponent(PredictiveMLInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance # with the approach instance you can call functions implemented in your CustomTabularEmbeddingApproach class 

    def setup_model_for_task(self, train_df: pd.DataFrame, train_labels: pd.Series, task_type: str, dataset_information: dict):
        """
        Please implement any steps you need to train/setup/load your model in order to later produce row embeddings of the given table.

            Args:
                input_table: pd.DataFrame   The table to work with
                task_type: str          either "classification" or "regression"
                dataset_information: dict   Additional information about the dataset, look into dataset for details
        """
        self.task_type = task_type # either classification or regression
        self.num_classes = dataset_information["num_classes"] # None for regression, int for classification tasks
        self.categorical_indicator = dataset_information["categorical_indicator"] # (boolean per column in train_df)
        self.positive_label = dataset_information["positive_label"] # none for regression, str for classification task

        self.vectorizer = TableVectorizer()
        X_train = self.vectorizer.fit_transform(train_df)

        X_train.columns = [re.sub(r'[\[\]<>]', '_', col) for col in X_train.columns]

        assert len(X_train) == len(train_df), f"Orig df len {len(train_df)}, after preprocessing {len(X_train)}"

        if task_type == "classification":
            # encode train labels
            label_encoder = sklearn.preprocessing.LabelEncoder()

            # Fit and transform the categorical labels to integers
            y_train = label_encoder.fit_transform(train_labels)
            logger.debug(f"Mapping of labels: {list(label_encoder.classes_)}")

            self.num_classes = len(list(label_encoder.classes_))
            assert self.num_classes > 1

            # train model depending on problem_type
            if self.num_classes == 2:
                assert self.num_classes == 2

                # TODO: get idx of positive label in label encodings

                model_xgb = xgb.XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=100,
                    enable_categorical=False
                )

                model_xgb.fit(X_train, y_train)    
            else:
                assert self.num_classes > 2

                model_xgb = xgb.XGBClassifier(
                    objective="multi:softprob",  # Use multi:softprob for multiclass probabilities
                    num_class=self.num_classes,      # Specify the number of classes
                    n_estimators=100,
                    enable_categorical=False
                )

                model_xgb.fit(X_train, y_train)

            model_lin_reg = None

        elif task_type == "regression":
            # do not need to transform labels for regression
            y_train = train_labels

            model_xgb = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            enable_categorical=False  
            )

            model_xgb.fit(X_train, y_train)

            model_lin_reg = sklearn.linear_model.LinearRegression()
            model_lin_reg.fit(X_train, y_train)

        self.models = {"XGBoost": model_xgb,
                        "LinearRegression": model_lin_reg}
        
        logger.info(f"Done with the training.")

        
 
    def predict_test_cases(self, test_df: pd.DataFrame, task_type: str):
        """
        Predict the target for the given test dataframe.
        
        The return type of the method depends on the 'task_type' parameter, as detailed below,
        to facilitate the calculation of evaluation metrics using scikit-learn.

            Args:
                test_df (pd.DataFrame): The input dataframe containing test cases for prediction. 
                                        It has the same features as the training data.
                task_type: str          Eßither "classification" or "regression"

            Returns: 
                If task_type is "classification":
                    - For binary classification (to compute ROC AUC): 
                      A NumPy array or a Pandas DataFrame/Series of shape (n_samples,), 
                      where each element is the probability of the positive class. 
                      This format must be suitable for `sklearn.metrics.roc_auc_score` 
                      with default parameters.

                    - For multiclass classification (to compute log loss): 
                      A NumPy array or a Pandas DataFrame/Series of shape 
                      (n_samples, n_classes), where each element represents the 
                      predicted probability of a sample belonging to a specific class. 
                      The columns should be ordered alphabetically by class label, 
                      as expected by `sklearn.metrics.log_loss`. 

                If task_type is "regression":
                    - A NumPy array or a Pandas DataFrame/Series of shape (n_samples,), 
                      containing the predicted continuous values for each sample. 
                      This is the required format for `sklearn.metrics.root_mean_squared_error`.
        """
        X_test = self.vectorizer.transform(test_df)
        X_test.columns = [re.sub(r'[\[\]<>]', '_', col) for col in X_test.columns]
        y_pred_values = {}

        for model_name, model in self.models.items():
            if model is None:
                continue

            if task_type == "classification":
                assert self.num_classes > 1

                # TODO: take positive class or rely on second one to be the positive one?
                y_pred = model.predict_proba(X_test)
            elif task_type == "regression":
                y_pred = model.predict(X_test)

            y_pred_values[model_name] = y_pred

        return y_pred_values


        
