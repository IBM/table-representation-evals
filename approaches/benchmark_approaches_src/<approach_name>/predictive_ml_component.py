import pandas as pd
import logging

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
        self.task_type = task_type
        pass
 
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
        if self.task_type == "classification":
            # Example implementation for multiclass classification:
            # predictions = model.predict_proba(test_df) 
            predictions = None
        elif self.task_type == "regression":
            # predictions = model.predict(test_df)  
            predictions = None 
        return predictions


        
