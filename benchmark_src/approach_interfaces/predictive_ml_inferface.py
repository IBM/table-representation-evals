import pandas as pd
from abc import ABC, abstractmethod

class PredictiveMLInterface(ABC):

    def __init__(self, approach_instance):
        pass
        
    @abstractmethod
    def setup_model_for_task(self, train_df: pd.DataFrame, train_labels: pd.Series, task_type: str, dataset_information: dict):
        """
        Please implement any steps you need to train/setup/load your model in order to later produce row embeddings of the given table.

            Args:
                input_table: pd.DataFrame   The table to work with
                task_type: str          either "classification" or "regression"
                dataset_information: dict   Additional information about the dataset, look into dataset for details
                                            Keys:
                                                    dataset_information["categorical_indicator"]: list
                                                    dataset_information["num_classes"]: int
        """
        self.task_type = task_type
        pass


    @abstractmethod  
    def predict_test_cases(self, test_df: pd.DataFrame, task_type: str):
        """
        Predict the target for the given test dataframe.
        
        The return type of the method depends on the 'task_type' parameter, as detailed below,
        to facilitate the calculation of evaluation metrics using scikit-learn.

            Args:
                test_df (pd.DataFrame): The input dataframe containing test cases for prediction. 
                                        It has the same features as the training data.

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


        
