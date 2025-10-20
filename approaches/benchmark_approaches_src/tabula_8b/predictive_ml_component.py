import pandas as pd
from benchmark_src.approach_interfaces.predictive_ml_inferface import PredictiveMLInterface

class PredictiveMLComponent(PredictiveMLInterface):
    """
    Predictive ML component for Tabula-8B approach.
    Delegates predictive ML tasks to the approach instance.
    """
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, train_df: pd.DataFrame, train_labels: pd.Series, task_type: str, dataset_information: dict):
        """
        Set up the Tabula-8B model for predictive ML tasks.
        Args:
            train_df (pd.DataFrame): Training data.
            train_labels (pd.Series): Training labels.
            task_type (str): Either "classification" or "regression".
            dataset_information (dict): Additional dataset info.
        """
        # Delegate to the approach instance
        self.approach_instance.load_predictive_ml_model(train_df, train_labels, task_type, dataset_information)

    def predict_test_cases(self, test_df: pd.DataFrame, task_type: str):
        """
        Predict the target for the given test dataframe using the Tabula-8B model.
        Args:
            test_df (pd.DataFrame): The input dataframe containing test cases for prediction.
            task_type (str): Either "classification" or "regression".
        Returns:
            np.ndarray: Predictions as required by the benchmark framework.
        """
        # Delegate to the approach instance
        return self.approach_instance.predict_test_cases(test_df, task_type)
