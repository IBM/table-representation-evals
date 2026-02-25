import pandas as pd
from benchmark_src.approach_interfaces.predictive_ml_inferface import PredictiveMLInterface

class PredictiveMLComponent(PredictiveMLInterface):
    """
    Predictive ML component for MITRA approach.
    Delegates predictive ML tasks to the approach instance.
    """
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, train_df: pd.DataFrame, train_labels: pd.Series, task_type: str, dataset_information: dict):
        """
        Set up the MITRA model for the given task.
        Delegates to the approach instance's load_predictive_ml_model method.
        """
        self.approach_instance.load_predictive_ml_model(train_df, train_labels, task_type, dataset_information)

    def predict_test_cases(self, test_df: pd.DataFrame, task_type: str):
        """
        Predict test cases using the MITRA model.
        Delegates to the approach instance's predict_test_cases method.
        """
        return self.approach_instance.predict_test_cases(test_df, task_type)

# Made with Bob
