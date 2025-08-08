import pandas as pd
from benchmark_src.approach_interfaces.predictive_ml_inferface import PredictiveMLInterface

class PredictiveMLComponent(PredictiveMLInterface):
    """
    Predictive ML component for TabPFN approach.
    Delegates predictive ML tasks to the approach instance.
    """
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, train_df: pd.DataFrame, train_labels: pd.Series, task_type: str, dataset_information: dict):
        # Delegate to the approach instance
        self.approach_instance.setup_model_for_task(train_df, train_labels, task_type, dataset_information)

    def predict_test_cases(self, test_df: pd.DataFrame, task_type: str):
        # Delegate to the approach instance
        return self.approach_instance.predict_test_tables(test_df, task_type) 