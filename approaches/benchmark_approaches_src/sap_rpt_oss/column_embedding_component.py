import pandas as pd
import numpy as np

from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface


class ColumnEmbeddingComponent(ColumnEmbeddingInterface):
    """
    Column embedding component for SAP RPT-1-OSS approach.
    Delegates to the SAP RPT-1-OSS embedder's get_column_embeddings.
    """

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        # Ensure model is loaded prior to embedding generation
        self.approach_instance.load_trained_model()

    def create_column_embeddings_for_table(self, input_table: pd.DataFrame):
        if len(input_table) > 1000:
            # if there are more than 500 columns, take only the first 100 rows:
            if len(input_table.columns) > 500:
                input_table = input_table.head(100)
            else:
                input_table = input_table.head(1000)
        column_embeddings, column_names = self.approach_instance.get_column_embeddings(input_table)
        # Return as numpy array and names for downstream tasks
        return column_embeddings, column_names

