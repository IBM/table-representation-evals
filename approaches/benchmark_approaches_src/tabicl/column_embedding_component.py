import pandas as pd
import numpy as np
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface

class ColumnEmbeddingComponent(ColumnEmbeddingInterface):
    """
    Column embedding component for TabICL approach.
    Uses TabICL's ColEmbedding stage to generate embeddings for each column.
    """
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        # Load the model if not already loaded
        self.approach_instance.load_trained_model()

    def create_column_embeddings_for_table(self, input_table: pd.DataFrame):
        # limit table size to first 1000 rows for efficiency
        if len(input_table) > 1000:
            # if there are more than 500 columns, take only the first 100 rows:
            if len(input_table.columns) > 500:
                input_table = input_table.head(100)
            else:
                input_table = input_table.head(1000)
        # Get column embeddings using the approach instance
        return self.approach_instance.get_column_embeddings(input_table)
