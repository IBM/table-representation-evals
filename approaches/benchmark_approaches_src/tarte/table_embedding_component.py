import numpy as np
import pandas as pd

from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface


class TableEmbeddingComponent(TableEmbeddingInterface):
    """
    Table embedding component for TARTE.

    The table embedding is the mean of pre-trained row representations.
    TARTE has no CLS token or table-level token, so mean-pooling is the
    natural aggregation.

    create_query_embedding() is not supported: TARTE operates on structured
    tabular features and has no natural-language query encoder.
    """

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self):
        self.approach_instance.load_trained_model()

    def create_table_embedding(self, input_table: pd.DataFrame) -> np.ndarray:
        return self.approach_instance.get_table_embedding(input_table)

    def create_query_embedding(self, query: str) -> np.ndarray:
        raise NotImplementedError(
            "TARTE operates on structured tabular features and cannot embed "
            "natural-language queries. Exclude table_retrieval and nl2* tasks "
            "from the TARTE config."
        )
