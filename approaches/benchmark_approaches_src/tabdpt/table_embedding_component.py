import numpy as np
import pandas as pd
from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface


class TableEmbeddingComponent(TableEmbeddingInterface):
    """
    Table embedding component for TabDPT.

    The table embedding is the mean of the pre-head row representations across
    all rows in the table.  TabDPT has no CLS token or dedicated table-level
    token, so mean-pooling is the natural aggregation.

    Note: TabDPT operates on raw floats and has no text understanding, so
    create_query_embedding() is not supported — table_retrieval tasks that
    require NL query matching are excluded in the config.
    """

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self):
        self.approach_instance.load_trained_model()

    def create_table_embedding(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Returns a (ninp,) embedding for the table, as the mean of all row
        representations produced by TabDPT's final transformer layer.
        """
        return self.approach_instance.get_table_embedding(input_table)

    def create_query_embedding(self, query: str) -> np.ndarray:
        raise NotImplementedError(
            "TabDPT operates on raw numerical features and cannot embed "
            "natural-language queries.  Exclude nl2* and table_retrieval "
            "tasks that require query embeddings from the TabDPT config."
        )
