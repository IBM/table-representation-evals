from abc import ABC, abstractmethod
from typing import List, Any
import pandas as pd


class TableEmbeddingInterface(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def setup_model_for_task(self):
        pass

    @abstractmethod
    def create_table_embedding(self, input_table: pd.DataFrame):
        pass

    @abstractmethod
    def create_query_embedding(self, query: str):
        pass

    def fit_corpus(self, tables: List[pd.DataFrame]) -> None:
        """
        Optional. Default is a no-op — most approaches don't need this and can ignore it.

        Override only if your approach must fit on the full table corpus before it can embed
        individual tables (e.g. a TF-IDF vocabulary). Called once by table_retrieval,
        table_shuffling, and table_type_detection before any create_table_embedding() calls.
        """
        pass