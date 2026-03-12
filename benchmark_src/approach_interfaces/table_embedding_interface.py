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