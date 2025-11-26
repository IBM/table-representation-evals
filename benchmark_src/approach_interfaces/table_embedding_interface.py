from abc import ABC, abstractmethod
from typing import List, Any


class TableEmbeddingInterface(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def setup_model_for_task(self):
        pass

    @abstractmethod
    def create_table_embedding(self, input_table: List[List[Any]]):
        pass

    @abstractmethod
    def create_query_embedding(self, query: str):
        pass