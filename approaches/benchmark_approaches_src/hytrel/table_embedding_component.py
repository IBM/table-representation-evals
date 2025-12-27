from typing import Any, List
import pandas as pd
import numpy as np
import logging

from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface

logger = logging.getLogger(__name__)


class TableEmbeddingComponent(TableEmbeddingInterface):
    """
    Table embedding component for HyTrel approach.
    Uses HyTrel's table hyperedge embedding to generate embeddings for entire tables.
    """

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance
        super().__init__()

    def setup_model_for_task(self):
        """Load the model if not already loaded."""
        self.approach_instance.load_trained_model()

    def create_table_embedding(self, input_table: List[List[Any]]):
        """
        Create an embedding for the entire table using HyTrel's table hyperedge.
        
        Args:
            input_table: List[List[Any]] - Table as list of rows (first row is headers)
            
        Returns:
            np.ndarray: Table embedding of shape (embedding_dim,)
        """
        # Convert List[List] to DataFrame
        if len(input_table) == 0:
            raise ValueError("Input table is empty")
        
        # First row is headers, rest are data rows
        headers = [str(cell) for cell in input_table[0]]
        data_rows = input_table[1:] if len(input_table) > 1 else []
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Get table embedding using the approach instance
        table_embedding = self.approach_instance.get_table_embedding(df)
        
        # Ensure it's 1D
        if table_embedding.ndim > 1:
            table_embedding = table_embedding.squeeze()
        
        
        norm = np.linalg.norm(table_embedding)
        if norm > 0:
            table_embedding = table_embedding / norm
        
        return table_embedding

    def create_query_embedding(self, query: str):
        """
        Create an embedding for a natural language query.
        
        For HyTrel, we treat the query as a single-cell table and embed it.
        This allows us to use the full HyTrel model for query embedding.
        
        Args:
            query: str - Natural language query text
            
        Returns:
            np.ndarray: Query embedding of shape (embedding_dim,)
        """
        # Treat query as a single-cell table with one column "query"
     
        query_table = pd.DataFrame({"query": [query]})
        
        # Get table embedding (which will be the table hyperedge)
        query_embedding = self.approach_instance.get_table_embedding(query_table)
        
        # Ensure it's 1D
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.squeeze()
        
        # L2 normalize for cosine similarity (matching sentence transformer behavior)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        return query_embedding


