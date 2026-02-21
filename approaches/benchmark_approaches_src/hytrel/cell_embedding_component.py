import pandas as pd
import logging
import numpy as np
import torch

from benchmark_src.approach_interfaces.cell_embedding_interface import CellEmbeddingInterface

logger = logging.getLogger(__name__)

### Cell embedding component for HyTrel approach
### Extracts cell-level embeddings from the HyTrel hypergraph model


class CellEmbeddingComponent(CellEmbeddingInterface):

    def __init__(self, approach_instance):
        """Initialize the cell embedding component.
        
        Args:
            approach_instance: Instance of HyTrelEmbedder with access to the model
        """
        self.approach_instance = approach_instance

        
    def setup_model_for_task(self, dataset_information: dict):
        """
        Load the HyTrel model for cell embedding extraction.

        Args:
            dataset_information: dict - Additional information about the dataset
        """
        self.approach_instance.load_trained_model()
        logger.info("HyTrel model loaded for cell embedding task")

    def create_cell_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Create an embedding for each cell of the given table using HyTrel's source node embeddings.
        
        In HyTrel's bipartite hypergraph:
        - Source nodes (s) represent individual cells
        - Target nodes (t) represent hyperedges (table, columns, rows)
        
        This method uses column hyperedge embeddings for headers and source embeddings for data cells.

        Args:
            input_table: pd.DataFrame - The table to work with

        Returns:
            np.ndarray: Matrix of cell embeddings with shape [#rows+1 (header), #columns, embedding_dimension]
                       Row 0 contains header embeddings, rows 1+ contain data cell embeddings
        """
        return self.approach_instance.get_cell_embeddings(input_table)

# Made with Bob
