import pandas as pd
import logging

from benchmark_src.approach_interfaces.cell_embedding_interface import CellEmbeddingInterface

logger = logging.getLogger(__name__)


class CellEmbeddingComponent(CellEmbeddingInterface):
    """
    Cell embedding component for TabICL approach.
    Uses TabICL's column embeddings to generate embeddings for each cell.
    """
    
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame = None, dataset_information: dict = None):
        """
        Load the TabICL model for cell embedding generation.
        
        Args:
            input_table: pd.DataFrame - The table to work with (optional)
            dataset_information: dict - Additional information about the dataset (optional)
        """
        logger.info("Setting up TabICL model for cell embedding task")
        self.approach_instance.load_trained_model()

    def create_cell_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Create an embedding for each cell of the given table using TabICL.
        
        Args:
            input_table: pd.DataFrame - The table to work with
        
        Returns:
            np.ndarray: Matrix of cell embeddings with shape [num_rows, num_columns, embedding_dim]
        """
        logger.info(f"Creating cell embeddings for table of shape {input_table.shape}")
        
        # Use the approach's get_cell_embeddings method
        cell_embeddings, column_names = self.approach_instance.get_cell_embeddings(input_table)
        
        logger.info(f"Generated cell embeddings with shape: {cell_embeddings.shape}")
        
        return cell_embeddings

# Made with Bob
