import pandas as pd
import logging
import torch

from benchmark_src.approach_interfaces.cell_embedding_interface import CellEmbeddingInterface

logger = logging.getLogger(__name__)

### Implement this component if your approach is able to provide cell embeddings for a given table in a self-supervised way (no labels).
### Otherwise, just delete this file.



class CellEmbeddingComponent(CellEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance  # with the approach instance you can call functions implemented in your CustomTabularEmbeddingApproach class 

        
    def setup_model_for_task(self, dataset_information: dict):
        """
        Please implement any steps you need to train/setup/load your model in order to later produce cell embeddings of the given table.

            Args:
                input_table: pd.DataFrame   The table to work with
                dataset_information: dict   Additional information about the dataset, look into dataset for details
        """
        self.approach_instance.load_trained_model()
        logger.info(f"Ran setup for task!")

    def create_cell_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Create an embedding for each cell of the given table and return it as a numpy array.

            Args:
                input_table: pd.DataFrame   The table to work with

            Returns: 
                np.ndarray: the matrix of the row embeddings with shape [#row+1 (header), #column, embedding_dimension]
        """
        num_rows, num_cols = input_table.shape

        logger.debug(f"GritLM: Embedding table of shape {num_rows}x{num_cols}")

        # Prepare text for each cell
        # Row 0 will be the header embeddings
        cell_texts = []
        for r in range(-1, num_rows):  # -1 for header row
            for c in range(num_cols):
                header_text = str(input_table.columns[c])  # get column name
                if r == -1:
                    text = header_text  # header embedding
                else:
                    cell_value = str(input_table.iat[r, c])
                    # Combine header + cell for richer embedding
                    text = f"{header_text}: {cell_value}"
                cell_texts.append(text)

        # Embed all at once
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings_flat = self.approach_instance.model.encode(
                cell_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
            )

        embedding_dim = embeddings_flat.shape[1]
        logger.info(f"GritLM embedding dim={embedding_dim}")

        # Reshape into [num_rows+1, num_cols, embedding_dim]
        cell_embeddings = embeddings_flat.reshape((num_rows + 1, num_cols, embedding_dim))

        return cell_embeddings