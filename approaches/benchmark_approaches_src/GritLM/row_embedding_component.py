import pandas as pd
import logging
import torch

from benchmark_src.approach_interfaces.row_embedding_interface import RowEmbeddingInterface

### Implement this component if your approach is able to provide row embeddings for a given table in a self-supervised way (no labels).
logger = logging.getLogger(__name__)

class RowEmbeddingComponent(RowEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance

    def setup_model_for_task(self, input_table: pd.DataFrame, dataset_information: dict):
        """
        Please implement any steps you need to train/setup/load your model in order to later produce row embeddings of the given table.

            Args:
                input_table: pd.DataFrame   The table to work with
                dataset_information: dict   Additional information about the dataset, look into dataset for details
        """
        self.approach_instance.load_trained_model()


    def gritlm_instruction(self, instruction):
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

    def create_row_embeddings_for_table(self, input_table: pd.DataFrame):
        """
        Create an embedding for each row of the given table and return it as a numpy array.

            Args:
                preprocessed_data: the output of the preprocessing function (as defined in approach.py)

            Returns: 
                np.ndarray: the matrix of the row embeddings with shape [#rows, embedding_dimension]
        """
        logger.debug(f"GritLM create_row_embeddings_for_table_called")
        all_rows = self.approach_instance.preprocessing(input_table=input_table)
        logger.debug(f"GritLM: Preprocessed the rows, next encode them. Have {len(all_rows)} rows")
        # encode the rows
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            row_embeddings = self.approach_instance.model.encode(all_rows, instruction=self.gritlm_instruction(""), show_progress_bar=True)
        logger.debug(f"Done creating the row embeddings")
        return row_embeddings