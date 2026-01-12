import pandas as pd
import torch
from benchmark_src.approach_interfaces.column_embedding_interface import ColumnEmbeddingInterface

### Implement this component if your approach is able to provide column embeddings for a given table in a self-supervised way (no labels).
### Otherwise, just delete this file.


class ColumnEmbeddingComponent(ColumnEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance # with the approach instance you can call functions implemented in your CustomTabularEmbeddingApproach class

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

    def create_column_embeddings_for_table(self, input_table: str):
        """
        Create an embedding for each column of the given table and return it as a numpy array.

            Args:
                input_table: pd.DataFrame   The table to work with

            Returns: 
                np.ndarray: the matrix of the row embeddings with shape [#rows, embedding_dimension]
        """
        all_columns = self.approach_instance.preprocessing(input_table=input_table, component=self)
        # encode the columns
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            column_embeddings = self.approach_instance.model.encode(list(all_columns.values()),
                                                               instruction=self.gritlm_instruction(""),
                                                                 show_progress_bar=True)
        return column_embeddings, all_columns.keys()