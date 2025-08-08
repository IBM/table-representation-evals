from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
import logging
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

from benchmark_approaches_src.sentence_transformer import approach_utils 

## Custom imports 
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedder(BaseTabularEmbeddingApproach):
    """
    This is your main embedding approach class.

    It inherits from BaseTabularEmbeddingApproach provided by the benchmark suite.
    You can implement methods here that you want to re-use in multiple components.

    All tasks (like similarity search or row embeddings) are handled by
    separate component files (e.g., `row_embedding_component.py`).
    If you provide these files, your BaseTabularEmbeddingApproach will automatically
    load them and make their functionality available. 
    Delete the component file that you do not want to implement.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg) # IMPORTANT: Call the base class constructor first

        # initialize your approach, save everything you need as custom class attributes
        # with cfg.approach.<your_parameter> you can access the custom parameters you set for your approach in the hydra config.
        # e.g. self.<your_parameter> = cfg.approach.<your_parameter>

        assert cfg.approach.embedding_model is not None, f"Please pass an embedding_model name"
        self.embedding_model_name = cfg.approach.embedding_model

        logger.info("SentenceTransformerEmbedder: Initialized.")
        
    # --- Methods that you want to re-use in multiple components  ---

    def train_model_self_supervised(self, your_custom_parameters=None):
        """
        If your approach is trained / adapted to the input table in a self-supervised way, 
        we recommend you to implement it here and call the method from the task-specific components.

        """
        # --- YOUR IMPLEMENTATION GOES HERE ---
        pass
    
    def preprocessing(self, input_table: pd.DataFrame):
        """
        Linearize the rows into strings
        """
        # convert all rows to strings

        all_rows = []
        for _, row in tqdm(input_table.iterrows()):
            table_row_string = approach_utils.convert_row_to_string(row)
            all_rows.append(table_row_string)

        preprocessed_data = all_rows # return the preprocessed_data in which ever format you like
        return preprocessed_data

    
    def load_trained_model(self):
        """
        Load the trained model and set it as a class variable to access later
        """
        model = SentenceTransformer(self.embedding_model_name)
        print(f"Loaded model!")
        self.model = model
