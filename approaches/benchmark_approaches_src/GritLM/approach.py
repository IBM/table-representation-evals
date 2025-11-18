from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
import logging
from omegaconf import DictConfig
import pandas as pd
import torch

from benchmark_approaches_src.sentence_transformer import approach_utils 

## Custom imports
from gritlm import GritLM

logger = logging.getLogger(__name__)

class GritLMEmbedder(BaseTabularEmbeddingApproach):
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

        logger.info("GritLMEmbedder: Initialized.")

    # --- Methods that you want to re-use in multiple components  ---

    def preprocessing(self, input_table: pd.DataFrame):
        """
        Preprocess the data for GritLM

            Args:
                dataset_information: dict   Additional information about the dataset, look into dataset for details
        
            Returns:
                preprocessed_data in approach specific format
        """
        # convert all rows to strings
        all_rows = []
        for _, row in input_table.iterrows():
            table_row_string = approach_utils.convert_row_to_string(row)
            all_rows.append(table_row_string)

        preprocessed_data = all_rows # return the preprocessed_data in which ever format you like
        return preprocessed_data

    def train_model_self_supervised(self):
        """
        If your approach is trained / adapted to the input table in a self-supervised way, 
        we recommend you to implement it here and call the method from the task-specific components.

        """
        # no training necessary for GritLM
        pass
    
    def load_trained_model(self):
        """
        Load the trained model and set it as a class variable to access later
        """

        model = GritLM(self.embedding_model_name, mode="embedding", torch_dtype=torch.float16, device_map='auto')
        logger.info(f"GritLM: Loaded model!")
        dtypes = {param.dtype for param in model.parameters()}
        logger.info(f"GritLM model parameters are of type: {dtypes}")
        uses_flash_attention = False
        for name, module in model.named_modules():
            if "Attention" in type(module).__name__:
                if hasattr(module, "flash"):
                    logger.info(f"{name} uses FlashAttention: {module.flash}")
                    uses_flash_attention = True
        if not uses_flash_attention:
            logger.info(f"No flash attention is used")
        self.model = model
