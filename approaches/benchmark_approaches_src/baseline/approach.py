from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
import logging
from omegaconf import DictConfig
import pandas as pd

#from benchmark_approaches_src.<approach_folder> import approach_utils 

## Custom imports 


logger = logging.getLogger(__name__)

class BaselineEmbedder(BaseTabularEmbeddingApproach):
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

        logger.info("BaselineApproach: Initialized.")
        
    # --- Methods that you want to re-use in multiple components  ---

    def train_model_self_supervised(self, your_custom_parameters=None):
        """
        If your approach is trained / adapted to the input table in a self-supervised way, 
        we recommend you to implement it here and call the method from the task-specific components.

        """
        # --- YOUR IMPLEMENTATION GOES HERE ---
        pass
    
    
    def load_trained_model(self):
        """
        Load the trained model and set it as a class variable to access later
        """
        # --- YOUR IMPLEMENTATION GOES HERE ---
        model = None
        self.model = model
        pass