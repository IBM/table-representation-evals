from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
import logging
from omegaconf import DictConfig
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

logger = logging.getLogger(__name__)

class TabuLA8BEmbedder(BaseTabularEmbeddingApproach):
    """
    TabuLA-8B embedding approach for tabular data.
    
    This approach uses the TabuLA-8B model which is specifically designed for 
    tabular data representation and can generate embeddings for table rows.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        # Initialize TabuLA-8B specific parameters
        self.model_name = cfg.approach.model_name if hasattr(cfg.approach, 'model_name') else "mlfoundations/tabula-8b"
        self.device = cfg.approach.device if hasattr(cfg.approach, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = cfg.approach.max_length if hasattr(cfg.approach, 'max_length') else 512
        self.batch_size = cfg.approach.batch_size if hasattr(cfg.approach, 'batch_size') else 8
        
        # Initialize model and tokenizer as None, will be loaded when needed
        self.model = None
        self.tokenizer = None
        
        logger.info(f"TabuLA8BEmbedder: Initialized with model {self.model_name} on device {self.device}")
    
    def load_trained_model(self):
        """
        Load the TabuLA-8B model and tokenizer.
        """
        try:
            logger.info(f"Loading TabuLA-8B model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("TabuLA-8B model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading TabuLA-8B model: {e}")
            raise
    
    def preprocessing(self, input_table: pd.DataFrame):
        """
        Preprocess the input table for TabuLA-8B.
        
        Args:
            input_table: pd.DataFrame - The table to preprocess
            
        Returns:
            list: List of preprocessed table strings
        """
        preprocessed_data = []
        
        for _, row in input_table.iterrows():
            # Convert row to string format suitable for TabuLA-8B
            table_row_string = self._convert_row_to_string(row)
            preprocessed_data.append(table_row_string)
        
        return preprocessed_data
    
    def _convert_row_to_string(self, row: pd.Series):
        """
        Convert a table row to a string format suitable for TabuLA-8B.
        
        Args:
            row: pd.Series - A single row from the table
            
        Returns:
            str: String representation of the row
        """
        # Convert all values to strings and handle NaN values
        row_dict = {}
        for col, val in row.items():
            if pd.isna(val):
                row_dict[col] = "N/A"
            else:
                row_dict[col] = str(val)
        
        # Create a structured string representation
        row_parts = []
        for col, val in row_dict.items():
            row_parts.append(f"{col}: {val}")
        
        return " | ".join(row_parts)
    
    def get_embeddings(self, texts: list):
        """
        Generate embeddings for a list of text inputs using TabuLA-8B.
        
        Args:
            texts: list - List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings with shape [num_texts, embedding_dim]
        """
        if self.model is None or self.tokenizer is None:
            self.load_trained_model()
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize the batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the last hidden state as embeddings
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        return np.vstack(embeddings) 
