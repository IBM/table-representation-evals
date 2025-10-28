#!/usr/bin/env python3
"""
HyTrel Embedding Approach for Tabular Data

This module implements the HyTrel (Hypergraph-enhanced Tabular Data Representation Learning)
approach for generating row embeddings from tabular data using hypergraph neural networks.

Based on: https://github.com/awslabs/hypergraph-tabular-lm
"""

import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoConfig

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach

# Add HyTrel source to Python path
HYTREL_SRC_PATH = os.path.join(os.path.dirname(__file__), 'hytrel_src')
if HYTREL_SRC_PATH not in sys.path:
    sys.path.insert(0, HYTREL_SRC_PATH)

# Import HyTrel components
try:
    from data import BipartiteData
    from model import Encoder
except ImportError as e:
    raise ImportError(f"Failed to import HyTrel components: {e}. Please ensure hytrel_src is properly set up.")

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_MAX_TOKEN_LENGTH = 16
DEFAULT_VOCAB_SIZE = 30527

# HyTrel special tokens
MISSING_CAP_TAG = '[TAB]'
MISSING_CELL_TAG = "[CELL]"
MISSING_HEADER_TAG = "[HEAD]"
ROW_TAG = "[ROW]"
PAD_TAG = "[PAD]"

@dataclass
class OptimizerConfig:
    """Optimizer configuration for HyTrel model loading compatibility."""
    batch_size: int = 256
    base_learning_rate: float = 1e-3
    weight_decay: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-5
    lr_scheduler_type: str = "linear"
    warmup_step_ratio: float = 0.1
    seed: int = 42
    optimizer: str = "Adam"
    checkpoint_path: str = ""

class HyTrelEmbedder(BaseTabularEmbeddingApproach):
    """
    HyTrel embedding approach for tabular data.
    
    This class implements the HyTrel (Hypergraph-enhanced Tabular Data Representation Learning)
    approach for generating row embeddings from tabular data using hypergraph neural networks.
    
    The approach converts tabular data into a bipartite hypergraph where:
    - Source nodes (s) represent individual cells
    - Target nodes (t) represent hyperedges (table, columns, rows)
    - Edges connect cells to their corresponding hyperedges
    
    Attributes:
        cfg (DictConfig): Configuration object
        model (Encoder): HyTrel model for generating embeddings
        device (torch.device): Device to run the model on
        tokenizer (AutoTokenizer): BERT tokenizer for text processing
        max_token_length (int): Maximum token length for tokenization
    """
    
    def __init__(self, cfg: DictConfig):
        """Initialize the HyTrel embedder.
        
        Args:
            cfg (DictConfig): Configuration object containing model parameters
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer for HyTrel
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_token_length = DEFAULT_MAX_TOKEN_LENGTH
        
        logger.info(f"HyTrelEmbedder initialized on device: {self.device}")

    def _tokenize_word(self, word: str) -> Tuple[List[str], List[int]]:
        """Tokenize a word using the same approach as HyTrel.
        
        Args:
            word (str): Word to tokenize
            
        Returns:
            Tuple[List[str], List[int]]: Tokenized wordpieces and attention mask
        """
        # Apply scientific notation to numbers (from HyTrel data.py)
        number_pattern = re.compile(r"(\d+)\.?(\d*)")
        
        def number_repl(matchobj):
            pre = matchobj.group(1).lstrip("0")
            post = matchobj.group(2)
            if pre and int(pre):
                exponent = len(pre) - 1
            else:
                exponent = -re.search("(?!0)", post).start() - 1
                post = post.lstrip("0")
            return (pre + post).rstrip("0") + " scinotexp " + str(exponent)
        
        def apply_scientific_notation(line: str) -> str:
            return re.sub(number_pattern, number_repl, line)
        
        # Apply scientific notation and tokenize
        word = apply_scientific_notation(str(word))
        wordpieces = self.tokenizer.tokenize(word)[:self.max_token_length]
        
        # Pad to max_token_length
        if len(wordpieces) < self.max_token_length:
            wordpieces.extend([PAD_TAG] * (self.max_token_length - len(wordpieces)))
        
        # Create attention mask
        mask = [1] * len(wordpieces[:self.max_token_length]) + [0] * (self.max_token_length - len(wordpieces[:self.max_token_length]))
        
        return wordpieces, mask

    def load_trained_model(self):
        """Load the pre-trained HyTrel model."""
        if self.model is not None:
            return
            
        logger.info("Loading HyTrel model...")
        try:
            # Create model configuration
            config = self._create_model_config()
            
            # Initialize model
            self.model = Encoder(config)
            
            # Load pre-trained weights if available
            checkpoint_path = getattr(self.cfg.approach, "checkpoint_path", None)
            if checkpoint_path and os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)
            else:
                logger.warning("No pre-trained checkpoint found. Using randomly initialized model.")
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"HyTrel model loaded on {self.device}")
            
        except ImportError as e:
            logger.error(f"Failed to import HyTrel components: {e}")
            raise ImportError("HyTrel model components not found. Please ensure the hytrel_src directory is properly set up.")
        except Exception as e:
            logger.error(f"Failed to load HyTrel model: {e}")
            raise

    def _create_model_config(self) -> AutoConfig:
        """Create model configuration from config file.
        
        Returns:
            AutoConfig: Configured model configuration
        """
        # Get model configuration from config or use defaults
        model_config = getattr(self.cfg.approach, "model_config", {})
        
        # Load default BERT-base-uncased configuration
        config = AutoConfig.from_pretrained("bert-base-uncased")
        
        # Update with HyTrel-specific configuration
        config.update({
            'vocab_size': model_config.get('vocab_size', DEFAULT_VOCAB_SIZE),
            "pre_norm": model_config.get('pre_norm', False),
            "activation_dropout": model_config.get('activation_dropout', 0.1),
            "gated_proj": model_config.get('gated_proj', False),
            "num_hidden_layers": model_config.get('num_hidden_layers', 12),
            "hidden_act": model_config.get('hidden_act', "gelu"),
            "attention_probs_dropout_prob": model_config.get('attention_probs_dropout_prob', 0.1)
        })
        
        # Update with any additional model config from YAML
        for key, value in model_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config

    def _load_checkpoint(self, checkpoint_path: str):
        """Load pre-trained weights from checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        logger.info(f"Loading pre-trained weights from {checkpoint_path}")
        try:
            # Add OptimizerConfig to __main__ module for checkpoint compatibility
            self._inject_optimizer_config()
            
            # Load checkpoint
            state_dict = torch.load(open(checkpoint_path, 'rb'), map_location=self.device)
            
            # Handle different checkpoint formats
            if 'module' in state_dict:
                # DeepSpeed checkpoint format
                self._load_deepspeed_checkpoint(state_dict)
            elif 'state_dict' in state_dict:
                # PyTorch Lightning checkpoint
                self._load_lightning_checkpoint(state_dict)
            elif 'model_state_dict' in state_dict:
                # Standard PyTorch checkpoint
                self.model.load_state_dict(state_dict['model_state_dict'], strict=False)
            else:
                # Direct state dict
                self.model.load_state_dict(state_dict, strict=False)
            
            logger.info("Successfully loaded pre-trained weights")
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}. Using randomly initialized model.")

    def _inject_optimizer_config(self):
        """Inject OptimizerConfig into __main__ module for checkpoint compatibility."""
        import types
        
        # Create or get the __main__ module
        if '__main__' not in sys.modules:
            main_module = types.ModuleType('__main__')
            sys.modules['__main__'] = main_module
        else:
            main_module = sys.modules['__main__']
        
        # Add OptimizerConfig to __main__ module
        main_module.OptimizerConfig = OptimizerConfig

    def _load_deepspeed_checkpoint(self, state_dict: Dict[str, Any]):
        """Load DeepSpeed checkpoint format.
        
        Args:
            state_dict (Dict[str, Any]): State dictionary from checkpoint
        """
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        for k, v in state_dict['module'].items():
            if 'model' in k:
                name = k[13:]  # remove `module.model.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)

    def _load_lightning_checkpoint(self, state_dict: Dict[str, Any]):
        """Load PyTorch Lightning checkpoint format.
        
        Args:
            state_dict (Dict[str, Any]): State dictionary from checkpoint
        """
        model_state_dict = {}
        for key, value in state_dict['state_dict'].items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                model_state_dict[new_key] = value
            else:
                model_state_dict[key] = value
        
        self.model.load_state_dict(model_state_dict, strict=False)

    def preprocessing(self, input_table: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input table for HyTrel model.
        
        Args:
            input_table (pd.DataFrame): Input table to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed table ready for HyTrel processing
        """
        # Create a copy to avoid modifying the original
        input_table_clean = input_table.copy()
        
        # Handle categorical columns first
        for col in input_table_clean.columns:
            if input_table_clean[col].dtype.name == 'category':
                # Add empty string as a category if it doesn't exist
                if "" not in input_table_clean[col].cat.categories:
                    input_table_clean[col] = input_table_clean[col].cat.add_categories([""])
                # Fill NaN values with empty string
                input_table_clean[col] = input_table_clean[col].fillna("")
            else:
                # For non-categorical columns, fill NaN values with empty string
                input_table_clean[col] = input_table_clean[col].fillna("")
        
        # Convert all columns to string for tokenization
        for col in input_table_clean.columns:
            input_table_clean[col] = input_table_clean[col].astype(str)
        
        return input_table_clean

    def get_row_embeddings(self, input_table: pd.DataFrame) -> np.ndarray:
        """Generate row embeddings using HyTrel model.
        
        Args:
            input_table (pd.DataFrame): Input table with rows to embed
            
        Returns:
            np.ndarray: Row embeddings of shape (num_rows, embedding_dim)
        """
        self.load_trained_model()
        
        # Get target embeddings (table + columns + rows)
        target_embeddings, num_rows, num_cols = self._get_target_embeddings(input_table)
        
        # Extract row embeddings (indices num_cols+1 to num_cols+num_rows)
        row_start_idx = 1 + num_cols
        row_embeddings = target_embeddings[row_start_idx:row_start_idx + num_rows]
        
        # Validate size
        if len(row_embeddings) != num_rows:
            logger.warning(
                f"Row embeddings size mismatch: expected {num_rows} rows, "
                f"but got {len(row_embeddings)} embeddings."
            )
        
        # Convert to numpy
        row_embeddings = row_embeddings.cpu().numpy() if isinstance(row_embeddings, torch.Tensor) else row_embeddings
        logger.info(f"Generated row embeddings with shape: {row_embeddings.shape}")
        return row_embeddings

    def get_column_embeddings(self, input_table: pd.DataFrame) -> tuple:
        """Generate column embeddings using HyTrel model.
        
        Args:
            input_table (pd.DataFrame): Input table with columns to embed
            
        Returns:
            tuple: (column_embeddings, column_names) where column_embeddings has shape (num_columns, embedding_dim)
        """
        self.load_trained_model()
        
        # Get target embeddings (table + columns + rows)
        target_embeddings, num_rows, num_cols = self._get_target_embeddings(input_table)
        column_names = list(input_table.columns)
        
        # Extract column embeddings (indices 1 to num_cols, skip index 0 which is table)
        column_embeddings = target_embeddings[1:num_cols + 1]
        
        # Validate size
        if len(column_embeddings) != num_cols:
            logger.warning(
                f"Column embeddings size mismatch: expected {num_cols} columns, "
                f"but got {len(column_embeddings)} embeddings."
            )
        
        # Convert to numpy
        column_embeddings = column_embeddings.cpu().numpy() if isinstance(column_embeddings, torch.Tensor) else column_embeddings
        logger.info(f"Generated column embeddings with shape: {column_embeddings.shape}")
        return column_embeddings, column_names

    def _get_target_embeddings(self, input_table: pd.DataFrame) -> tuple:
        """Get target embeddings from HyTrel model for the given table.
        
        Args:
            input_table (pd.DataFrame): Input table
            
        Returns:
            tuple: (target_embeddings, num_rows, num_cols) where target_embeddings is a torch.Tensor
        """
        # Preprocess the input table
        input_table_clean = self.preprocessing(input_table)
        
        # Convert table to HyTrel hypergraph format
        bigraph = self._convert_table_to_hytrel_format(input_table_clean)
        
        # Get dimensions
        num_rows = len(input_table_clean)
        num_cols = len(input_table_clean.columns)
        
        # Generate target embeddings using the model
        with torch.no_grad():
            bigraph = bigraph.to(self.device)
            outputs = self.model(bigraph)
            
            # Extract target embeddings (Encoder returns (embedding_s, embedding_t))
            if isinstance(outputs, tuple):
                _, target_embeddings = outputs
            else:
                target_embeddings = outputs
        
        return target_embeddings, num_rows, num_cols

    def _convert_table_to_hytrel_format(self, table: pd.DataFrame) -> BipartiteData:
        """Convert pandas DataFrame to HyTrel hypergraph format.
        
        This method creates a bipartite hypergraph where:
        - Source nodes (s) represent individual cells
        - Target nodes (t) represent hyperedges (table, columns, rows)
        - Edges connect cells to their corresponding hyperedges
        
        Args:
            table (pd.DataFrame): Input table to convert
            
        Returns:
            BipartiteData: HyTrel hypergraph representation
        """
        # Convert table to list format
        header = table.columns.tolist()
        data = table.values.tolist()
        
        # Initialize lists for tokenized data
        wordpieces_xs_all, mask_xs_all = [], []
        wordpieces_xt_all, mask_xt_all = [], []
        nodes, edge_index = [], []
        
        # Table-level hyperedge (caption) - index 0
        self._add_table_hyperedge(wordpieces_xt_all, mask_xt_all)
        
        # Header to hyper-edges (t nodes) - indices 1 to num_cols
        self._add_column_hyperedges(header, wordpieces_xt_all, mask_xt_all)
        
        # Row to hyper-edges (t nodes) - indices num_cols+1 to num_cols+num_rows
        self._add_row_hyperedges(len(data), wordpieces_xt_all, mask_xt_all)
        
        # Cell to nodes (s nodes)
        self._add_cell_nodes(data, header, wordpieces_xs_all, mask_xs_all, nodes, edge_index)
        
        # Convert to tensors
        xs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xs_all], dtype=torch.long)
        xt_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xt_all], dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        
        # Create BipartiteData object
        return BipartiteData(
            edge_index=edge_index,
            x_s=xs_ids,
            x_t=xt_ids
        )

    def _add_table_hyperedge(self, wordpieces_xt_all: List, mask_xt_all: List):
        """Add table-level hyperedge."""
        wordpieces = [MISSING_CAP_TAG] + [PAD_TAG] * (self.max_token_length - 1)
        mask = [1] + [0] * (self.max_token_length - 1)
        wordpieces_xt_all.append(wordpieces)
        mask_xt_all.append(mask)

    def _add_column_hyperedges(self, header: List, wordpieces_xt_all: List, mask_xt_all: List):
        """Add column-level hyperedges."""
        for head in header:
            if pd.isna(head) or str(head).strip() == '':
                wordpieces = [MISSING_HEADER_TAG] + [PAD_TAG] * (self.max_token_length - 1)
                mask = [1] + [0] * (self.max_token_length - 1)
            else:
                wordpieces, mask = self._tokenize_word(str(head))
            wordpieces_xt_all.append(wordpieces)
            mask_xt_all.append(mask)

    def _add_row_hyperedges(self, num_rows: int, wordpieces_xt_all: List, mask_xt_all: List):
        """Add row-level hyperedges."""
        for _ in range(num_rows):
            wordpieces = [ROW_TAG] + [PAD_TAG] * (self.max_token_length - 1)
            mask = [1] + [0] * (self.max_token_length - 1)
            wordpieces_xt_all.append(wordpieces)
            mask_xt_all.append(mask)

    def _add_cell_nodes(self, data: List, header: List, wordpieces_xs_all: List, 
                       mask_xs_all: List, nodes: List, edge_index: List):
        """Add cell nodes and connect them to hyperedges."""
        for row_i, row in enumerate(data):
            for col_i, word in enumerate(row):
                if pd.isna(word) or str(word).strip() == '':
                    wordpieces = [MISSING_CELL_TAG] + [PAD_TAG] * (self.max_token_length - 1)
                    mask = [1] + [0] * (self.max_token_length - 1)
                else:
                    wordpieces, mask = self._tokenize_word(str(word))
                
                wordpieces_xs_all.append(wordpieces)
                mask_xs_all.append(mask)
                node_id = len(nodes)
                nodes.append(node_id)
                
                # Connect to hyperedges
                edge_index.append([node_id, 0])  # connect to table-level hyper-edge
                edge_index.append([node_id, col_i + 1])  # connect to col-level hyper-edge
                edge_index.append([node_id, row_i + 1 + len(header)])  # connect to row-level hyper-edge


