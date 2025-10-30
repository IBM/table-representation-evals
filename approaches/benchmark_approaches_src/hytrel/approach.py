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
    from data import TableDataModule, CAP_TAG as HY_CAP_TAG, HEADER_TAG as HY_HEADER_TAG, ROW_TAG as HY_ROW_TAG, MISSING_CAP_TAG as HY_MISSING_CAP_TAG
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

    def load_trained_model(self):
        """Load the pre-trained HyTrel model."""
        if self.model is not None:
            return
            
        logger.info("Loading HyTrel model...")
        try:
            # Create model configuration and initialize Encoder per hytrel_src
            config = self._create_model_config()
            self.model = Encoder(config)

            # Load pre-trained weights using the same logic as hytrel_src/evaluate_*.py
            checkpoint_path = getattr(self.cfg.approach, "checkpoint_path", None)
            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f"Loading checkpoint from: {checkpoint_path}")
                # Inject OptimizerConfig into __main__ before loading to handle pickle references
                self._inject_optimizer_config()
                
                # Load checkpoint - ensure __main__ module has OptimizerConfig for pickle
                # This handles cases where pickle looks for OptimizerConfig in __main__
                # (especially in joblib/loky subprocess contexts)
                original_main = sys.modules.get('__main__', None)
                import types
                
                # Ensure current __main__ has OptimizerConfig
                if '__main__' in sys.modules:
                    sys.modules['__main__'].OptimizerConfig = OptimizerConfig
                else:
                    # Create new __main__ if it doesn't exist
                    main_module = types.ModuleType('__main__')
                    main_module.OptimizerConfig = OptimizerConfig
                    sys.modules['__main__'] = main_module
                
                try:
                    # Use weights_only=False since we trust our checkpoints (compatible with PyTorch 2.6+)
                    state_dict = torch.load(open(checkpoint_path, 'rb'), map_location=self.device, weights_only=False)
                except (AttributeError, TypeError) as e:
                    if "OptimizerConfig" in str(e):
                        # If still failing, try adding to the specific module mentioned in error
                        # Error format: "Can't get attribute 'OptimizerConfig' on <module 'MODULE_NAME' from ...>"
                        error_str = str(e)
                        import re
                        # Extract module name from error message - look for pattern: module 'MODULE_NAME'
                        mod_matches = re.findall(r"module ['\"]([^'\"]+)['\"]", error_str)
                        for mod_name in mod_matches:
                            if mod_name not in sys.modules:
                                fake_mod = types.ModuleType(mod_name)
                                fake_mod.OptimizerConfig = OptimizerConfig
                                sys.modules[mod_name] = fake_mod
                            elif not hasattr(sys.modules[mod_name], 'OptimizerConfig'):
                                sys.modules[mod_name].OptimizerConfig = OptimizerConfig
                        # Try loading again with weights_only=False
                        state_dict = torch.load(open(checkpoint_path, 'rb'), map_location=self.device, weights_only=False)
                    else:
                        raise
                from collections import OrderedDict
                new_state_dict = OrderedDict()

                if 'module' in state_dict:  # DeepSpeed format
                    for k, v in state_dict['module'].items():
                        if 'model' in k:
                            name = k[13:]  # remove `module.model.`
                            new_state_dict[name] = v
                elif 'state_dict' in state_dict:  # PyTorch Lightning format
                    for k, v in state_dict['state_dict'].items():
                        if k.startswith('model.'):
                            name = k[6:]  # remove `model.`
                            new_state_dict[name] = v
                        else:
                            new_state_dict[k] = v
                elif 'model_state_dict' in state_dict:  # Plain dict under key
                    new_state_dict = state_dict['model_state_dict']
                else:  # Direct state dict
                    new_state_dict = state_dict

                self.model.load_state_dict(new_state_dict, strict=True)
            else:
                logger.warning("No pre-trained checkpoint found. Using randomly initialized model.")
                if checkpoint_path:
                    logger.warning(f"Checkpoint path provided but file not found: {checkpoint_path}")

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
            
            # Load checkpoint with weights_only=False (compatible with PyTorch 2.6+)
            state_dict = torch.load(open(checkpoint_path, 'rb'), map_location=self.device, weights_only=False)
            
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
        """Inject OptimizerConfig into __main__ module and builtins for checkpoint compatibility."""
        import types
        import builtins
        
        # Create or get the __main__ module
        if '__main__' not in sys.modules:
            main_module = types.ModuleType('__main__')
            sys.modules['__main__'] = main_module
        else:
            main_module = sys.modules['__main__']
        
        # Add OptimizerConfig to __main__ module
        main_module.OptimizerConfig = OptimizerConfig
        # Also add to builtins so pickle can find it in any context
        builtins.OptimizerConfig = OptimizerConfig

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
        
        # Preprocess to get the exact table used in embedding generation
        input_table_clean = self.preprocessing(input_table)
        
        column_names = list(input_table_clean.columns)
        num_rows = len(input_table_clean)
        num_cols = len(column_names)
        
        # Build hypergraph and forward using hytrel_src defaults
        bigraph = self._convert_table_to_hytrel_format(input_table_clean)
        with torch.no_grad():
            bigraph = bigraph.to(self.device)
            outputs = self.model(bigraph)
            # hytrel_src.model.Encoder returns (embedding_s, embedding_t)
            if isinstance(outputs, tuple):
                _, embedding_t = outputs
            else:
                embedding_t = outputs

        # Use target hyperedge embeddings for columns: indices 1..num_cols (index 0 is table)
        column_embeddings_tensor = embedding_t[1:num_cols + 1]
        column_embeddings = column_embeddings_tensor.detach().cpu().numpy()

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
                target_embeddings = outputs      #
        
        return target_embeddings, num_rows, num_cols


    def _convert_table_to_hytrel_format(self, table: pd.DataFrame) -> BipartiteData:
        """Use hytrel_src's graph construction to build BipartiteData from a DataFrame.
        
        Args:
            table: Input DataFrame
        """
        # Use preprocessing to handle NaN and missing values
        table_clean = self.preprocessing(table)
        
        # Import functions from parallel_clean
        from parallel_clean import sanitize_text, clean_cell_value
        from data import CAP_TAG, HEADER_TAG, ROW_TAG
        
        # Extract caption if available (from DataFrame attributes or metadata)
        caption = None
        # Check DataFrame.attrs for caption (pandas >= 1.3.0 supports metadata)
        if hasattr(table_clean, 'attrs') and 'caption' in table_clean.attrs:
            caption = table_clean.attrs['caption']
        # Also check original table's attrs if it exists
        elif hasattr(table, 'attrs') and 'caption' in table.attrs:
            caption = table.attrs['caption']
        
        # Build text format using the same logic as json2string, but without MAX_ROW_LEN limit
        # Sanitize caption
        cap = '' if caption is None else str(caption)
        cap = sanitize_text(cap, entity='cap')
        if not cap:
            cap = HY_MISSING_CAP_TAG
        
        # Sanitize headers
        headers = [sanitize_text(str(h), entity='header') for h in table_clean.columns]
        header_text = ' | '.join(headers)
        
        # Sanitize cells using clean_cell_value (which calls sanitize_text)
        cells_data = table_clean.values.tolist()
        cells = [list(map(clean_cell_value, row)) for row in cells_data]
        cells_text = [' | '.join(row) for row in cells]
        
        #bypassing the json2string function from parallel_clean.py to avoid the MAX_ROW_LEN limit
        # Format matching parallel_clean.py json2string() format:
        # '<caption>CAP <header>HEADERS <row>ROW1 <row>ROW2...'
        text = ' '.join([CAP_TAG, cap, HEADER_TAG, header_text])
        cell_text = ' '.join([ROW_TAG + ' ' + row for row in cells_text])
        sample = ' '.join([text, cell_text])
        
        num_rows = len(table_clean)
        num_cols = len(table_clean.columns)
        
        # Determine checkpoint type from path or config to set correct flags
        checkpoint_path = getattr(self.cfg.approach, "checkpoint_path", None)
        is_contrast = False
        is_electra = False
        
        if checkpoint_path:
            # Try to infer from checkpoint path
            checkpoint_path_lower = str(checkpoint_path).lower()
            if 'contrast' in checkpoint_path_lower:
                is_contrast = True
            elif 'electra' in checkpoint_path_lower:
                is_electra = True
        
        # For inference, we use basic graph structure (no contrast/electra corruption needed)
        # But we can still detect the checkpoint type for logging/future use
        class _Args:
            max_token_length = self.max_token_length
            max_column_length = num_cols
            max_row_length = num_rows
            electra = False  # Always False for inference - no corruption needed
            contrast_bipartite_edge = False  # Always False for inference - no corrupted edges needed
            bipartite_edge_corrupt_ratio = 0.0
            num_workers = 0
            valid_ratio = 0.0
        
        # Log detected checkpoint type for reference
        if is_contrast or is_electra:
            logger.debug(f"Detected checkpoint type: {'contrast' if is_contrast else 'electra'} from path: {checkpoint_path}")

        # Initialize a lightweight TableDataModule to reuse its graph builders
        tdm = TableDataModule(tokenizer=self.tokenizer, data_args=_Args, seed=42, batch_size=1, py_logger=logger, objective='none')
        graphs = tdm._text2graph([sample])
        
        if len(graphs) == 0:
            # Debug: print the sample format to help diagnose parsing issues
            logger.error(f"Failed to parse table. Sample format: {sample[:200]}...")
            logger.error(f"Table shape: {table.shape}, headers: {headers[:5] if len(headers) > 5 else headers}")
            raise ValueError(f"Failed to construct graph from table. Parsing failed. "
                           f"Check that table has valid structure (non-empty rows/columns).")
        
        if len(graphs) > 1:
            logger.warning(f"Expected 1 graph but got {len(graphs)}, using first one")
        
        return graphs[0]

