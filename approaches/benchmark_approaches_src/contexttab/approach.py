#!/usr/bin/env python3
"""
ConTextTab Embedding Approach for Tabular Data

This module implements the ConTextTab (A Semantics-Aware Tabular In-Context Learner)
approach for generating row embeddings from tabular data using semantic understanding.

Based on: https://github.com/SAP-samples/contexttab
Paper: "ConTextTab: A Semantics-Aware Tabular In-Context Learner"
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

# Import PyTorch compatibility layer before any other imports
try:
    from .torch_compatibility import *
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from torch_compatibility import *

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach

# Configure logging
logger = logging.getLogger(__name__)


class ConTextTabEmbedder(BaseTabularEmbeddingApproach):
    """
    ConTextTab embedding approach for tabular data.
    
    This class implements the ConTextTab (A Semantics-Aware Tabular In-Context Learner)
    approach for generating row embeddings from tabular data using semantic understanding.
    
    ConTextTab combines the best of table-native ICL frameworks with semantic understanding
    by employing specialized embeddings for different data modalities and training on 
    large-scale real-world tabular data.
    
    Attributes:
        cfg (DictConfig): Configuration object
        model: ConTextTab model for generating embeddings
        device: Device to run the model on
    """
    
    def __init__(self, cfg: DictConfig):
        """Initialize the ConTextTab embedder.
        
        Args:
            cfg (DictConfig): Configuration object containing model parameters
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"ConTextTabEmbedder initialized on device: {self.device}")

    def load_trained_model(self):
        """Load the pre-trained ConTextTab model."""
        if self.model is not None:
            return
            
        logger.info("Loading ConTextTab model...")
        try:
            from contexttab import ConTextTabClassifier, ConTextTabRegressor
            
            # Get model configuration from config or use defaults
            bagging = getattr(self.cfg.approach, "bagging", 1)
            max_context_size = getattr(self.cfg.approach, "max_context_size", 2048)
            test_chunk_size = getattr(self.cfg.approach, "test_chunk_size", 1000)
            
            # Initialize both classifier and regressor
            self.model = {
                'classifier': ConTextTabClassifier(
                    bagging=bagging,
                    max_context_size=max_context_size,
                    test_chunk_size=test_chunk_size
                ),
                'regressor': ConTextTabRegressor(
                    bagging=bagging,
                    max_context_size=max_context_size,
                    test_chunk_size=test_chunk_size
                )
            }
            
            logger.info(f"ConTextTab model loaded with bagging={bagging}, max_context_size={max_context_size}")
            
        except ImportError as e:
            logger.error(f"Failed to import ConTextTab: {e}")
            raise ImportError("ConTextTab not found. Please install it with: pip install git+https://github.com/SAP-samples/contexttab")
        except Exception as e:
            logger.error(f"Failed to load ConTextTab model: {e}")
            raise RuntimeError(f"Failed to initialize ConTextTab model: {e}")

    def preprocessing(self, input_table: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input table for ConTextTab model.
        
        Args:
            input_table (pd.DataFrame): Input table to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed table ready for ConTextTab processing
        """
      
        input_table_clean = input_table.copy()
        
        # Convert any non-string columns to string for consistency
        for col in input_table_clean.columns:
            if input_table_clean[col].dtype != 'object':
                input_table_clean[col] = input_table_clean[col].astype(str)
        
        return input_table_clean

    def get_row_embeddings(self, input_table: pd.DataFrame) -> np.ndarray:
        """Generate row embeddings using ConTextTab model.
        
        Args:
            input_table (pd.DataFrame): Input table with rows to embed
            
        Returns:
            np.ndarray: Row embeddings of shape (num_rows, embedding_dim)
        """
        self.load_trained_model()
        input_table_clean = self.preprocessing(input_table)
        
        logger.info(f"Processing {len(input_table_clean)} rows for embedding generation")
        # Extract embeddings directly from the model's internal representations
        embeddings = self._extract_embeddings_from_model(input_table_clean)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def _extract_embeddings_from_model(self, input_table: pd.DataFrame) -> np.ndarray:
        """Extract embeddings directly from ConTextTab's internal representations.
        
        Uses the last column as the query for in-context learning, which is the most
        natural approach for ConTextTab's architecture.
        
        Args:
            input_table (pd.DataFrame): Input table with rows to embed
            
        Returns:
            np.ndarray: Row embeddings from the model's encoder outputs
        """
        # Use the last column as the query 
        dummy_labels = pd.Series([0] * len(input_table), name='target')
        
        # Context: all columns except the last, Query: last column
        X_train = input_table.iloc[:, :-1]  # All columns except the last (context)
        
        # Get tokenized data using the model's internal tokenizer
        data, labels, label_classes = self.model['classifier'].tokenizer(
            X_train, dummy_labels.to_frame(), input_table.iloc[:, -1:], dummy_labels.to_frame(),
            self.model['classifier'].classification_or_regression
        )
        
        # Move to device
        device = next(self.model['classifier'].model.parameters()).device
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        # Get the model's internal representations
        with torch.no_grad():
            # Get embeddings from the model
            input_embeds = self.model['classifier'].model.embeddings(data, False)  # False = not regression
            
            # Build attention mask for 2D attention
            extended_attention_mask = self.model['classifier'].model.build_context_attention_mask(
                data, input_embeds.device)
            extended_attention_mask = extended_attention_mask.type(input_embeds.dtype)
            
            # Pass through the encoder layers to get contextualized representations
            encoder_outputs = input_embeds
            for layer in self.model['classifier'].model.in_context_encoder:
                encoder_outputs = layer(encoder_outputs, extended_attention_mask)
            
            # Extract row embeddings by averaging across all columns
            # encoder_outputs shape: (num_rows, num_columns, hidden_size)
            # Average across the column dimension to get a single embedding per row
            row_embeddings = encoder_outputs.mean(dim=1)  # Average across all columns
            
            return row_embeddings.cpu().numpy()

    def load_predictive_ml_model(self, train_df: pd.DataFrame, train_labels: pd.Series, 
                                task_type: str, dataset_information: dict):
        """Set up the ConTextTab model for predictive ML tasks.
        
        Args:
            train_df (pd.DataFrame): Training data
            train_labels (pd.Series): Training labels
            task_type (str): Either "classification" or "regression"
            dataset_information (dict): Additional dataset info
        """
        self.load_trained_model()
        
        if task_type == "classification":
            self.model['classifier'].fit(train_df, train_labels)
            self.active_model = 'classifier'
        elif task_type == "regression":
            self.model['regressor'].fit(train_df, train_labels)
            self.active_model = 'regressor'
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def predict_test_cases(self, test_df: pd.DataFrame, task_type: str) -> np.ndarray:
        """Predict the target for the given test dataframe using the ConTextTab model.
        
        Args:
            test_df (pd.DataFrame): The input dataframe containing test cases for prediction
            task_type (str): Either "classification" or "regression"
            
        Returns:
            np.ndarray: Predictions as required by the benchmark framework
        """
        if task_type == "classification":
            return self.model['classifier'].predict_proba(test_df)
        elif task_type == "regression":
            return self.model['regressor'].predict(test_df)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def get_column_embeddings(self, input_table: pd.DataFrame) -> tuple:
        
        """Generate column embeddings using ConTextTab model.
        
          For each column j, treat all other columns as context and column j as the query
          Run the in-context encoder to obtain contextualized representations
          Take the representation of the query column (assumed last position) for all rows and
          average across rows to yield a single embedding per column
        
        Returns:
            (np.ndarray, list[str]): (column_embeddings [num_cols, hidden], column_names)
        """

        self.load_trained_model()
        table = self.preprocessing(input_table)
        # Limit to 1000 rows for efficiency/consistency
        if len(table) > 100:
            table = table.head(100)
        column_names = list(table.columns)
        num_cols = len(column_names)

        device = next(self.model['classifier'].model.parameters()).device
        col_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for col_idx, col_name in enumerate(column_names):
                # Context: all columns except current; Query: current column
                X_context = table.drop(columns=[col_name])
                Y_query = table[[col_name]]
                dummy_labels = pd.Series([0] * len(table), name='target')

                # Tokenize using the classifier tokenizer path
                data, labels, label_classes = self.model['classifier'].tokenizer(
                    X_context,
                    dummy_labels.to_frame(),
                    Y_query,
                    dummy_labels.to_frame(),
                    self.model['classifier'].classification_or_regression,
                )

                # Move tensors to device
                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

                # Build embeddings and attention mask
                input_embeds = self.model['classifier'].model.embeddings(data, False)
                attn_mask = self.model['classifier'].model.build_context_attention_mask(data, input_embeds.device)
                attn_mask = attn_mask.type(input_embeds.dtype)

                # Forward through in-context encoder
                encoder_outputs = input_embeds
                for layer in self.model['classifier'].model.in_context_encoder:
                    encoder_outputs = layer(encoder_outputs, attn_mask)

                # encoder_outputs shape expected: (num_rows, num_columns_packed, hidden)
                # We assume the last position corresponds to the query column representation
                query_reps = encoder_outputs[:, -1, :]  # (num_rows, hidden)
                col_emb = query_reps.mean(dim=0)  # (hidden,)
                col_embeddings.append(col_emb.detach().cpu().numpy())

        column_embeddings = np.vstack(col_embeddings) if col_embeddings else np.zeros((0, 0), dtype=np.float32)
        
        return column_embeddings, column_names