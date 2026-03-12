#!/usr/bin/env python3
"""
SAP RPT-1-OSS (formerly ConTextTab) Embedding Approach for Tabular Data

This module implements the SAP RPT-1-OSS (A Semantics-Aware Tabular In-Context Learner)
approach for generating row embeddings from tabular data using semantic understanding.

Based on: https://github.com/SAP-samples/sap-rpt-1-oss
Paper: "ConTextTab: A Semantics-Aware Tabular In-Context Learner"
Note: This model was formerly known as ConTextTab and has been renamed to sap-rpt-1-oss
"""

# CRITICAL: Import PyTorch compatibility layer FIRST, before any other imports
# This patches torch.nn.attention before sap_rpt_oss tries to import it
import sys
import os
import importlib.util

# Apply compatibility patch before importing sap_rpt_oss
_compat_path = os.path.join(os.path.dirname(__file__), 'torch_compatibility.py')
if os.path.exists(_compat_path):
    spec = importlib.util.spec_from_file_location("torch_compatibility", _compat_path)
    torch_compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(torch_compat)
else:
    # Fallback: try importing as module
    try:
        from .torch_compatibility import *
    except ImportError:
        sys.path.append(os.path.dirname(__file__))
        from torch_compatibility import *

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach

# Configure logging
logger = logging.getLogger(__name__)


class SAP_RPT_OSS_Embedder(BaseTabularEmbeddingApproach):
    """
    SAP RPT-1-OSS embedding approach for tabular data.
    
    This class implements the SAP RPT-1-OSS (formerly ConTextTab) approach for generating
    row embeddings from tabular data using semantic understanding.
    
    Attributes:
        cfg (DictConfig): Configuration object
        model: SAP RPT-1-OSS model for generating embeddings
        device: Device to run the model on
    """
    
    def __init__(self, cfg: DictConfig):
        """Initialize the SAP RPT-1-OSS embedder.
        
        Args:
            cfg (DictConfig): Configuration object containing model parameters
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"SAP_RPT_OSS_Embedder initialized on device: {self.device}")

    def load_trained_model(self):
        """Load the pre-trained SAP RPT-1-OSS model."""
        if self.model is not None:
            return
            
        logger.info("Loading SAP RPT-1-OSS model...")
        try:
            from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor
            
            bagging = getattr(self.cfg.approach, "bagging", 1)
            max_context_size = getattr(self.cfg.approach, "max_context_size", 2048)
            
            self.model = {
                'classifier': SAP_RPT_OSS_Classifier(
                    bagging=bagging,
                    max_context_size=max_context_size
                ),
                'regressor': SAP_RPT_OSS_Regressor(
                    bagging=bagging,
                    max_context_size=max_context_size
                )
            }
            
            logger.info(f"SAP RPT-1-OSS model loaded with bagging={bagging}, max_context_size={max_context_size}")
            
        except ImportError as e:
            logger.error(f"Failed to import SAP RPT-1-OSS: {e}")
            raise ImportError("SAP RPT-1-OSS not found. Please install it with: pip install git+https://github.com/SAP-samples/sap-rpt-1-oss")
        except Exception as e:
            logger.error(f"Failed to load SAP RPT-1-OSS model: {e}")
            raise RuntimeError(f"Failed to initialize SAP RPT-1-OSS model: {e}")

    def preprocessing(self, input_table: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input table for SAP RPT-1-OSS model.
        
        Args:
            input_table (pd.DataFrame): Input table to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed table ready for SAP RPT-1-OSS processing
        """
        input_table_clean = input_table.copy()
        for col in input_table_clean.columns:
            if input_table_clean[col].dtype != 'object':
                input_table_clean[col] = input_table_clean[col].astype(str)
        
        return input_table_clean

    def get_row_embeddings(self, input_table: pd.DataFrame, train_size: int = None, train_labels: pd.Series = None) -> np.ndarray:
        """Generate row embeddings using SAP RPT-1-OSS model.
        
        Args:
            input_table (pd.DataFrame): Input table with rows to embed
            train_size (int, optional): Number of rows to use for training context
            train_labels (pd.Series, optional): Labels for training rows
            
        Returns:
            np.ndarray: Row embeddings of shape (num_rows, embedding_dim) or (test_rows, embedding_dim)
        """
        self.load_trained_model()
        input_table_clean = self.preprocessing(input_table)
        
        logger.info(f"Processing {len(input_table_clean)} rows for embedding generation")
        embeddings = self._extract_embeddings_from_model(input_table_clean, train_size=train_size, train_labels=train_labels)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def _extract_embeddings_from_model(self, input_table: pd.DataFrame, aggregate_tokens: bool = True,
                                      train_size: int = None, train_labels: pd.Series = None) -> np.ndarray:
        """Extract embeddings directly from SAP RPT-1-OSS's internal representations.
        
        For row embeddings, we use all columns as both context and query.
        The tokenizer expects X_query to have the same columns as X_context.
        
        Args:
            input_table (pd.DataFrame): Input table with rows to embed
            aggregate_tokens (bool): If True, average across tokens for row embeddings.
                                    If False, return token-level embeddings for cells.
            train_size (int, optional): Number of rows to use for training context
            train_labels (pd.Series, optional): Labels for training rows
            
        Returns:
            np.ndarray: Row embeddings from the model's encoder outputs
                       Shape: [num_rows, embedding_dim] if aggregate_tokens=True
                              [num_rows, num_cols, embedding_dim] if aggregate_tokens=False
        """
        # Determine if we should use real labels or dummy labels
        if train_labels is not None:
            # Convert labels to strings to ensure consistent typing for numpy operations
            train_labels_str = train_labels.astype(str)
            
            # Fit the model with training data and labels
            train_df = input_table[:len(train_labels_str)].copy() if train_size is None else input_table[:train_size].copy()
            self.model['classifier'].fit(train_df, train_labels_str)
            
            # Use real labels for training portion
            if train_size is not None:
                # Inference phase: train on training portion, extract embeddings for test portion
                X_context = input_table[:train_size].copy()
                y_context = train_labels_str.to_frame()
                X_query = input_table[train_size:].copy()
                y_query = pd.Series(['0'] * len(X_query), name='target').to_frame()
            else:
                # Training phase: use all data with real labels
                X_context = input_table.copy()
                y_context = train_labels_str.to_frame()
                X_query = input_table.copy()
                y_query = train_labels_str.to_frame()
        else:
            # Use dummy labels when train_labels not provided (e.g., for cell embeddings)
            dummy_labels = pd.Series([0] * len(input_table), name='target')
            X_context = input_table.copy()
            X_query = input_table.copy()
            y_context = dummy_labels.to_frame()
            y_query = dummy_labels.to_frame()
        
        try:
            data, labels, label_classes = self.model['classifier'].tokenizer(
                X_context, y_context, X_query, y_query,
                self.model['classifier'].classification_or_regression
            )
        except AttributeError as e:
            logger.error(f"Tokenizer API may differ from ConTextTab: {e}")
            logger.error("You may need to adjust this method based on the actual SAP_RPT_OSS API")
            raise NotImplementedError(f"SAP_RPT_OSS tokenizer API differs from expected. Error: {e}")
        
        device = next(self.model['classifier'].model.parameters()).device
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        with torch.no_grad():
            input_embeds = self.model['classifier'].model.embeddings(data, False)
            extended_attention_mask = self.model['classifier'].model.build_context_attention_mask(
                data, input_embeds.device)
            extended_attention_mask = extended_attention_mask.type(input_embeds.dtype)
            
            encoder_outputs = input_embeds
            for layer in self.model['classifier'].model.in_context_encoder:
                encoder_outputs = layer(encoder_outputs, extended_attention_mask)
            
            # Extract query embeddings
            num_context_rows = len(X_context)
            num_cols = len(input_table.columns)
            query_embeddings = encoder_outputs[num_context_rows:, :, :]
            
            # Exclude the target column token (last token) - only keep feature column tokens
            # query_embeddings shape: [num_query_rows, num_cols + 1, embedding_dim]
            # We want: [num_query_rows, num_cols, embedding_dim]
            query_embeddings = query_embeddings[:, :num_cols, :]
            
            if aggregate_tokens:
                # For row embeddings: average across tokens (feature columns only)
                row_embeddings = query_embeddings.mean(dim=1)
                return row_embeddings.cpu().to(torch.float32).numpy()
            else:
                # For cell embeddings: return token-level embeddings (feature columns only)
                return query_embeddings.cpu().to(torch.float32).numpy()

    def load_predictive_ml_model(self, train_df: pd.DataFrame, train_labels: pd.Series, 
                                task_type: str, dataset_information: dict):
        """Set up the SAP RPT-1-OSS model for predictive ML tasks.
        
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
        """Predict the target for the given test dataframe using the SAP RPT-1-OSS model.
        
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
        """Generate column embeddings using SAP RPT-1-OSS model.
        
        For each column j, treat all other columns as context and column j as the query
        Run the in-context encoder to obtain contextualized representations
        Take the representation of the query column (assumed last position) for all rows and
        average across rows to yield a single embedding per column
        
        Returns:
            (np.ndarray, list[str]): (column_embeddings [num_cols, hidden], column_names)
        """
        self.load_trained_model()
        table = self.preprocessing(input_table)
        column_names = list(table.columns)
        num_cols = len(column_names)

        device = next(self.model['classifier'].model.parameters()).device
        col_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for col_idx, col_name in enumerate(column_names):
                X_context = table.drop(columns=[col_name])
                Y_query = table[[col_name]]
                dummy_labels = pd.Series([0] * len(table), name='target')
                try:
                    data, labels, label_classes = self.model['classifier'].tokenizer(
                        X_context,
                        dummy_labels.to_frame(),
                        X_context.copy(),
                        dummy_labels.to_frame(),
                        self.model['classifier'].classification_or_regression,
                    )
                except AttributeError as e:
                    logger.error(f"Tokenizer API may differ from ConTextTab: {e}")
                    logger.error("You may need to adjust this method based on the actual SAP_RPT_OSS API")
                    raise NotImplementedError(f"SAP_RPT_OSS tokenizer API differs from expected. Error: {e}")

                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                input_embeds = self.model['classifier'].model.embeddings(data, False)
                attn_mask = self.model['classifier'].model.build_context_attention_mask(data, input_embeds.device)
                attn_mask = attn_mask.type(input_embeds.dtype)

                encoder_outputs = input_embeds
                for layer in self.model['classifier'].model.in_context_encoder:
                    encoder_outputs = layer(encoder_outputs, attn_mask)

                query_reps = encoder_outputs[:, -1, :]
                col_emb = query_reps.mean(dim=0)
                col_embeddings.append(col_emb.detach().cpu().float().numpy())

        column_embeddings = np.vstack(col_embeddings) if col_embeddings else np.zeros((0, 0), dtype=np.float32)
        
        return column_embeddings, column_names

