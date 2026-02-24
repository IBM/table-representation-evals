#!/usr/bin/env python3
"""
Cell Embedding Component for SAP RPT-1-OSS

This module implements cell-level embeddings using the SAP RPT-1-OSS model.
It reuses the embedding extraction logic from the main approach but returns
embeddings for each cell without averaging across tokens.
"""

import pandas as pd
import logging
import numpy as np

from benchmark_src.approach_interfaces.cell_embedding_interface import CellEmbeddingInterface

logger = logging.getLogger(__name__)


class CellEmbeddingComponent(CellEmbeddingInterface):
    """
    Cell embedding component for SAP RPT-1-OSS.
    
    This component generates embeddings for each cell in a table by extracting
    representations from the SAP RPT-1-OSS model's encoder without averaging
    across tokens (unlike row embeddings which do average).
    """

    def __init__(self, approach_instance):
        """Initialize the cell embedding component.
        
        Args:
            approach_instance: Instance of SAP_RPT_OSS_Embedder with loaded model
        """
        self.approach_instance = approach_instance

    def setup_model_for_task(self, dataset_information: dict):
        """Setup the model for cell embedding task.
        
        Args:
            dataset_information (dict): Additional information about the dataset
        """
        self.approach_instance.load_trained_model()
        logger.info("SAP RPT-OSS model loaded for cell embedding task")

    def create_cell_embeddings_for_table(self, input_table: pd.DataFrame) -> np.ndarray:
        """Create embeddings for each cell in the table.
        
        This method extracts cell-level embeddings from the SAP RPT-1-OSS model's
        encoder outputs. The tokenizer creates one token per column (with target column
        excluded in approach.py), so token embeddings directly correspond to cell embeddings.
        
        The output includes a header row (row 0) with column name embeddings,
        followed by data cell embeddings (rows 1+).
        
        Args:
            input_table (pd.DataFrame): Input table with cells to embed
            
        Returns:
            np.ndarray: Cell embeddings with shape [num_rows+1, num_cols, embedding_dim]
                       Row 0 contains header embeddings, rows 1+ contain data cell embeddings
        """
        num_rows, num_cols = input_table.shape
        logger.info(f"Creating cell embeddings for table of shape {num_rows}x{num_cols}")
        
        # Preprocess the input table
        input_table_clean = self.approach_instance.preprocessing(input_table)
        
        # Extract token-level embeddings for data cells (without aggregation)
        # Shape: [num_rows, num_cols, embedding_dim] (target column already excluded)
        data_cell_embeddings = self.approach_instance._extract_embeddings_from_model(
            input_table_clean, aggregate_tokens=False
        )
        logger.info(f"Cell embeddings shape: {data_cell_embeddings.shape}")
        
        # Create header embeddings (one per column)
        header_embeddings = self._create_header_embeddings(input_table_clean)
        
        # Combine header and data embeddings: [1 + num_rows, num_cols, embedding_dim]
        cell_embeddings = np.vstack([header_embeddings[np.newaxis, :, :], data_cell_embeddings])
        
        logger.info(f"Generated cell embeddings with shape: {cell_embeddings.shape}")
        return cell_embeddings

# Made with Bob

    def _create_header_embeddings(self, input_table: pd.DataFrame) -> np.ndarray:
        """Create embeddings for column headers.
        
        This creates a simple embedding for each column header by treating
        each column name as a single-row table and extracting its embedding.
        
        Args:
            input_table (pd.DataFrame): Preprocessed input table
            
        Returns:
            np.ndarray: Header embeddings with shape [num_cols, embedding_dim]
        """
        num_cols = len(input_table.columns)
        embedding_dim = None
        header_embeddings_list = []
        
        # For each column, create a single-row table with just that column
        # and extract its embedding
        for col_name in input_table.columns:
            # Create a single-row table with this column
            single_col_table = pd.DataFrame({col_name: [str(col_name)]})
            
            # Extract embedding for this single cell
            col_token_embeddings = self.approach_instance._extract_embeddings_from_model(
                single_col_table, aggregate_tokens=False
            )
            
            # Average across tokens to get a single embedding for the header
            # Shape: [1, num_tokens, embedding_dim] -> [embedding_dim]
            header_emb = col_token_embeddings[0].mean(axis=0)
            header_embeddings_list.append(header_emb)
            
            if embedding_dim is None:
                embedding_dim = header_emb.shape[0]
        
        # Stack into [num_cols, embedding_dim]
        header_embeddings = np.stack(header_embeddings_list, axis=0)
        
        return header_embeddings
