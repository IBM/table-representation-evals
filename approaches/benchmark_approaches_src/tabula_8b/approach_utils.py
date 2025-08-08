import pandas as pd
import numpy as np
from typing import List, Dict, Any

def convert_row_to_string(row: pd.Series) -> str:
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

def preprocess_table_for_tabula(input_table: pd.DataFrame) -> List[str]:
    """
    Preprocess an entire table for TabuLA-8B embedding.
    
    Args:
        input_table: pd.DataFrame - The table to preprocess
        
    Returns:
        List[str]: List of preprocessed table row strings
    """
    preprocessed_data = []
    
    for _, row in input_table.iterrows():
        table_row_string = convert_row_to_string(row)
        preprocessed_data.append(table_row_string)
    
    return preprocessed_data

def validate_tabula_config(config: Dict[str, Any]) -> bool:
    """
    Validate the TabuLA-8B configuration parameters.
    
    Args:
        config: Dict[str, Any] - Configuration dictionary
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_params = ['model_name']
    optional_params = ['device', 'max_length', 'batch_size']
    
    # Check required parameters
    for param in required_params:
        if param not in config:
            print(f"Warning: Required parameter '{param}' not found in config")
            return False
    
    # Check optional parameters have valid values
    if 'max_length' in config and config['max_length'] <= 0:
        print("Error: max_length must be positive")
        return False
    
    if 'batch_size' in config and config['batch_size'] <= 0:
        print("Error: batch_size must be positive")
        return False
    
    return True 