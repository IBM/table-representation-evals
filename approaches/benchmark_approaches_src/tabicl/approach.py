from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
import logging
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import sys
import os

# Add TabICL source to Python path
tabicl_src_path = os.path.join(os.path.dirname(__file__), 'tabicl', 'src')
if tabicl_src_path not in sys.path:
    sys.path.insert(0, tabicl_src_path)

from tabicl import TabICLClassifier

logger = logging.getLogger(__name__)

class TabICLEmbedder(BaseTabularEmbeddingApproach):
    """
    TabICL embedding approach for tabular data.
    Uses the TabICL model to generate row embeddings for each row in a table.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = None
        logger.info("TabICLEmbedder: Initialized.")

    def load_trained_model(self):
        if self.model is None:
            logger.info("Loading TabICL model...")
            n_estimators = getattr(self.cfg.approach, "n_estimators", 32)
            # Use CPU with optimizations for Apple Silicon stability
            self.model = TabICLClassifier(
                n_estimators=n_estimators, 
                device="cpu",  # Use CPU for stability
            )
            logger.info(f"TabICL model loaded with n_estimators={n_estimators} on CPU with optimizations.")

    def preprocessing(self, input_table: pd.DataFrame):
        # No special preprocessing needed for TabICL, just return the DataFrame
        return input_table

    def get_row_embeddings(self, input_table: pd.DataFrame):
     
        self.load_trained_model()
        print("input_table shape:", input_table.shape)
        
        # Convert all columns to numerical values
        input_table_clean = input_table.copy()
        
        # Handle string columns by converting to categorical codes
        for col in input_table_clean.columns:
            if input_table_clean[col].dtype == 'object':
                # Convert string columns to categorical codes
                input_table_clean[col] = pd.Categorical(input_table_clean[col]).codes
            elif input_table_clean[col].dtype == 'category':
                # Convert categorical columns to codes
                input_table_clean[col] = input_table_clean[col].cat.codes
        
        # Handle NaN values by filling them
        input_table_clean = input_table_clean.fillna(0)  # Fill NaN with 0
        print("input_table_clean shape:", input_table_clean.shape)
        print("Data types after conversion:", input_table_clean.dtypes)
        
        y = np.zeros(len(input_table_clean))
        self.model.fit(input_table_clean, y)
        _, row_embeddings = self.model.predict_proba(input_table_clean)
        
        num_rows = input_table.shape[0]
        print("Raw row_embeddings shape:", row_embeddings.shape)
        print("Expected num_rows:", num_rows)
        
        # Handle ensemble members: reshape and average
        # row_embeddings shape: (n_ensemble_members * n_samples, embedding_dim)
        n_ensemble = row_embeddings.shape[1] // num_rows
        print("Number of ensemble members:", n_ensemble)
        
        # Reshape to (n_ensemble_members, n_samples, embedding_dim) and average
        single_row_embeddings = row_embeddings.reshape(n_ensemble*row_embeddings.shape[0], num_rows, -1).mean(axis=0)
        print("single_row_embeddings shape:", single_row_embeddings.shape)
        # since train and test is the same, take the first num_rows embeddings
        
        # Ensure the embeddings are in the correct numpy array format for sentence_transformers
        single_row_embeddings = np.array(single_row_embeddings, dtype=np.float32)
        
        return single_row_embeddings 

    def load_predictive_ml_model(self, train_df: pd.DataFrame, train_labels: pd.Series, task_type: str, dataset_information: dict):
        """
        Set up the TabICL model for predictive ML tasks.
        Args:
            train_df (pd.DataFrame): Training data.
            train_labels (pd.Series): Training labels.
            task_type (str): Either "classification" or "regression".
            dataset_information (dict): Additional dataset info.
        """
        if task_type == "classification":
            n_estimators = getattr(self.cfg.approach, "n_estimators", 32)
            # Use CPU with optimizations for Apple Silicon stability
            self.model = TabICLClassifier(
                n_estimators=n_estimators, 
                device="cpu",  # Use CPU for stability
            )
            
            # Preprocess the training data to handle categorical/string columns
            train_df_processed = self._preprocess_for_tabicl(train_df)
            self.model.fit(train_df_processed, train_labels)
        elif task_type == "regression":
            raise NotImplementedError("TabICLClassifier currently does not support regression tasks.")
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def predict_test_cases(self, test_df: pd.DataFrame, task_type: str):
        """
        Predict the target for the given test dataframe using the TabICL model directly.
        Args:
            test_df (pd.DataFrame): The input dataframe containing test cases for prediction.
            task_type (str): Either "classification" or "regression".
        Returns:
            np.ndarray or pd.DataFrame: Predictions as required by the benchmark framework.
        """
        if task_type == "classification":
            # Preprocess the test data to handle categorical/string columns
            test_df_processed = self._preprocess_for_tabicl(test_df)
            proba_tuple = self.model.predict_proba(test_df_processed)
            
            # TabICL returns logits, not probabilities - need to convert using softmax
            if isinstance(proba_tuple, tuple):
                logits = proba_tuple[0]  # First element contains the logits

                return logits
            else:
                # Fallback if not a tuple
                print(f"Is not tuple")
                return proba_tuple
        
        elif task_type == "regression":
            raise NotImplementedError("TabICLClassifier currently does not support regression tasks.")
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def _preprocess_for_tabicl(self, input_table: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input table for TabICL model by converting categorical/string columns to numerical codes.
        
        Args:
            input_table (pd.DataFrame): Input table to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed table with numerical values only
        """
        input_table_clean = input_table.copy()
        
        # Handle string columns by converting to categorical codes
        for col in input_table_clean.columns:
            if input_table_clean[col].dtype == 'object':
                # Convert string columns to categorical codes
                input_table_clean[col] = pd.Categorical(input_table_clean[col]).codes
            elif input_table_clean[col].dtype == 'category':
                # Convert categorical columns to codes
                input_table_clean[col] = input_table_clean[col].cat.codes
        
        # Handle NaN values by filling them
        input_table_clean = input_table_clean.fillna(0)  # Fill NaN with 0
        
        return input_table_clean 