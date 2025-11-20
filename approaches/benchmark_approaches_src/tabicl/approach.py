from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
import logging
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import sys
import os
import torch

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
            use_memory_efficient = getattr(self.cfg.approach, "use_memory_efficient_model", True)
            device = getattr(self.cfg.approach, "device", "cpu")

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_kwargs = {
                "n_estimators": n_estimators,
                "device": device,
            }
            
            if use_memory_efficient:
                model_kwargs.update({
                    "batch_size": 4,
                })
                logger.info(f"TabICL model loaded with n_estimators={n_estimators}, batch_size=4 on CPU with memory optimizations.")
            else:
                logger.info(f"TabICL model loaded with n_estimators={n_estimators} on CPU.")
            
            self.model = TabICLClassifier(**model_kwargs)

    def preprocessing(self, input_table: pd.DataFrame):
        return input_table

    def get_row_embeddings(self, input_table: pd.DataFrame):
        
        self.load_trained_model()
        print("input_table shape:", input_table.shape)
        
        input_table_clean = self._preprocess_for_tabicl(input_table)

        y = np.zeros(len(input_table_clean))
        self.model.fit(input_table_clean, y)
        _, row_embeddings, _ = self.model.predict_proba(input_table_clean)
        
        n_samples = len(input_table_clean)
        test_embeddings = row_embeddings[n_samples:]
        
        print("single_row_embeddings shape:", test_embeddings.shape)
        single_row_embeddings = np.array(test_embeddings, dtype=np.float32)
        
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
            self.model = TabICLClassifier(
                n_estimators=n_estimators, 
                device="cpu",
            )
            
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
            test_df_processed = self._preprocess_for_tabicl(test_df)
            proba_tuple = self.model.predict_proba(test_df_processed)
            
            if isinstance(proba_tuple, tuple):
                logits = proba_tuple[0]
                return logits
            else:
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
        
        for col in input_table_clean.columns:
            if input_table_clean[col].dtype == 'object':
                input_table_clean[col] = pd.Categorical(input_table_clean[col]).codes
            elif input_table_clean[col].dtype == 'category':
                input_table_clean[col] = input_table_clean[col].cat.codes
        
        input_table_clean = input_table_clean.fillna(0)
        
        return input_table_clean 

    def get_column_embeddings(self, input_table: pd.DataFrame) -> tuple:
        """
        Generate column embeddings using TabICL's ColEmbedding stage.
        
        Args:
            input_table (pd.DataFrame): Input table with columns to embed
            
        Returns:
            tuple: (column_embeddings, column_names) where column_embeddings has shape (num_columns, embedding_dim)
        """
        # Ensure model is loaded
        # NOTE: this is not needed for TabICL, uncomment if running into threading issue

        #import torch
        #torch.set_num_threads(1)
        
        self.load_trained_model()
        
        print("input_table shape:", input_table.shape)
        
        input_table_clean = self._preprocess_for_tabicl(input_table)
        
        print(f"input_table_clean shape after preprocessing: {input_table_clean.shape}")
        print(f"Original columns: {len(input_table.columns)}, Clean table columns: {len(input_table_clean.columns)}")
        
        y = np.zeros(len(input_table_clean))
        logger.info("Fitting model for column embeddings")
        self.model.fit(input_table_clean, y)
        _, _, column_embeddings = self.model.predict_proba(input_table_clean)
        
        print(f"column_embeddings shape: {column_embeddings.shape}")
        print(f"Number of column names: {len(input_table_clean.columns)}")
        
        return column_embeddings, input_table_clean.columns

