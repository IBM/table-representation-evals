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
            use_memory_efficient = getattr(self.cfg.approach, "use_memory_efficient_model", True)
            
            # Use CPU with optimizations for Apple Silicon stability
            model_kwargs = {
                "n_estimators": n_estimators,
                "device": "cpu",
            }
            
            # Memory optimization is now handled through subsampling in get_column_embeddings
            if use_memory_efficient:
                # Reduce batch size to minimize memory usage during inference
                model_kwargs.update({
                    "batch_size": 4,  # Reduce from default 8 to 4 for memory efficiency
                })
                logger.info(f"TabICL model loaded with n_estimators={n_estimators}, batch_size=4 on CPU with memory optimizations.")
            else:
                logger.info(f"TabICL model loaded with n_estimators={n_estimators} on CPU.")
            
            self.model = TabICLClassifier(**model_kwargs)

    def preprocessing(self, input_table: pd.DataFrame):
        # No special preprocessing needed for TabICL, just return the DataFrame
        return input_table

    def get_row_embeddings(self, input_table: pd.DataFrame):
        
        self.load_trained_model()
        print("input_table shape:", input_table.shape)
        
        input_table_clean = self._preprocess_for_tabicl(input_table)

        # Use the same data for both training and testing since we only want embeddings
        y = np.zeros(len(input_table_clean))
        self.model.fit(input_table_clean, y)
        _, row_embeddings, _ = self.model.predict_proba(input_table_clean)
        
        # TabICL returns embeddings for both training and test samples
        # We only want the test sample embeddings (second half of the output)
        n_samples = len(input_table_clean)
        test_embeddings = row_embeddings[n_samples:]  # Extract only test sample embeddings
        
        print("single_row_embeddings shape:", test_embeddings.shape)
        
        # Ensure the embeddings are in the correct numpy array format
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

    def get_column_embeddings(self, input_table: pd.DataFrame) -> tuple:
        """
        Generate column embeddings using TabICL's ColEmbedding stage.
        
        Args:
            input_table (pd.DataFrame): Input table with columns to embed
            
        Returns:
            tuple: (column_embeddings, column_names) where column_embeddings has shape (num_columns, embedding_dim)
        """
        # Ensure model is loaded

        # Set PyTorch to use single thread to avoid threading issues during model initialization
        # This prevents segfaults that occur with nn.init.trunc_normal_() in multithreaded contexts
        import torch
        torch.set_num_threads(1)
        #torch.set_num_interop_threads(1)
        
        self.load_trained_model()
        
        print("input_table shape:", input_table.shape)
        
        # Subsample rows for column embedding extraction 
        if len(input_table) > max_rows_for_embeddings:
            print(f"Subsampling from {len(input_table)} to {max_rows_for_embeddings} rows for column embeddings")
            input_table = input_table.sample(n=max_rows_for_embeddings, random_state=42)
        
        # Convert all columns to numerical values
        input_table_clean = self._preprocess_for_tabicl(input_table)
        
        print(f"input_table_clean shape after preprocessing: {input_table_clean.shape}")
        print(f"Original columns: {len(input_table.columns)}, Clean table columns: {len(input_table_clean.columns)}")
        
        # Prepare dummy labels
        y = np.zeros(len(input_table_clean))
        
        # Fit the model on each table individually
        # Note: We need to fit for each table because the columns might be different
        logger.info("Fitting model for column embeddings")
        self.model.fit(input_table_clean, y)
        
        # Get column embeddings using the fitted model
        _, _, column_embeddings = self.model.predict_proba(input_table_clean)
        
        print(f"column_embeddings shape: {column_embeddings.shape}")
        
        return column_embeddings, input_table_clean.columns

