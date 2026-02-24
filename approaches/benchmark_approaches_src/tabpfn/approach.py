from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
import logging
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor

logger = logging.getLogger(__name__)

class TabPFNEmbedder(BaseTabularEmbeddingApproach):
    """
    TabPFN embedding approach for tabular data.
    Uses the TabPFN model to generate row embeddings for each row in a table.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = None
        self.classifier = None
        self.regressor = None
        logger.info("TabPFNEmbedder: Initialized.")

    def load_trained_model(self):
        if self.model is None:
            logger.info("Loading TabPFN models...")
            device = getattr(self.cfg.approach, "device", "cuda")
            self.classifier = TabPFNClassifier(device=device)
            self.regressor = TabPFNRegressor(device=device)
            self.model = self.classifier  # Default to classifier for embeddings
            logger.info(f"TabPFN models loaded with device={device}.")

    def preprocessing(self, input_table: pd.DataFrame):
        """Preprocess data for TabPFN by converting categorical columns to numerical."""
        # TabPFN requires all features to be numerical
        processed_table = input_table.copy()
        
        # Convert categorical columns to numerical codes
        for col in processed_table.columns:
            if processed_table[col].dtype == 'object' or processed_table[col].dtype.name == 'category':
                # Use label encoding for categorical columns
                processed_table[col] = pd.Categorical(processed_table[col]).codes
                # Handle any remaining NaN values
                processed_table[col] = processed_table[col].fillna(-1)
        
        # Fill any remaining NaN values with 0
        processed_table = processed_table.fillna(0)
        
        return processed_table

    def get_row_embeddings(self, input_table: pd.DataFrame, train_size: int = None, train_labels: np.ndarray = None):
        """
        Get row embeddings using TabPFN's embedding extractor.
        
        Args:
            input_table (pd.DataFrame): Input table to extract embeddings from
            train_size (int, optional): Number of rows to use for training
            train_labels (np.ndarray, optional): Labels for training rows
            
        Returns:
            np.ndarray: Row embeddings for test portion (or all rows if no train_size)
        """
        self.load_trained_model()
        
        # Preprocess the input table
        processed_table = self.preprocessing(input_table)
        
        # If train_labels are provided, use them for fitting
        if train_labels is not None:
            if train_size is not None:
                # Inference phase: Fit on training portion, extract embeddings for test portion
                self.model.fit(processed_table[:train_size], train_labels)
                # Extract embeddings for all rows
                all_embeddings = self.model.get_embeddings(processed_table, data_source='test')
                # Extract embeddings for rows after train_size
                # Handle ensemble output: slice on the correct axis (axis 1 for samples)
                if len(all_embeddings.shape) == 3:  # (n_estimators, n_samples, embedding_dim)
                    test_embeddings = all_embeddings[:, train_size:, :]
                else:  # (n_samples, embedding_dim)
                    test_embeddings = all_embeddings[train_size:]
            else:
                # Training phase: Fit on all data with provided labels, return embeddings for same data
                self.model.fit(processed_table, train_labels)
                # Extract embeddings for the training data itself
                test_embeddings = self.model.get_embeddings(processed_table, data_source='train')
        else:
            # Use dummy labels when train_labels not provided
            # Old behavior: use all data as training with dummy labels
            y_dummy = np.zeros(len(processed_table))
            self.model.fit(processed_table, y_dummy)
            # Extract embeddings for all rows
            test_embeddings = self.model.get_embeddings(processed_table, data_source='test')
        
        # Handle ensemble output: average across ensemble members
        if len(test_embeddings.shape) == 3:  # (n_estimators, n_samples, embedding_dim)
            test_embeddings = test_embeddings.mean(axis=0)  # Average across ensemble dimension
        
        logger.info(f"Extracted TabPFN embeddings with shape: {test_embeddings.shape}")
        
        return np.array(test_embeddings, dtype=np.float32)

    def setup_model_for_task(self, train_df: pd.DataFrame, train_labels: pd.Series, task_type: str, dataset_information: dict):
        """
        Set up the TabPFN model for predictive ML tasks.
        Args:
            train_df (pd.DataFrame): Training data.
            train_labels (pd.Series): Training labels.
            task_type (str): Either "classification" or "regression".
            dataset_information (dict): Additional dataset info.
        """
        self.load_trained_model()
        
        # Preprocess the training data
        processed_train_df = self.preprocessing(train_df)
        
        if task_type == "classification":
            self.model = self.classifier
            self.model.fit(processed_train_df, train_labels)
            logger.info("TabPFN classifier fitted for classification task.")
        elif task_type == "regression":
            self.model = self.regressor
            self.model.fit(processed_train_df, train_labels)
            logger.info("TabPFN regressor fitted for regression task.")
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def predict_test_tables(self, test_df: pd.DataFrame, task_type: str):
        """
        Predict the target for the given test dataframe using the TabPFN model.
        Args:
            test_df (pd.DataFrame): The input dataframe containing test cases for prediction.
            task_type (str): Either "classification" or "regression".
        Returns:
            np.ndarray: Predictions as required by the benchmark framework.
        """
        # Preprocess the test data
        processed_test_df = self.preprocessing(test_df)
        
        if task_type == "classification":
            # For classification, return probabilities for all classes
            proba = self.model.predict_proba(processed_test_df)
            return proba  # Return all class probabilities
                
        elif task_type == "regression":
            # For regression, return the predicted values
            return self.model.predict(processed_test_df)
        else:
            raise ValueError(f"Unknown task_type: {task_type}") 