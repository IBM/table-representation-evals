from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
import logging
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import tempfile
import os

logger = logging.getLogger(__name__)

class MITRAApproach(BaseTabularEmbeddingApproach):
    """
    MITRA (Multimodal Tabular Transformer) approach for tabular data.
    Uses AutoGluon's TabularPredictor with MITRA model for predictive ML tasks.
    Classification: https://huggingface.co/autogluon/mitra-classifier
    Regression: https://huggingface.co/autogluon/mitra-regressor
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.predictor = None
        self.task_type = None
        self.temp_dir = None
        logger.info("MITRAApproach: Initialized.")

    def preprocessing(self, input_table: pd.DataFrame):
        """Basic preprocessing for MITRA."""
        return input_table

    def load_predictive_ml_model(self, train_df: pd.DataFrame, train_labels: pd.Series, task_type: str, dataset_information: dict):
        """
        Set up the MITRA model for predictive ML tasks using AutoGluon.
        
        Args:
            train_df (pd.DataFrame): Training data.
            train_labels (pd.Series): Training labels.
            task_type (str): Either "classification" or "regression".
            dataset_information (dict): Additional dataset info including:
                - categorical_indicator: list of booleans indicating categorical features
                - num_classes: int, number of classes for classification
        """
        self.task_type = task_type
        
        if task_type == "classification":
            num_classes = dataset_information.get("num_classes", len(train_labels.unique()))
            
            logger.info(f"Setting up MITRA model for classification")
            logger.info(f"Number of classes: {num_classes}")
            logger.info(f"Training samples: {len(train_df)}")
            
            # Create a temporary directory for AutoGluon
            self.temp_dir = tempfile.mkdtemp(prefix="mitra_")
            logger.info(f"Using temporary directory: {self.temp_dir}")
            
            # Combine training data and labels
            train_data = train_df.copy()
            train_data['target'] = train_labels
            
            # Initialize TabularPredictor with MITRA
            self.predictor = TabularPredictor(
                label='target',
                path=self.temp_dir,
                verbosity=2
            )
            
            # Train with MITRA model
            # According to AutoGluon docs, just specify 'MITRA' in hyperparameters
            # Set ag.max_memory_usage_ratio to allow using more memory
            hyperparameters: dict[str, dict[str, bool]]= {'MITRA': {'fine_tune': True, 'fine_tune_steps': 10},}
            
            logger.info(f"Training MITRA model")
            self.predictor.fit(
                train_data=train_data,
                hyperparameters=hyperparameters,
                time_limit=120,
                presets='medium_quality'
            )
            
            logger.info(f"MITRA model trained successfully")
            
        elif task_type == "regression":
            logger.info(f"Setting up MITRA model for regression")
            logger.info(f"Training samples: {len(train_df)}")
            
            # Create a temporary directory for AutoGluon
            self.temp_dir = tempfile.mkdtemp(prefix="mitra_")
            logger.info(f"Using temporary directory: {self.temp_dir}")
            
            # Combine training data and labels
            train_data = train_df.copy()
            train_data['target'] = train_labels
            
            # Initialize TabularPredictor with MITRA for regression
            self.predictor = TabularPredictor(
                label='target',
                problem_type='regression',
                path=self.temp_dir,
                verbosity=2
            )
            
            # Train with MITRA regressor model
            # Use the MITRA regressor from https://huggingface.co/autogluon/mitra-regressor
            hyperparameters = {'MITRA': {'fine_tune': False},}
            
            logger.info(f"Training MITRA regressor model")
            self.predictor.fit(
                train_data=train_data,
                hyperparameters=hyperparameters,
                time_limit=120,
                presets='medium_quality'
            )
            
            logger.info(f"MITRA regressor model trained successfully")
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def predict_test_cases(self, test_df: pd.DataFrame, task_type: str):
        """
        Predict the target for the given test dataframe using the MITRA model.
        
        Args:
            test_df (pd.DataFrame): The input dataframe containing test cases for prediction.
            task_type (str): Either "classification" or "regression".
            
        Returns:
            np.ndarray: 
                - For binary classification: shape (n_samples,) with probabilities of positive class
                - For multiclass classification: shape (n_samples, n_classes) with class probabilities
                - For regression: shape (n_samples,) with predicted values
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_predictive_ml_model first.")
        
        if task_type == "classification":
            logger.info(f"Predicting {len(test_df)} test cases with MITRA classifier")
            
            # Get prediction probabilities
            predictions = self.predictor.predict_proba(test_df)
            
            logger.info(f"Predictions shape: {predictions.shape}")
            
            # Convert to numpy array and return full probability matrix
            # The benchmark code will handle extracting the positive class for binary classification
            predictions_array = predictions.values
            
            return predictions_array
                
        elif task_type == "regression":
            logger.info(f"Predicting {len(test_df)} test cases with MITRA regressor")
            
            # Get predictions for regression
            predictions = self.predictor.predict(test_df)
            
            logger.info(f"Predictions shape: {predictions.shape}")
            
            # Convert to numpy array
            predictions_array = predictions.values
            
            return predictions_array
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def __del__(self):
        """Cleanup temporary directory when object is destroyed."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {self.temp_dir}: {e}")

# Made with Bob
