from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
import logging
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import torch

from tabicl import TabICLClassifier, TabICLRegressor

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
        self.train_size = getattr(self.cfg.approach, "train_size", None)
        self._use_train_size_for_embeddings = getattr(self.cfg.approach, "use_train_size_for_embeddings", False)
        self.task_type = None  # Will be set by setup_model_for_task
        logger.info("TabICLEmbedder: Initialized.")

    def _is_regression_task(self):
        """
        Determine if the task is regression based on stored task_type.
        Returns True for regression, False for classification.
        For non-predictive tasks (e.g., row similarity), defaults to classification.
        """
        if self.task_type is None:
            # For non-predictive tasks (row similarity, column similarity, etc.),
            # default to classification mode (use classifier for embeddings)
            logger.debug("task_type not set (non-predictive task), defaulting to classification mode")
            return False
        return self.task_type == "regression"
    
    def _load_model_for_embeddings(self, is_regression):
        """
        Load the appropriate TabICL model (classifier or regressor) for generating embeddings.
        """
        # Check if we need to reload the model (e.g., switching between classifier and regressor)
        needs_reload = False
        if self.model is not None:
            current_is_regressor = isinstance(self.model, TabICLRegressor)
            if is_regression != current_is_regressor:
                logger.info(f"Switching from {'regressor' if current_is_regressor else 'classifier'} to {'regressor' if is_regression else 'classifier'}")
                needs_reload = True
        
        if self.model is None or needs_reload:
            logger.info("Loading TabICL model for embeddings...")
            n_estimators = getattr(self.cfg.approach, "n_estimators", 32)
            use_memory_efficient = getattr(self.cfg.approach, "use_memory_efficient_model", True)
            device = getattr(self.cfg.approach, "device", "cpu")
            checkpoint_version = getattr(self.cfg.approach, "checkpoint_version", "tabicl-classifier-v2-20260212.ckpt")

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_kwargs = {
                "n_estimators": n_estimators,
                "device": device,
            }
            
            # Choose appropriate checkpoint and model class based on task type
            if is_regression:
                # Check if checkpoint is v2
                is_v2 = "v2" in checkpoint_version.lower()
                if not is_v2:
                    raise NotImplementedError("TabICL regression is only supported with v2 checkpoints. Please use a v2 checkpoint.")
                
                model_kwargs["checkpoint_version"] = "tabicl-regressor-v2-20260212.ckpt"
                self.model = TabICLRegressor(**model_kwargs)
                logger.info(f"TabICL Regressor loaded with checkpoint_version={model_kwargs['checkpoint_version']}, n_estimators={n_estimators} on {device}.")
            else:
                model_kwargs["checkpoint_version"] = checkpoint_version
                self.model = TabICLClassifier(**model_kwargs)
                logger.info(f"TabICL Classifier loaded with checkpoint_version={checkpoint_version}, n_estimators={n_estimators} on {device}.")
            
            if use_memory_efficient:
                logger.info("Using memory efficient mode with batch_size=4")

    def load_trained_model(self):
        """Legacy method for backward compatibility - loads classifier by default."""
        if self.model is None:
            self._load_model_for_embeddings(is_regression=False)

    def preprocessing(self, input_table: pd.DataFrame):
        return input_table

    def get_row_embeddings(self, input_table: pd.DataFrame, train_size: int = None, train_labels: np.ndarray = None):
        
        input_table_clean = self._preprocess_for_tabicl(input_table)
        effective_train_size = train_size if train_size is not None else (self.train_size if self._use_train_size_for_embeddings else None)
        
        # Prepare labels
        if train_labels is not None:
            train_labels_array = self._convert_labels_to_numeric(train_labels)
        else:
            train_labels_array = np.zeros(len(input_table_clean))
        
        # Detect task type from stored task_type
        is_regression = self._is_regression_task()
        
        # Load appropriate model based on task type
        self._load_model_for_embeddings(is_regression)
        
        # Fit model and get embeddings
        if train_labels is not None and effective_train_size is not None:
            self.model.fit(input_table_clean[:effective_train_size], train_labels_array[:effective_train_size])
        else:
            self.model.fit(input_table_clean, train_labels_array)
        
        # Get row embeddings
        row_embeddings = self._extract_row_embeddings(input_table_clean, train_labels_array)
        
        # Extract test portion if needed
        if train_labels is not None and effective_train_size is not None:
            return row_embeddings[effective_train_size:].astype(np.float32)
        return row_embeddings.astype(np.float32)
    
    def _convert_labels_to_numeric(self, train_labels):
        """Convert labels to numeric array, handling various input types."""
        if hasattr(train_labels, 'to_numpy'):
            train_labels_array = train_labels.to_numpy()
        elif hasattr(train_labels, 'values'):
            values = train_labels.values
            train_labels_array = values.codes if hasattr(values, 'codes') else np.asarray(values)
        else:
            train_labels_array = np.asarray(train_labels)
        
        # Ensure numeric dtype
        if train_labels_array.dtype == np.object_ or not np.issubdtype(train_labels_array.dtype, np.number):
            try:
                train_labels_array = train_labels_array.astype(np.float64)
            except (ValueError, TypeError):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                train_labels_array = le.fit_transform(train_labels_array).astype(np.float64)
        
        return train_labels_array
    
    def _extract_row_embeddings(self, input_table_clean, train_labels_array):
        """Extract row embeddings from TabICL model."""
        X_tensor = torch.from_numpy(input_table_clean.values).float().unsqueeze(0).to(self.model.device_)
        y_tensor = torch.from_numpy(train_labels_array).float().unsqueeze(0).to(self.model.device_)
        
        with torch.no_grad():
            col_embeddings = self.model.model_.col_embedder(
                X_tensor,
                y_train=y_tensor,
                embed_with_test=False,
                feature_shuffles=None,
                mgr_config=self.model.inference_config_.COL_CONFIG,
            )
            # Replace NaN with zeros (NaN in CLS tokens causes row_interactor to fail)
            col_embeddings = torch.nan_to_num(col_embeddings, nan=0.0)
            
            row_embeddings = self.model.model_.row_interactor(
                col_embeddings,
                mgr_config=self.model.inference_config_.ROW_CONFIG,
            )
        
        return row_embeddings.cpu().numpy().squeeze(0)

    def load_predictive_ml_model(self, train_df: pd.DataFrame, train_labels: pd.Series, task_type: str, dataset_information: dict):
        """
        Set up the TabICL model for predictive ML tasks.
        Args:
            train_df (pd.DataFrame): Training data.
            train_labels (pd.Series): Training labels.
            task_type (str): Either "classification" or "regression".
            dataset_information (dict): Additional dataset info.
        """
        n_estimators = getattr(self.cfg.approach, "n_estimators", 32)
        checkpoint_version = getattr(self.cfg.approach, "checkpoint_version", "tabicl-classifier-v2-20260212.ckpt")
        device = getattr(self.cfg.approach, "device", "cpu")
        
        # Resolve "auto" device to actual device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if checkpoint is v2
        is_v2 = "v2" in checkpoint_version.lower()
        
        if task_type == "classification":
            self.model = TabICLClassifier(
                n_estimators=n_estimators,
                device=device,
                checkpoint_version=checkpoint_version,
            )
            
            train_df_processed = self._preprocess_for_tabicl(train_df)
            print(f"Done preprocessing the table, now starting to fit the model")
            self.model.fit(train_df_processed, train_labels)
            print(f"Finished fitting the TabICL model for classification.")
            
        elif task_type == "regression":
            if not is_v2:
                raise NotImplementedError("TabICL regression is only supported with v2 checkpoints. Please use a v2 checkpoint.")
            
            # Use regressor-specific checkpoint for v2
            regressor_checkpoint = "tabicl-regressor-v2-20260212.ckpt"
            
            self.model = TabICLRegressor(
                n_estimators=n_estimators,
                device=device,
                checkpoint_version=regressor_checkpoint,
            )
            
            train_df_processed = self._preprocess_for_tabicl(train_df)
            print(f"Done preprocessing the table, now starting to fit the model")
            self.model.fit(train_df_processed, train_labels)
            print(f"Finished fitting the TabICL model for regression.")
            
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
            print(f"Done preprocessing the table, now starting to predict the {len(test_df_processed)} test cases")
            proba_tuple = self.model.predict_proba(test_df_processed)
            print(f"Finished predicting with the TabICL model.")
            
            if isinstance(proba_tuple, tuple):
                logits = proba_tuple[0]
                return logits
            else:
                print(f"Is not tuple")
                return proba_tuple
        
        elif task_type == "regression":
            # Check if model is loaded and is the correct type
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_predictive_ml_model first.")
            
            # Verify it's a TabICLRegressor (which requires v2)
            if not isinstance(self.model, TabICLRegressor):
                raise NotImplementedError("TabICL regression is only supported with v2 checkpoints. Please use a v2 checkpoint.")
            
            test_df_processed = self._preprocess_for_tabicl(test_df)
            print(f"Done preprocessing the table, now starting to predict the {len(test_df_processed)} test cases")
            predictions = self.model.predict(test_df_processed)
            print(f"Finished predicting with the TabICL model for regression.")
            return predictions
            
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
        
        # Ensure all columns are numeric types (convert any remaining object types)
        for col in input_table_clean.columns:
            if input_table_clean[col].dtype == 'object':
                # Try to convert to numeric, coerce errors to NaN, then fill with 0
                input_table_clean[col] = pd.to_numeric(input_table_clean[col], errors='coerce').fillna(0)
        
        # Final safety check: convert all to float64
        input_table_clean = input_table_clean.astype('float64')
        
        return input_table_clean

    def _get_col_embeddings_without_cls(self, input_table: pd.DataFrame):
        """
        Helper function to extract raw column embeddings without CLS tokens.
        
        Args:
            input_table (pd.DataFrame): Input table
            
        Returns:
            tuple: (col_embeddings_without_cls, column_names, input_table_clean)
                - col_embeddings_without_cls: shape (B, T, H-cls, D) where B=batch, T=rows, H=columns, D=embedding_dim
                - column_names: list of column names
                - input_table_clean: preprocessed input table
        """
        self.load_trained_model()
        
        print("input_table shape:", input_table.shape)
        
        input_table_clean = self._preprocess_for_tabicl(input_table)
        
        print(f"input_table_clean shape after base preprocessing: {input_table_clean.shape}")
        print(f"Original columns: {len(input_table.columns)}, Clean table columns: {len(input_table_clean.columns)}")
        
        # Always use all data as training for embeddings
        y = np.zeros(len(input_table_clean))
        logger.info("Fitting model for embeddings")
        self.model.fit(input_table_clean, y)
        
        # Convert to torch tensors and get column embeddings directly
        X_tensor = torch.from_numpy(input_table_clean.values).float().unsqueeze(0).to(self.model.device_)
        y_tensor = torch.from_numpy(y).float().unsqueeze(0).to(self.model.device_)
        
        with torch.no_grad():
            # Get column embeddings directly from col_embedder
            col_embeddings = self.model.model_.col_embedder(
                X_tensor,
                y_train=y_tensor,
                embed_with_test=False,
                feature_shuffles=None,
                mgr_config=self.model.inference_config_.COL_CONFIG,
            )
        
        # Extract column embeddings: col_embeddings has shape (B, T, H, D)
        # where H includes CLS tokens (first row_num_cls) + actual features
        # Exclude the CLS tokens (first row_num_cls tokens) from the column dimension
        row_num_cls = self.model.model_.row_num_cls
        col_embeddings_without_cls = col_embeddings[:, :, row_num_cls:, :]
        
        # Get the feature names that TabICL actually used after its internal preprocessing
        if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
            column_names = self.model.feature_names_in_
            print(f"Columns kept after TabICL's internal preprocessing: {list(column_names)}")
        else:
            # Fallback to input_table_clean columns if feature_names_in_ is not available
            column_names = input_table_clean.columns
            print(f"Using input_table_clean columns (feature_names_in_ not available): {list(column_names)}")
        
        return col_embeddings_without_cls, column_names, input_table_clean

    def get_column_embeddings(self, input_table: pd.DataFrame) -> tuple:
        """
        Generate column embeddings using TabICL's ColEmbedding stage.
        
        Args:
            input_table (pd.DataFrame): Input table with columns to embed
            
        Returns:
            tuple: (column_embeddings, column_names) where column_embeddings has shape (num_columns, embedding_dim)
        """
        # Get raw column embeddings without CLS tokens
        col_embeddings_without_cls, column_names, _ = self._get_col_embeddings_without_cls(input_table)
        
        # Aggregate across rows (T dimension) to get per-column embeddings
        # Take mean across the row dimension to get (B, H-cls, D), then squeeze batch dimension
        column_embeddings = col_embeddings_without_cls.mean(dim=1).squeeze(0).cpu().numpy()
        
        print(f"column_embeddings shape: {column_embeddings.shape}")
        print(f"Number of column names: {len(column_names)}")
        
        return column_embeddings, column_names

    def get_cell_embeddings(self, input_table: pd.DataFrame) -> tuple:
        """
        Generate cell embeddings using TabICL's ColEmbedding stage.
        
        Args:
            input_table (pd.DataFrame): Input table with cells to embed
            
        Returns:
            tuple: (cell_embeddings, column_names) where cell_embeddings has shape (num_rows, num_columns, embedding_dim)
        """
        # Get raw column embeddings without CLS tokens
        col_embeddings_without_cls, tabicl_column_names, input_table_clean = self._get_col_embeddings_without_cls(input_table)
        
        # col_embeddings_without_cls has shape (B, T, H-cls, D)
        # Squeeze batch dimension to get (T, H-cls, D) = (num_rows, num_columns, embedding_dim)
        cell_embeddings_raw = col_embeddings_without_cls.squeeze(0).cpu().numpy()
        
        # Check if column names match
        input_table_clean_columns = list(input_table_clean.columns)
        tabicl_column_names_list = list(tabicl_column_names)
        
        if input_table_clean_columns == tabicl_column_names_list:
            # Column names match, return as is
            cell_embeddings = cell_embeddings_raw
            column_names = tabicl_column_names
        else:
            # Column names don't match - need to reorder/fill missing columns
            logger.warning(f"Column mismatch: input_table_clean has {len(input_table_clean_columns)} columns, "
                         f"TabICL returned {len(tabicl_column_names_list)} columns")
            
            num_rows = cell_embeddings_raw.shape[0]
            embedding_dim = cell_embeddings_raw.shape[2]
            num_cols_needed = len(input_table_clean_columns)
            
            # Create mapping from tabicl column names to indices
            tabicl_col_to_idx = {col: idx for idx, col in enumerate(tabicl_column_names_list)}
            
            # Initialize output array
            cell_embeddings = np.zeros((num_rows, num_cols_needed, embedding_dim), dtype=np.float32)
            
            # Fill in embeddings for each column in input_table_clean
            for col_idx, col_name in enumerate(input_table_clean_columns):
                if col_name in tabicl_col_to_idx:
                    # Column exists in TabICL output, use its embeddings
                    tabicl_idx = tabicl_col_to_idx[col_name]
                    cell_embeddings[:, col_idx, :] = cell_embeddings_raw[:, tabicl_idx, :]
                else:
                    # Column missing from TabICL output, use mean of row embeddings as fallback
                    logger.warning(f"Column '{col_name}' not found in TabICL output, using row mean as fallback")
                    for row_idx in range(num_rows):
                        # Mean across all columns for this row
                        row_mean = cell_embeddings_raw[row_idx, :, :].mean(axis=0)
                        cell_embeddings[row_idx, col_idx, :] = row_mean
            
            column_names = input_table_clean_columns
        
        # Add header row embedding (mean across all rows for each column)
        # cell_embeddings currently has shape (num_rows, num_columns, embedding_dim)
        # We need to add a header row at index 0
        header_embeddings = cell_embeddings.mean(axis=0, keepdims=True)  # Shape: (1, num_columns, embedding_dim)
        
        # Concatenate header row with data rows
        cell_embeddings_with_header = np.concatenate([header_embeddings, cell_embeddings], axis=0)
        # Final shape: (num_rows + 1, num_columns, embedding_dim)
        
        print(f"cell_embeddings_with_header shape: {cell_embeddings_with_header.shape}")
        print(f"Number of rows (including header): {cell_embeddings_with_header.shape[0]}, Number of columns: {cell_embeddings_with_header.shape[1]}")
        print(f"Number of column names: {len(column_names)}")
        
        return cell_embeddings_with_header, column_names

