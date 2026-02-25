from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
import logging
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import torch

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
        self.train_size = getattr(self.cfg.approach, "train_size", None)
        self._use_train_size_for_embeddings = getattr(self.cfg.approach, "use_train_size_for_embeddings", False)
        logger.info("TabICLEmbedder: Initialized.")

    def load_trained_model(self):
        if self.model is None:
            logger.info("Loading TabICL model...")
            n_estimators = getattr(self.cfg.approach, "n_estimators", 32)
            use_memory_efficient = getattr(self.cfg.approach, "use_memory_efficient_model", True)
            device = getattr(self.cfg.approach, "device", "cpu")
            checkpoint_version = getattr(self.cfg.approach, "checkpoint_version", "tabicl-classifier-v2-20260212.ckpt")

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_kwargs = {
                "n_estimators": n_estimators,
                "device": device,
                "checkpoint_version": checkpoint_version,
            }
            
            if use_memory_efficient:
                model_kwargs.update({
                    "batch_size": 4,
                })
                logger.info(f"TabICL model loaded with checkpoint_version={checkpoint_version}, n_estimators={n_estimators}, batch_size=4 on CPU with memory optimizations.")
            else:
                logger.info(f"TabICL model loaded with checkpoint_version={checkpoint_version}, n_estimators={n_estimators} on CPU.")
            
            self.model = TabICLClassifier(**model_kwargs)

    def preprocessing(self, input_table: pd.DataFrame):
        return input_table

    def get_row_embeddings(self, input_table: pd.DataFrame, train_size: int = None, train_labels: np.ndarray = None):
        
        self.load_trained_model()
        print("input_table shape:", input_table.shape)
        
        input_table_clean = self._preprocess_for_tabicl(input_table)

        # Use train_size parameter if provided, otherwise check config
        effective_train_size = train_size if train_size is not None else (self.train_size if self._use_train_size_for_embeddings else None)
        
        # If train_labels are provided, use them for fitting
        if train_labels is not None:
            if effective_train_size is not None:
                # Fit on training portion with actual labels
                self.model.fit(input_table_clean[:effective_train_size], train_labels)
                
                # Convert to torch tensors and get row embeddings directly
                X_tensor = torch.from_numpy(input_table_clean.values).float().unsqueeze(0).to(self.model.device_)
                y_tensor = torch.from_numpy(train_labels).float().unsqueeze(0).to(self.model.device_)
                
                with torch.no_grad():
                    # Get column embeddings first
                    col_embeddings = self.model.model_.col_embedder(
                        X_tensor,
                        y_train=y_tensor,
                        embed_with_test=False,
                        feature_shuffles=None,
                        mgr_config=self.model.inference_config_.COL_CONFIG,
                    )
                    # Then get row embeddings
                    row_embeddings = self.model.model_.row_interactor(
                        col_embeddings,
                        mgr_config=self.model.inference_config_.ROW_CONFIG,
                    )
                
                # Convert back to numpy and extract test embeddings
                row_embeddings = row_embeddings.cpu().numpy().squeeze(0)
                # Extract embeddings for rows after train_size
                test_embeddings = row_embeddings[effective_train_size:]
            else:
                # Fit on all data with provided labels
                self.model.fit(input_table_clean, train_labels)
                
                # Convert to torch tensors and get row embeddings directly
                X_tensor = torch.from_numpy(input_table_clean.values).float().unsqueeze(0).to(self.model.device_)
                y_tensor = torch.from_numpy(train_labels).float().unsqueeze(0).to(self.model.device_)
                
                with torch.no_grad():
                    # Get column embeddings first
                    col_embeddings = self.model.model_.col_embedder(
                        X_tensor,
                        y_train=y_tensor,
                        embed_with_test=False,
                        feature_shuffles=None,
                        mgr_config=self.model.inference_config_.COL_CONFIG,
                    )
                    # Then get row embeddings
                    row_embeddings = self.model.model_.row_interactor(
                        col_embeddings,
                        mgr_config=self.model.inference_config_.ROW_CONFIG,
                    )
                
                # Convert back to numpy
                row_embeddings = row_embeddings.cpu().numpy().squeeze(0)
                # When no effective_train_size, return all embeddings
                test_embeddings = row_embeddings
        else:
            # Use dummy labels when train_labels not provided
            # Old behavior: use all data as training with dummy labels
            y = np.zeros(len(input_table_clean))
            self.model.fit(input_table_clean, y)
            
            # Convert to torch tensors and get row embeddings directly
            X_tensor = torch.from_numpy(input_table_clean.values).float().unsqueeze(0).to(self.model.device_)
            y_tensor = torch.from_numpy(y).float().unsqueeze(0).to(self.model.device_)
            
            with torch.no_grad():
                # Get column embeddings first
                col_embeddings = self.model.model_.col_embedder(
                    X_tensor,
                    y_train=y_tensor,
                    embed_with_test=False,
                    feature_shuffles=None,
                    mgr_config=self.model.inference_config_.COL_CONFIG,
                )
                # Then get row embeddings
                row_embeddings = self.model.model_.row_interactor(
                    col_embeddings,
                    mgr_config=self.model.inference_config_.ROW_CONFIG,
                )
            
            # Convert back to numpy
            row_embeddings = row_embeddings.cpu().numpy().squeeze(0)
            # When no train_labels provided, return all embeddings
            test_embeddings = row_embeddings
        
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
            print(f"Done preprocessing the table, now starting to fit the model")
            self.model.fit(train_df_processed, train_labels)
            print(f"Finished fitting the TabICL model for classification.")
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
        
        print(f"input_table_clean shape after base preprocessing: {input_table_clean.shape}")
        print(f"Original columns: {len(input_table.columns)}, Clean table columns: {len(input_table_clean.columns)}")
        
        # Always use all data as training for column embeddings
        y = np.zeros(len(input_table_clean))
        logger.info("Fitting model for column embeddings")
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
        
        # Aggregate across rows (T dimension) to get per-column embeddings
        # Take mean across the row dimension to get (B, H-cls, D), then squeeze batch dimension
        column_embeddings = col_embeddings_without_cls.mean(dim=1).squeeze(0).cpu().numpy()
        
        print(f"column_embeddings shape: {column_embeddings.shape}")
        
        # Get the feature names that TabICL actually used after its internal preprocessing
        # The model stores feature_names_in_ after fitting
        if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
            column_names = self.model.feature_names_in_
            print(f"Columns kept after TabICL's internal preprocessing: {list(column_names)}")
        else:
            # Fallback to input_table_clean columns if feature_names_in_ is not available
            column_names = input_table_clean.columns
            print(f"Using input_table_clean columns (feature_names_in_ not available): {list(column_names)}")
        
        print(f"Number of column names: {len(column_names)}")
        
        return column_embeddings, column_names

