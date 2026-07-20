"""
TabDPT Embedding Approach for Tabular Data.

Paper: TabDPT: Scaling Tabular Foundation Models
Repo:  https://github.com/layer6ai-labs/TabDPT-inference

TabDPT is an in-context learner: at inference time the entire training set is
passed as context (K/V source rows) and test rows attend to it via cross-row
attention.  Labels are fused into the Value vectors at every transformer layer.

Supported embeddings
--------------------
Row embeddings  — the final-layer hidden state for each eval row, extracted
                  from just before the output head.  Shape: (n_rows, ninp).
                  For unsupervised tasks (no labels available) a near-constant
                  dummy label is used (all zero except one row, the minimum
                  variation TabDPTClassifier's num_classes > 1 check allows);
                  the model still produces useful contextual row
                  representations from its learned feature interactions.

Table embedding — mean-pool of the row embeddings across all eval rows.
                  Shape: (ninp,).

Cell / column embeddings are NOT available: TabDPT projects all features of a
row into a single ninp-dim vector at the input (Linear(F→ninp)).  There is no
per-feature/per-column dimension anywhere in the network.

Predictive ML   — uses the standard TabDPTClassifier / TabDPTRegressor sklearn
                  API directly; no modification to the model is required.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TabDPT model subclass that also returns pre-head row representations
# ---------------------------------------------------------------------------

def _make_embedding_model(base_model):
    """
    Wrap a loaded TabDPTModel so that forward() returns
    (logits, row_embeddings) instead of just logits.

    row_embeddings shape: (T_eval, B, ninp)  — same device as the model.

    We patch by replacing forward() on the instance so we don't need to
    touch the installed package.
    """
    def forward_with_embeddings(x_src, y_src, num_features):
        # ---- replicate TabDPTModel.forward() up to self.head ----
        x = x_src.transpose(0, 1)   # (B,T,F) → (T,B,F)
        y = y_src.transpose(0, 1)   # (B,T) → (T,B)

        eval_pos = y.shape[0]        # number of context (training) rows
        n_think = base_model.n_thinking_rows

        from tabdpt.utils import clip_outliers, normalize_data
        x = clip_outliers(x, -1 if base_model.training else eval_pos,
                          n_sigma=base_model.clip_sigma)
        x = normalize_data(x, -1 if base_model.training else eval_pos)
        x = clip_outliers(x, -1 if base_model.training else eval_pos,
                          n_sigma=base_model.clip_sigma)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = base_model.encoder(x)
        src = base_model.enc_norm(x)

        if n_think > 0:
            B = src.shape[1]
            src = torch.cat(
                [base_model.thinking_embed.unsqueeze(1).expand(n_think, B, -1), src],
                dim=0,
            )

        for l, layer in enumerate(base_model.transformer_encoder):
            y_emb = base_model.y_encoders[l](y.unsqueeze(-1))
            if n_think > 0:
                B2 = y_emb.shape[1]
                y_emb = torch.cat(
                    [y_emb.new_zeros(n_think, B2, y_emb.shape[-1]), y_emb], dim=0
                )
            residual = layer(src, y_emb, eval_pos + n_think)
            src = src + residual

        # Pre-head representations for the eval rows
        eval_representations = src[eval_pos + n_think:]  # (T_eval, B, ninp)
        logits = base_model.head(eval_representations.float())
        return logits, eval_representations

    base_model.forward_with_embeddings = forward_with_embeddings
    return base_model


# ---------------------------------------------------------------------------
# Main approach class
# ---------------------------------------------------------------------------

class TabDPTEmbedder(BaseTabularEmbeddingApproach):
    """
    TabDPT-based row and table embedder.

    Config parameters (tabdpt.yaml):
        device:           "cuda" | "cpu" | "auto"  (default: auto)
        context_size:     Max context rows passed to the model (default: 1024).
                          Larger values = richer context but more VRAM.
        use_retrieval:    If True, use FAISS retrieval context for embedding
                          extraction (better for large datasets, slower).
                          Default: False (stacked context).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg

        device_cfg = cfg.approach.get("device", "auto")
        if device_cfg == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_cfg

        self.context_size: int = int(cfg.approach.get("context_size", 1024))
        self.use_retrieval: bool = bool(cfg.approach.get("use_retrieval", False))

        self._classifier = None   # TabDPTClassifier — sklearn API for predictive_ml
        self._regressor = None    # TabDPTRegressor  — sklearn API for predictive_ml
        self._embed_model = None  # patched TabDPTModel for embedding extraction

        self._pk_column: Optional[str] = None    # primary-key column to exclude from features, set by RowEmbeddingComponent
        self._cat_encoder = None  # OrdinalEncoder fit on train_df, reused for test_df in predictive_ml

        logger.info(f"TabDPTEmbedder initialised (device={self.device})")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_trained_model(self):
        """Load TabDPT classifier (idempotent). Used by embedding components."""
        if self._embed_model is not None:
            return
        try:
            from tabdpt import TabDPTClassifier
        except ImportError as exc:
            raise ImportError(
                "TabDPT not found. Run "
                "approaches/benchmark_approaches_src/tabdpt/setup.sh first."
            ) from exc

        logger.info("Loading TabDPT model for embedding extraction...")
        clf = TabDPTClassifier(device=self.device)
        # Trigger weight download / cache load by fitting on a tiny dummy table
        # (needs 2+ classes to satisfy TabDPTClassifier's assertion).
        _dummy_X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        _dummy_y = np.array([0, 1, 0])
        clf.fit(_dummy_X, _dummy_y)

        # Patch the underlying model to also return pre-head representations
        self._embed_model = _make_embedding_model(clf.model)
        # Keep the estimator: we reuse its fit()/._prepare_prediction() pipeline
        # which handles imputation, scaling, and PCA for wide tables.
        self._clf_estimator = clf
        logger.info("TabDPT embedding model ready.")

    def _load_predictive_model(self, task_type: str):
        """Load classifier or regressor for predictive_ml (idempotent per type)."""
        try:
            from tabdpt import TabDPTClassifier, TabDPTRegressor
        except ImportError as exc:
            raise ImportError(
                "TabDPT not found. Run "
                "approaches/benchmark_approaches_src/tabdpt/setup.sh first."
            ) from exc

        if task_type == "classification" and self._classifier is None:
            self._classifier = TabDPTClassifier(device=self.device)
            logger.info("TabDPT classifier loaded.")
        elif task_type == "regression" and self._regressor is None:
            self._regressor = TabDPTRegressor(device=self.device)
            logger.info("TabDPT regressor loaded.")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocessing(self, input_table: pd.DataFrame, fit_encoder: bool = True) -> np.ndarray:
        """
        Convert DataFrame to float64 numpy array (TabDPT requires raw floats).

        Drops the primary-key column, if one was set by the caller (see
        RowEmbeddingComponent.setup_model_for_task) — TabDPT was trained and
        evaluated on OpenML tables, which never expose a raw row identifier
        as a feature.

        Categorical columns are ordinal-encoded. Pass fit_encoder=False to
        reuse the encoder fit on a previous call (e.g. on train_df) instead
        of fitting a new one, so train/test share one category->code mapping.
        """
        df = input_table.copy()
        if self._pk_column and self._pk_column in df.columns:
            df = df.drop(columns=[self._pk_column])

        cat_cols = [c for c in df.columns if df[c].dtype == "object" or hasattr(df[c].dtype, "categories")]
        if cat_cols:
            df[cat_cols] = df[cat_cols].astype(str).fillna("__nan__")
            if fit_encoder or self._cat_encoder is None:
                self._cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                self._cat_encoder.fit(df[cat_cols])
            df[cat_cols] = self._cat_encoder.transform(df[cat_cols])

        df = df.fillna(0).astype("float64")
        return df.values

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def get_row_embeddings(
        self,
        input_table: pd.DataFrame,
        train_size: Optional[int] = None,
        train_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract row embeddings for all rows in input_table.

        The model needs a context: rows with known labels that are passed as
        K/V source.  When train_labels is provided the first train_size rows
        are used as context and the remaining rows are the eval queries.
        Otherwise all rows serve as both context and eval with a near-constant
        dummy label (all zero except one row) — the model still produces
        meaningful structural representations.

        Returns:
            np.ndarray of shape (n_eval_rows, ninp).
        """
        self.load_trained_model()
        X = self.preprocessing(input_table)
        n = len(X)

        if train_labels is not None and train_size is not None:
            # Supervised inference: fit on training split, embed test split
            y_ctx = self._prepare_labels(train_labels[:train_size])
            X_ctx = X[:train_size]
            X_eval = X[train_size:]
        else:
            # Unsupervised: use all rows as context with dummy labels.
            # TabDPTClassifier requires num_classes > 1, so a fully-constant
            # label (used by e.g. tabpfn/tabicl/sap_rpt_oss for the same
            # purpose) isn't possible here; flip only the first context row
            # to minimize how much of the context carries a label that has
            # no relation to its actual content.
            ctx_size = min(n, self.context_size)
            X_ctx = X[:ctx_size]
            y_ctx = np.zeros(ctx_size, dtype=np.float32)
            y_ctx[0] = 1.0
            X_eval = X

        return self._extract_embeddings(X_ctx, y_ctx, X_eval)

    def get_table_embedding(self, input_table: pd.DataFrame) -> np.ndarray:
        """
        Returns a single (ninp,) table-level embedding as the mean of all
        eval-row representations.
        """
        row_embs = self.get_row_embeddings(input_table)
        return row_embs.mean(axis=0)

    def _prepare_labels(self, labels) -> np.ndarray:
        """Normalise labels to a float32 numpy array."""
        if hasattr(labels, "to_numpy"):
            arr = labels.to_numpy()
        elif hasattr(labels, "values"):
            v = labels.values
            arr = v.codes if hasattr(v, "codes") else np.asarray(v)
        else:
            arr = np.asarray(labels)

        if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            try:
                arr = arr.astype(np.float64)
            except (ValueError, TypeError):
                arr = LabelEncoder().fit_transform(arr).astype(np.float64)
        return arr.astype(np.float32)

    def _extract_embeddings(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_eval: np.ndarray,
    ) -> np.ndarray:
        """
        Core embedding extraction.  Uses TabDPT's own estimator pipeline
        (imputation, scaling, PCA for wide tables) via fit() + _prepare_prediction(),
        then calls the patched forward to extract pre-head row representations.

        Returns:
            np.ndarray of shape (len(X_eval), ninp).
        """
        from tabdpt.utils import pad_x

        est = self._clf_estimator
        model = self._embed_model
        device = next(model.parameters()).device

        # Fit the estimator on context rows — this populates imputer, scaler,
        # and PCA projection matrix (self.V) for wide tables, exactly as TabDPT does.
        est.fit(X_ctx, y_ctx)

        # _prepare_prediction applies imputation, scaling, and optional PCA/subsampling,
        # returning (train_x, train_y, test_x) as GPU tensors (NOT yet padded to max_features).
        # pad_x must be applied to match the encoder's expected input width, exactly as
        # TabDPTClassifier._prepare_stacked_context() does.
        train_x, train_y, _ = est._prepare_prediction(X_ctx)
        num_features = torch.tensor([train_x.shape[1]], device=device)

        # Context tensors: (1, T_ctx, max_features) — batch dim + padded features
        # y_ctx_t contains only context labels; eval_pos = T_ctx is derived from y_src.shape[0]
        X_ctx_t = pad_x(train_x.unsqueeze(0), est.max_features)
        y_ctx_t = train_y.unsqueeze(0)  # (1, T_ctx)

        all_embeddings = []
        batch_size = max(1, self.context_size)

        for start in range(0, len(X_eval), batch_size):
            chunk = X_eval[start: start + batch_size]
            # Reuse _prepare_prediction for the eval chunk to apply identical transforms
            _, _, test_x = est._prepare_prediction(chunk)
            X_eval_t = pad_x(test_x.unsqueeze(0), est.max_features)  # (1, chunk_size, max_features)

            x_src = torch.cat([X_ctx_t, X_eval_t], dim=1)

            with torch.no_grad():
                _, row_embeds = model.forward_with_embeddings(
                    x_src, y_ctx_t, num_features
                )
            # row_embeds: (T_eval_chunk, 1, ninp) → squeeze batch dim
            all_embeddings.append(row_embeds.squeeze(1).cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)

    # ------------------------------------------------------------------
    # Predictive ML (standard sklearn API — no model modification needed)
    # ------------------------------------------------------------------

    def load_predictive_ml_model(
        self,
        train_df: pd.DataFrame,
        train_labels: pd.Series,
        task_type: str,
        dataset_information: dict,
    ):
        self._load_predictive_model(task_type)
        X_train = self.preprocessing(train_df, fit_encoder=True)
        y_train = self._prepare_labels(train_labels)

        if task_type == "classification":
            self._classifier.fit(X_train, y_train)
            logger.info("TabDPT classifier fitted.")
        elif task_type == "regression":
            self._regressor.fit(X_train, y_train)
            logger.info("TabDPT regressor fitted.")
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def predict_test_cases(self, test_df: pd.DataFrame, task_type: str) -> np.ndarray:
        X_test = self.preprocessing(test_df, fit_encoder=False)
        if task_type == "classification":
            return self._classifier.predict_proba(X_test)
        elif task_type == "regression":
            return self._regressor.predict(X_test)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
