import logging

import pandas as pd
from omegaconf import DictConfig

from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach

logger = logging.getLogger(__name__)


class TFIDFTableEmbedder(BaseTabularEmbeddingApproach):
    def __init__(self, cfg: DictConfig):
        self.n_features = cfg.approach.get("n_features", 32768)
        self.table_row_limit = cfg.approach.get("table_row_limit", -1)
        super().__init__(cfg)

    def preprocessing(self, input_table: pd.DataFrame, component=None):
        pass

    def load_trained_model(self):
        pass
