import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface

logger = logging.getLogger(__name__)


def _to_text(table: pd.DataFrame, max_rows: int = -1) -> str:
    headers = [str(h) for h in table.columns]
    parts = headers[:]
    df = table if max_rows == -1 else table.head(max_rows)
    for _, row in df.iterrows():
        parts.extend(str(v) for v in row.tolist())
    return " ".join(parts)


class TableEmbeddingComponent(TableEmbeddingInterface):
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance
        self.table_row_limit = approach_instance.table_row_limit
        super().__init__()

    def setup_model_for_task(self):
        self.vectorizer = HashingVectorizer(
            n_features=self.approach_instance.n_features,
            norm="l2",
            alternate_sign=False,
            stop_words="english",
        )

    def create_table_embedding(self, input_table: pd.DataFrame) -> np.ndarray:
        return self.vectorizer.transform([_to_text(input_table, self.table_row_limit)]).toarray().ravel()

    def create_query_embedding(self, query: str) -> np.ndarray:
        return self.vectorizer.transform([query]).toarray().ravel()
