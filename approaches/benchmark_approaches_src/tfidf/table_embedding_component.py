import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface

logger = logging.getLogger(__name__)


def _to_text(table: pd.DataFrame) -> str:
    headers = [str(h) for h in table.columns]
    parts = headers[:]
    for _, row in table.iterrows():
        parts.extend(str(v) for v in row.tolist())
    return " ".join(parts)


class TableEmbeddingComponent(TableEmbeddingInterface):
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance
        super().__init__()

    def setup_model_for_task(self):
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            stop_words="english",
            max_features=self.approach_instance.n_features,
        )
        self.fitted = False

    def fit_corpus(self, tables) -> None:
        texts = [_to_text(t) for t in tables]
        self.vectorizer.fit(texts)
        self.fitted = True

    def create_table_embedding(self, input_table: pd.DataFrame) -> np.ndarray:
        return self.vectorizer.transform([_to_text(input_table)]).toarray().ravel()

    def create_query_embedding(self, query: str) -> np.ndarray:
        return self.vectorizer.transform([query]).toarray().ravel()
