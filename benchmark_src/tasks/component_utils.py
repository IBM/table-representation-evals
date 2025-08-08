import logging
import numpy as np
import pandas as pd

from benchmark_src.utils.resource_monitoring import monitor_resources

logger = logging.getLogger(__name__)

@monitor_resources(sample_interval=1)
def run_model_setup(component, **kwargs):
    component.setup_model_for_task(**kwargs)

def assert_row_embedding_format(row_embeddings: np.ndarray, input_table: pd.DataFrame):
    assert isinstance(row_embeddings, np.ndarray), "row_embeddings must be a NumPy array"
    assert row_embeddings.ndim == 2, "row_embeddings must be a 2-dimensional array (matrix)"
    expected_num_rows = len(input_table)
    assert row_embeddings.shape[0] == expected_num_rows, \
        f"row_embeddings must have {expected_num_rows} rows, but has {row_embeddings.shape[0]}"
    logger.debug(f"row_embeddings format is correct (number of rows checked, embedding dimension is flexible, in this case it's {row_embeddings.shape[1]}).")

