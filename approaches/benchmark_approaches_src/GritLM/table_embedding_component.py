import logging
from typing import Any, List

import torch

from benchmark_approaches_src.GritLM import approach_utils
from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface

logger = logging.getLogger(__name__)


class TableEmbeddingComponent(TableEmbeddingInterface):
    def __init__(self, approach_instance):
        self.approach_instance = approach_instance
        self.table_row_limit = approach_instance.table_row_limit
        super().__init__()

    def gritlm_instruction(self, instruction: str):
        """
        Format instruction for GritLM embed mode.
        """
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

    def setup_model_for_task(self):
        self.approach_instance.load_trained_model()

    def create_table_embedding(self, input_table: List[List[Any]]):
        """
        Serialize the table to Markdown (headers + rows) and embed with GritLM.
        """
        markdown_str = approach_utils.convert_array_to_markdown(
            input_table, max_rows=self.table_row_limit
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings = self.approach_instance.model.encode(
                [markdown_str],
                instruction=self.gritlm_instruction(""),
                show_progress_bar=False,
            )
        # model.encode returns a batch; unwrap single item for consistency
        return embeddings[0]

    def create_query_embedding(self, query: str):
        """
        Embed a natural language query with GritLM.
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings = self.approach_instance.model.encode(
                [query],
                instruction=self.gritlm_instruction(""),
                show_progress_bar=False,
            )
        return embeddings[0]

