from typing import Any, List

from benchmark_src.approach_interfaces.table_embedding_interface import TableEmbeddingInterface
from benchmark_approaches_src.sentence_transformer import approach_utils


class TableEmbeddingComponent(TableEmbeddingInterface):

    def __init__(self, approach_instance):
        self.approach_instance = approach_instance
        super().__init__()

    def setup_model_for_task(self):
        self.approach_instance.load_trained_model()

    def create_table_embedding(self, input_table: List[List[Any]]):
        """
        Serialize the full DataFrame to Markdown (header + rows) and embed it
        using the SentenceTransformer model provided by the approach.
        """
        markdown_str = approach_utils.convert_array_to_markdown(input_table)
        embedding = self.approach_instance.model.encode(
            markdown_str,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding
