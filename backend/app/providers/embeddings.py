from pathlib import Path

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from tokenizers import Tokenizer


class LocalEmbeddings:
    model_name = "all-MiniLM-L6-v2"
    dimensions = 384

    def __init__(self, cache_directory: Path):
        self.model = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
        self.model.DOWNLOAD_PATH = cache_directory / self.model_name
        self._tokenizer: Tokenizer | None = None

    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            # The first inference downloads and initializes the model and its matching tokenizer.
            self.embed(["Prepare the local embedding model."])
            self._tokenizer = Tokenizer.from_str(self.model.tokenizer.to_str())
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
        return self._tokenizer

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model(texts)]
