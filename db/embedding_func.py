from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)
import numpy as np


class EmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model):
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self.model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return [
            np.asarray(vector, dtype=np.float32)
            for vector in vectors
        ]