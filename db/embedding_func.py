import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class EmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model):
        self.model = model

    def name(self) -> str:
        return "shared-sentence-transformer"

    def get_config(self) -> dict:
        return {}

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self.model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return [np.asarray(vector, dtype=np.float32) for vector in vectors]
