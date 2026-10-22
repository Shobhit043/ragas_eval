from models.model import EmbeddModel
from ragas.embeddings.base import BaseRagasEmbedding


class SharedSentenceTransformerEmbeddings(BaseRagasEmbedding):
    """Adapter so Ragas uses the same SentenceTransformer as ChromaDB."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def embed_text(self, text: str, **kwargs) -> list[float]:
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return vector.tolist()

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text, **kwargs)


embd = EmbeddModel()
model = embd.load_model()
embeddings = SharedSentenceTransformerEmbeddings(model)
