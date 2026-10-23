from pathlib import Path

import chromadb

from db.embedding_func import EmbeddingFunction
from document_loader import DocumentChunk


class DB:
    def __init__(self, db_path="chroma_db", collection_name="data", model=None, client=None):
        if model is None:
            raise ValueError("model must be a SentenceTransformer instance")

        if client is None:
            if db_path == "chroma_db":
                db_path = Path(__file__).resolve().parent / db_path

            Path(db_path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(db_path))

        self.client = client
        self.collection_name = collection_name
        self.embedding_function = EmbeddingFunction(model)
        self.collection = self._create_collection()

    @classmethod
    def in_memory(cls, model, collection_name="data"):
        """Create a database that lives only in memory for the current session."""
        return cls(
            collection_name=collection_name,
            model=model,
            client=chromadb.EphemeralClient(),
        )

    def _create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    @property
    def num_docs(self):
        return self.collection.count()

    def add_documents(self, chunks: list[DocumentChunk], batch_size: int = 500):
        """Insert or update chunks using their stable IDs."""
        if not chunks:
            return
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[chunk["id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                metadatas=[chunk["metadata"] for chunk in batch],
            )

    def replace_documents(self, chunks: list[DocumentChunk]):
        """Replace the collection contents with the supplied chunks."""
        self.clear()
        self.add_documents(chunks)

    def search(self, query: str | list[str], k: int = 5) -> list[list[str]]:
        """Return up to ``k`` document chunks for each query."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, list):
            queries = query
        else:
            raise ValueError("query must be a string or a list of strings")

        if not queries:
            raise ValueError("query must not be empty")
        if self.num_docs == 0:
            return []

        results = self.collection.query(
            query_texts=queries,
            n_results=min(k, self.num_docs),
        )
        return results.get("documents") or []

    def delete(self, document_id: str) -> bool:
        """Delete one document and report whether the collection changed."""
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("document_id must be a non-empty string")

        before = self.num_docs
        self.collection.delete(ids=[document_id])
        return self.num_docs < before

    def clear(self):
        """Remove the collection and recreate it empty."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._create_collection()
