import os

import chromadb

from db.embedding_func import EmbeddingFunction


class DB:
    def __init__(self, db_path="chroma_db", collection_name="data", model=None):
        cwd = os.path.dirname(os.path.realpath(__file__))
        if db_path == "chroma_db":
            db_path = os.path.join(cwd, db_path)

        os.makedirs(db_path, exist_ok=True)

        if model is None:
            raise ValueError("model must be a SentenceTransformer instance")

        self.client = chromadb.PersistentClient(path=db_path)
        self.llm_ef = EmbeddingFunction(model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.llm_ef,
        )
        self.num_docs = self.collection.count()

    def addDocs(self, docs=None, metadata=None):
        if docs is None:
            raise ValueError("must pass string or list of string.")

        if isinstance(docs, str):
            docs = [docs]

        start = self.num_docs
        self.collection.add(
            documents=docs,
            ids=[str(i) for i in range(start, start + len(docs))],
            metadatas=metadata,
        )
        self.num_docs = self.collection.count()

    def semanticSearch(self, q=None):
        if q is None:
            raise ValueError("no query passed or None is passed.")
        if not isinstance(q, str) and not isinstance(q, list):
            raise ValueError("q must be a string or a list of strings")
        if isinstance(q, str):
            q = [q]

        if self.num_docs == 0:
            return []

        results = self.collection.query(
            query_texts=q,
            n_results=min(5, self.num_docs),
        )
        return results["documents"]

    def remove(self, target_id=None):
        if self.num_docs == 0:
            print("DB is already empty")
            return

        if target_id is None:
            raise ValueError("must pass target_id of document")

        if not isinstance(target_id, str):
            raise ValueError("target_id must be a string")

        try:
            self.collection.delete(ids=[target_id])
            self.num_docs = self.collection.count()
        except Exception:
            print("id not found in db")

    def clear(self):
        existing = self.collection.get()

        if not existing["ids"]:
            print("DB is already empty")
            return

        self.collection.delete(ids=existing["ids"])
        self.num_docs = 0
        print("DB cleared!")
