import chromadb
import os
from db.embedding_func import EmbeddingFunction


class DB:
    def __init__(self, db_path = "chroma_db", collection_name = "data", model = None):
        CWD = os.path.dirname(os.path.realpath(__file__))
        if db_path == "chroma_db":
            db_path = os.path.join(CWD, db_path)

        if os.path.isdir(db_path):
            os.makedirs(db_path, exist_ok=True)

        if model is None:
            raise ValueError("model must be an llm instance")
        
        

        self.num_docs = 0
        self.client = chromadb.PersistentClient(path=db_path)

        self.llm_ef = EmbeddingFunction(model)

        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.llm_ef
        )

    
    def addDocs(self, docs = None, metadata = None):
        if docs is None:
            raise ValueError("must pass string or list of string.")

        if isinstance(docs, str):
            docs = list(docs)

        num_of_docs = len(docs)
        self.collection.add(
            documents = docs,
            ids = [str(i) for i in range(self.num_docs, self.num_docs + num_of_docs)],
            metadatas=metadata
        )
        self.num_docs += num_of_docs
    
    def semanticSearch(self, q = None):
        if q is None:
            raise ValueError("no query passed or None is passed.")
        if not isinstance(q, str) and not isinstance(q, list):
            raise ValueError("q must be a string or a list of strings") 
        if isinstance(q, str):
            q = [q]

        results = self.collection.query(
            query_texts=q,
            n_results=min(5, self.num_docs)
        )
        return results['documents']

    def remove(self, target_id = None):
        if self.num_docs == 0:
            print("DB is already empty")
            return
        
        if target_id is None:
            raise ValueError("must pass target_id of document")
        
        if not isinstance(target_id, str):
            raise ValueError("target_id must be a string")
        
        try:
            self.collection.delete(ids=[target_id])
            self.num_docs -= 1
        except:
            print("id not found in db")

    def clear(self):
        existing = self.collection.get()

        if not existing["ids"]:
            print("DB is already empty")
            return

        self.collection.delete(ids=existing["ids"])
        self.num_docs = 0
        print("DB cleared!")

