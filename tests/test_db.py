import hashlib

import numpy as np
import pytest

from db.chroma import DB


class FakeModel:
    def __init__(self, dimensions=8):
        self.dimensions = dimensions

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False):
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.frombuffer(digest[: self.dimensions], dtype=np.uint8)
            vectors.append(vector.astype(np.float32))

        return np.array(vectors)


@pytest.fixture
def db(tmp_path):
    return DB(db_path=str(tmp_path / "chroma"), model=FakeModel())


def test_add_documents_and_search(db):
    db.addDocs(["alpha", "beta", "gamma"])

    assert db.num_docs == 3
    results = db.semanticSearch("alpha")
    assert len(results) == 1
    assert set(results[0]) == {"alpha", "beta", "gamma"}


def test_add_single_string_document(db):
    db.addDocs("alpha")

    assert db.num_docs == 1
    assert db.semanticSearch("alpha")[0] == ["alpha"]


def test_search_empty_database_returns_empty_list(db):
    assert db.semanticSearch("anything") == []


def test_remove_document(db):
    db.addDocs(["alpha", "beta"])
    doc_id = db.collection.get()["ids"][0]

    db.remove(doc_id)

    assert db.num_docs == 1
    assert len(db.collection.get()["ids"]) == 1


def test_clear_database(db):
    db.addDocs(["alpha", "beta"])

    db.clear()

    assert db.num_docs == 0
    assert db.semanticSearch("alpha") == []


def test_document_count_survives_reopen(tmp_path):
    path = str(tmp_path / "chroma")
    DB(db_path=path, model=FakeModel()).addDocs(["alpha", "beta"])

    reopened = DB(db_path=path, model=FakeModel())

    assert reopened.num_docs == 2
    assert set(reopened.semanticSearch("alpha")[0]) == {"alpha", "beta"}
