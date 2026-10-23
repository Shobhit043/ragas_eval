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


def make_chunks(*texts):
    return [
        {
            "id": str(index),
            "text": text,
            "metadata": {"source": "test.txt", "chunk_index": index},
        }
        for index, text in enumerate(texts)
    ]


@pytest.fixture
def db(tmp_path):
    return DB(db_path=str(tmp_path / "chroma"), model=FakeModel())


def test_add_documents_and_search(db):
    db.add_documents(make_chunks("alpha", "beta", "gamma"))

    assert db.num_docs == 3
    results = db.search("alpha")
    assert len(results) == 1
    assert set(results[0]) == {"alpha", "beta", "gamma"}


def test_add_documents_preserves_metadata(db):
    db.add_documents(make_chunks("alpha"))

    stored = db.collection.get(include=["metadatas"])

    assert stored["metadatas"] == [{"source": "test.txt", "chunk_index": 0}]


def test_add_documents_batches_requests(db):
    db.add_documents(make_chunks("alpha", "beta", "gamma"), batch_size=1)

    assert db.num_docs == 3


def test_upsert_updates_existing_chunk(db):
    db.add_documents(make_chunks("alpha"))
    db.add_documents(
        [
            {
                "id": "0",
                "text": "updated alpha",
                "metadata": {"source": "test.txt", "chunk_index": 0},
            }
        ]
    )

    assert db.num_docs == 1
    assert db.search("updated alpha")[0] == ["updated alpha"]


def test_replace_documents_removes_stale_chunks(db):
    db.add_documents(make_chunks("alpha", "beta"))

    db.replace_documents(make_chunks("gamma"))

    assert db.num_docs == 1
    assert db.search("gamma")[0] == ["gamma"]


def test_search_empty_database_returns_empty_list(db):
    assert db.search("anything") == []


def test_search_rejects_empty_query(db):
    with pytest.raises(ValueError, match="must not be empty"):
        db.search([])


def test_delete_document(db):
    db.add_documents(make_chunks("alpha", "beta"))

    assert db.delete("0") is True

    assert db.num_docs == 1
    assert len(db.collection.get()["ids"]) == 1


def test_delete_unknown_document_returns_false(db):
    db.add_documents(make_chunks("alpha"))

    assert db.delete("missing") is False


def test_clear_database(db):
    db.add_documents(make_chunks("alpha", "beta"))

    db.clear()

    assert db.num_docs == 0
    assert db.search("alpha") == []


def test_document_count_survives_reopen(tmp_path):
    path = str(tmp_path / "chroma")
    DB(db_path=path, model=FakeModel()).add_documents(make_chunks("alpha", "beta"))

    reopened = DB(db_path=path, model=FakeModel())

    assert reopened.num_docs == 2
    assert set(reopened.search("alpha")[0]) == {"alpha", "beta"}
