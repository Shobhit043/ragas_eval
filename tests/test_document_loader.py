import pytest

import document_loader


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakePdfReader:
    is_encrypted = False
    pages = [
        FakePage("First PDF page."),
        FakePage(""),
        FakePage("Third PDF page."),
    ]


def test_load_chunks_from_text_files_recursively(tmp_path):
    documents = tmp_path / "documents"
    (documents / "nested").mkdir(parents=True)
    (documents / "alpha.txt").write_text("Alpha content.", encoding="utf-8")
    (documents / "nested" / "beta.txt").write_text("Beta content.", encoding="utf-8")
    (documents / "ignored.md").write_text("Ignored.", encoding="utf-8")

    chunks = document_loader.load_chunks(documents, chunk_size=100, chunk_overlap=10)

    assert {chunk["metadata"]["source"] for chunk in chunks} == {
        "alpha.txt",
        "nested/beta.txt",
    }
    assert {chunk["text"] for chunk in chunks} == {"Alpha content.", "Beta content."}
    assert len({chunk["id"] for chunk in chunks}) == 2


def test_load_chunks_splits_long_text(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    text = " ".join(f"Sentence number {index}." for index in range(40))
    (documents / "long.txt").write_text(text, encoding="utf-8")

    chunks = document_loader.load_chunks(documents, chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert [chunk["metadata"]["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(len(chunk["text"]) <= 80 for chunk in chunks)


def test_load_chunks_extracts_pdf_pages(tmp_path, monkeypatch):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "paper.pdf").write_bytes(b"fake pdf")
    monkeypatch.setattr(document_loader, "PdfReader", lambda path: FakePdfReader())

    chunks = document_loader.load_chunks(documents, chunk_size=100, chunk_overlap=10)

    assert [chunk["metadata"]["page"] for chunk in chunks] == [1, 3]
    assert {chunk["text"] for chunk in chunks} == {"First PDF page.", "Third PDF page."}


def test_load_chunks_skips_hidden_files(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / ".hidden.txt").write_text("Hidden.", encoding="utf-8")
    (documents / "visible.txt").write_text("Visible.", encoding="utf-8")

    chunks = document_loader.load_chunks(documents, chunk_size=100, chunk_overlap=10)

    assert [chunk["text"] for chunk in chunks] == ["Visible."]


def test_load_chunks_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No .pdf or .txt files"):
        document_loader.load_chunks(tmp_path)


def test_load_chunks_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Documents directory not found"):
        document_loader.load_chunks(tmp_path / "missing")


def test_load_chunks_rejects_invalid_overlap(tmp_path):
    with pytest.raises(ValueError, match="chunk_overlap"):
        document_loader.load_chunks(tmp_path, chunk_size=100, chunk_overlap=100)
