import pytest

import app


class FakeUpload:
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def getbuffer(self):
        return memoryview(self.content)


def test_save_uploads_sanitizes_names(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "UPLOAD_DIR", tmp_path / "uploads")

    saved = app.save_uploads([FakeUpload("../notes.txt", b"hello")])

    assert saved == [tmp_path / "uploads" / "notes.txt"]
    assert saved[0].read_bytes() == b"hello"


def test_save_uploads_ignores_unsupported_files(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "UPLOAD_DIR", tmp_path / "uploads")

    assert app.save_uploads([FakeUpload("notes.md", b"ignored")]) == []


def test_metrics_for_reference_hides_reference_metrics():
    inputs = {
        "faithfulness": ("user_input", "response", "retrieved_contexts"),
        "answer_correctness": ("user_input", "response", "reference"),
    }

    assert app.metrics_for_reference(inputs, has_reference=False) == ["faithfulness"]
    assert app.metrics_for_reference(inputs, has_reference=True) == [
        "faithfulness",
        "answer_correctness",
    ]


async def test_evaluate_question_returns_answer_contexts_and_scores(monkeypatch):
    class FakeEvaluation:
        METRIC_INPUTS = {
            "faithfulness": ("user_input", "response", "retrieved_contexts"),
        }
        ragas_llm = object()

        def retrieve_contexts(self, db, question):
            return ["context"]

        async def generate_answer(self, context, question):
            assert context == "context"
            assert question == "question"
            return "answer"

        def build_metrics(self, llm, embeddings):
            return {"faithfulness": {"scorer": object(), "inputs": ("response",)}}

        async def score_metric(self, spec, sample):
            assert sample["reference"] == "reference"
            return 0.5

    monkeypatch.setattr(app, "get_evaluation", lambda: FakeEvaluation())
    monkeypatch.setattr(app, "get_database", lambda: object())
    monkeypatch.setattr(app, "get_embeddings", lambda: object())

    result = await app.evaluate_question("question", "reference", ["faithfulness"])

    assert result["answer"] == "answer"
    assert result["contexts"] == ["context"]
    assert result["scores"] == {"faithfulness": 0.5}


async def test_evaluate_question_rejects_empty_index(monkeypatch):
    class FakeEvaluation:
        METRIC_INPUTS = {
            "faithfulness": ("user_input", "response", "retrieved_contexts"),
        }

        def retrieve_contexts(self, db, question):
            return []

    monkeypatch.setattr(app, "get_evaluation", lambda: FakeEvaluation())
    monkeypatch.setattr(app, "get_database", lambda: object())

    with pytest.raises(RuntimeError, match="No documents are indexed"):
        await app.evaluate_question("question", "reference", ["faithfulness"])


async def test_evaluate_question_requires_reference_for_reference_metrics(monkeypatch):
    class FakeEvaluation:
        METRIC_INPUTS = {
            "answer_correctness": ("user_input", "response", "reference"),
        }

    monkeypatch.setattr(app, "get_evaluation", lambda: FakeEvaluation())

    with pytest.raises(ValueError, match="Reference-based metrics"):
        await app.evaluate_question("question", "", ["answer_correctness"])
