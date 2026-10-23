import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import app


def test_dataset_from_editor_skips_blank_rows():
    frame = pd.DataFrame(
        {
            "question": ["What is it?", "", "How does it work?"],
            "reference": ["A thing.", "", ""],
        }
    )

    dataset = app.dataset_from_editor(frame)

    assert dataset == [
        {"question": "What is it?", "reference": "A thing."},
        {"question": "How does it work?", "reference": ""},
    ]


def test_dataset_from_editor_rejects_reference_without_question():
    frame = pd.DataFrame({"question": [""], "reference": ["Orphan answer."]})

    with pytest.raises(ValueError, match="Row 1 needs a question"):
        app.dataset_from_editor(frame)


def test_dataset_from_editor_allows_an_empty_table():
    assert app.dataset_from_editor(pd.DataFrame(columns=["question", "reference"])) == []


def test_parse_batch_json_accepts_questions_and_references():
    payload = json.dumps(
        [
            {"question": "Question one?", "reference": "Answer one."},
            {"question": "Question two?"},
        ]
    ).encode("utf-8")

    assert app.parse_batch_json(payload) == [
        {"question": "Question one?", "reference": "Answer one."},
        {"question": "Question two?", "reference": ""},
    ]


def test_parse_batch_json_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="non-empty list"):
        app.parse_batch_json("{}")

    with pytest.raises(ValueError, match="needs a non-empty 'question'"):
        app.parse_batch_json('[{"reference": "answer"}]')


def test_run_batch_evaluation_returns_downloads_without_local_files(monkeypatch):
    results = pd.DataFrame(
        {
            "user_input": ["q1"],
            "response": ["a1"],
            "reference": ["r1"],
            "faithfulness": [0.5],
        }
    )

    class FakeEvaluation:
        METRIC_INPUTS = {
            "faithfulness": ("user_input", "response", "retrieved_contexts"),
        }
        ragas_llm = object()

        def build_metrics(self, llm, embeddings):
            return {"faithfulness": {"scorer": object(), "inputs": ("response",)}}

        def prune_cache(self, cache, questions):
            return False

        async def evaluate_dataset(self, dataset, db, metrics, cache, save=None):
            assert save is not None
            return results

        def generate_plots(self, df, metrics, output_dir):
            path = Path(output_dir) / "average_metric_scores.png"
            path.write_bytes(b"plot")
            return [str(path)]

    monkeypatch.setattr(app, "get_evaluation", lambda: FakeEvaluation())
    monkeypatch.setattr(app, "get_database", lambda: SimpleNamespace(num_docs=1))
    monkeypatch.setattr(app, "get_embeddings", lambda: object())

    result = app.run_batch_evaluation(
        [{"question": "q1", "reference": "r1"}],
        ["faithfulness"],
        cache={"questions": {}},
    )

    assert result["df"].equals(results)
    assert result["csv"] == results.to_csv(index=False).encode("utf-8")
    assert result["plots"] == [{"name": "average_metric_scores.png", "data": b"plot"}]


def test_run_batch_evaluation_rejects_empty_index(monkeypatch):
    monkeypatch.setattr(app, "get_evaluation", lambda: SimpleNamespace())
    monkeypatch.setattr(app, "get_database", lambda: SimpleNamespace(num_docs=0))

    with pytest.raises(RuntimeError, match="No documents are indexed"):
        app.run_batch_evaluation([{"question": "q1", "reference": "r1"}], ["faithfulness"])


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
