from types import SimpleNamespace

import pandas as pd

import eval


class FakeDB:
    def __init__(self, results=None):
        self.results = results if results is not None else [["context"]]

    def semanticSearch(self, question):
        return self.results


class FakeScorer:
    def __init__(self, value=0.75):
        self.value = value
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


def test_prepare_entry_creates_new_entry():
    cache = {"questions": {}}

    entry, updated = eval.prepare_entry(cache, "q1", "reference")

    assert updated is True
    assert entry["reference"] == "reference"
    assert entry["metrics"] == {}
    assert cache["questions"]["q1"] is entry


def test_prepare_entry_preserves_scores_for_legacy_entry():
    cache = {
        "questions": {
            "q1": {
                "response": "answer",
                "retrieved_contexts": ["context"],
                "metrics": {"faithfulness": 1.0},
            }
        }
    }

    entry, updated = eval.prepare_entry(cache, "q1", "reference")

    assert updated is True
    assert entry["metrics"] == {"faithfulness": 1.0}
    assert entry["reference"] == "reference"


def test_prepare_entry_clears_scores_when_reference_changes():
    cache = {"questions": {"q1": {"reference": "old", "metrics": {"faithfulness": 1.0}}}}

    entry, updated = eval.prepare_entry(cache, "q1", "new")

    assert updated is True
    assert entry["metrics"] == {}


def test_prepare_entry_is_unchanged_when_reference_matches():
    cache = {"questions": {"q1": {"reference": "same", "metrics": {"faithfulness": 1.0}}}}

    entry, updated = eval.prepare_entry(cache, "q1", "same")

    assert updated is False
    assert entry["metrics"] == {"faithfulness": 1.0}


def test_retrieve_contexts_handles_list_result():
    assert eval.retrieve_contexts(FakeDB([["one", "two"]]), "question") == ["one", "two"]


def test_retrieve_contexts_handles_string_result():
    assert eval.retrieve_contexts(FakeDB(["one"]), "question") == ["one"]


def test_retrieve_contexts_handles_empty_result():
    assert eval.retrieve_contexts(FakeDB([]), "question") == []


async def test_score_metric_passes_only_declared_inputs():
    scorer = FakeScorer(value=0.5)
    spec = {"scorer": scorer, "inputs": ("response", "reference")}
    sample = {
        "user_input": "question",
        "response": "answer",
        "reference": "reference",
        "retrieved_contexts": ["context"],
    }

    score = await eval.score_metric(spec, sample)

    assert score == 0.5
    assert scorer.calls == [{"response": "answer", "reference": "reference"}]


async def test_ensure_answer_reuses_matching_cache(monkeypatch):
    async def fail_if_called(context, question):
        raise AssertionError("generate_answer should not be called")

    monkeypatch.setattr(eval, "generate_answer", fail_if_called)
    monkeypatch.setattr(eval, "save_cache", lambda cache: None)
    entry = {
        "response": "cached",
        "retrieved_contexts": ["context"],
        "metrics": {"faithfulness": 1.0},
    }

    await eval.ensure_answer({"questions": {}}, FakeDB(), entry, "question")

    assert entry["response"] == "cached"
    assert entry["metrics"] == {"faithfulness": 1.0}


async def test_ensure_answer_regenerates_when_contexts_change(monkeypatch):
    saved = []

    async def fake_answer(context, question):
        return "new answer"

    monkeypatch.setattr(eval, "generate_answer", fake_answer)
    monkeypatch.setattr(eval, "save_cache", saved.append)
    entry = {
        "response": "old answer",
        "retrieved_contexts": ["old context"],
        "metrics": {"faithfulness": 1.0},
    }

    await eval.ensure_answer({"questions": {}}, FakeDB([["new context"]]), entry, "question")

    assert entry["response"] == "new answer"
    assert entry["retrieved_contexts"] == ["new context"]
    assert entry["metrics"] == {}
    assert len(saved) == 1


async def test_main_writes_results_and_cache(tmp_path, monkeypatch):
    class FakeRuntimeDB:
        def clear(self):
            return None

        def addDocs(self, docs=None, metadata=None):
            return None

        def semanticSearch(self, question):
            return [["context"]]

    async def fake_answer(context, question):
        return f"answer to {question}"

    metrics = {
        "semantic_similarity": {
            "scorer": FakeScorer(value=0.8),
            "inputs": ("reference", "response"),
        }
    }
    cache = {"questions": {}}
    results_path = tmp_path / "results.csv"

    monkeypatch.setattr(eval, "EmbeddModel", lambda: SimpleNamespace(load_model=lambda: object()))
    monkeypatch.setattr(eval, "create_embeddings", lambda model: object())
    monkeypatch.setattr(eval, "DB", lambda model: FakeRuntimeDB())
    monkeypatch.setattr(eval, "build_metrics", lambda llm, embeddings: metrics)
    monkeypatch.setattr(eval, "load_cache", lambda: cache)
    monkeypatch.setattr(eval, "save_cache", lambda data: None)
    monkeypatch.setattr(eval, "generate_answer", fake_answer)
    monkeypatch.setattr(eval, "generate_plots", lambda df, m, out: [])
    monkeypatch.setattr(eval, "user_inputs", ["q1", "q2"])
    monkeypatch.setattr(eval, "references", ["r1", "r2"])
    monkeypatch.setattr(eval, "docs", ["doc"])
    monkeypatch.setattr(eval, "RESULTS_PATH", str(results_path))
    monkeypatch.setattr(eval, "PLOTS_DIR", str(tmp_path / "plots"))

    await eval.main()

    results = pd.read_csv(results_path)
    assert results["user_input"].tolist() == ["q1", "q2"]
    assert results["semantic_similarity"].tolist() == [0.8, 0.8]
    assert cache["questions"]["q1"]["metrics"]["semantic_similarity"] == 0.8
