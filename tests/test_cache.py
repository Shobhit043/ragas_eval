import json

import pytest

import cache


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    path = tmp_path / "eval_cache.json"
    monkeypatch.setattr(cache, "CACHE_PATH", str(path))
    return path


def test_load_missing_cache_returns_empty(cache_path):
    assert cache.load_cache() == {"questions": {}}


def test_save_and_load_roundtrip(cache_path):
    data = {"questions": {"q1": {"response": "a", "metrics": {"faithfulness": 0.5}}}}

    cache.save_cache(data)

    assert cache.load_cache() == data
    assert json.loads(cache_path.read_text(encoding="utf-8")) == data


def test_prune_removes_stale_questions(cache_path):
    data = {
        "questions": {
            "keep": {"metrics": {}},
            "remove": {"metrics": {}},
        }
    }

    assert cache.prune_cache(data, ["keep"]) is True
    assert list(data["questions"]) == ["keep"]


def test_prune_keeps_current_questions(cache_path):
    data = {"questions": {"keep": {"metrics": {}}}}

    assert cache.prune_cache(data, ["keep"]) is False
    assert list(data["questions"]) == ["keep"]


def test_prune_empty_cache(cache_path):
    assert cache.prune_cache({"questions": {}}, ["q1"]) is False
