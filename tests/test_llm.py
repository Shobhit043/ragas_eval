from types import SimpleNamespace

import pytest

import llm


class TransientError(Exception):
    pass


def error_with_headers(headers):
    return SimpleNamespace(response=SimpleNamespace(headers=headers))


def test_retry_after_seconds_reads_header():
    error = error_with_headers({"retry-after": "7"})

    assert llm.retry_after_seconds(error, fallback=1) == 7.0


def test_retry_after_seconds_ignores_invalid_header():
    error = error_with_headers({"retry-after": "not-a-number"})

    assert llm.retry_after_seconds(error, fallback=2.5) == 2.5


def test_retry_after_seconds_without_response():
    assert llm.retry_after_seconds(TransientError(), fallback=3) == 3


def test_build_answer_messages():
    messages = llm.build_answer_messages("some context", "a question")

    assert messages[0]["role"] == "system"
    assert "some context" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "a question"}


async def test_with_retries_retries_transient_errors(monkeypatch):
    attempts = 0
    delays = []

    async def call():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientError()
        return "ok"

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(llm, "RETRYABLE_ERRORS", (TransientError,))
    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm.random, "uniform", lambda start, end: 0)

    assert await llm.with_retries(call, max_attempts=3, base_wait=0) == "ok"
    assert attempts == 3
    assert delays == [0, 0]


async def test_with_retries_raises_after_max_attempts(monkeypatch):
    attempts = 0

    async def call():
        nonlocal attempts
        attempts += 1
        raise TransientError()

    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(llm, "RETRYABLE_ERRORS", (TransientError,))
    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm.random, "uniform", lambda start, end: 0)

    with pytest.raises(TransientError):
        await llm.with_retries(call, max_attempts=2, base_wait=0)

    assert attempts == 2
