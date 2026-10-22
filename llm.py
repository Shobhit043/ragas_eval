import os
import random
import asyncio

import instructor
from dotenv import load_dotenv
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from ragas.llms.base import InstructorLLM

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "nvidia/nemotron-3.5-lightning-30b-a3b"

RETRYABLE_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    raise ValueError("NVIDIA_API_KEY is missing from your .env file")

nvidia_client = AsyncOpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=nvidia_api_key,
    max_retries=0,
    timeout=180.0,
)


def retry_after_seconds(error, fallback):
    response = getattr(error, "response", None)
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
    return fallback


async def with_retries(
    call,
    *args,
    max_attempts=8,
    base_wait=5,
    max_wait=120,
    **kwargs,
):
    for attempt in range(1, max_attempts + 1):
        try:
            return await call(*args, **kwargs)

        except RETRYABLE_ERRORS as error:
            if attempt == max_attempts:
                raise

            fallback_wait = min(max_wait, base_wait * 2 ** (attempt - 1))
            wait_time = retry_after_seconds(error, fallback_wait)
            wait_time += random.uniform(0, 2)

            print(
                f"Rate limit or transient error "
                f"(attempt {attempt}/{max_attempts}). "
                f"Retrying in {wait_time:.0f} seconds..."
            )
            await asyncio.sleep(wait_time)


def build_answer_messages(context, question):
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                f"Use the following context to answer:\n\n{context}"
            ),
        },
        {"role": "user", "content": question},
    ]


async def generate_answer(context, question):
    response = await with_retries(
        nvidia_client.chat.completions.create,
        model=MODEL_NAME,
        messages=build_answer_messages(context, question),
        max_tokens=4096,
        temperature=0.2,
        reasoning_effort="low",
    )

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError(
            "The model returned an empty answer. Its reasoning may have "
            "used the entire max_tokens budget."
        )
    return answer


ragas_llm = InstructorLLM(
    client=instructor.from_openai(nvidia_client, mode=instructor.Mode.MD_JSON),
    model=MODEL_NAME,
    provider="openai",
    max_tokens=8192,
    reasoning_effort="none",
)
