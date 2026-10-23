import json
import os

CACHE_PATH = "eval_cache.json"


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {"questions": {}}

    with open(CACHE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2, ensure_ascii=False)


def prune_cache(cache, questions):
    """Remove cached entries for questions that are no longer in the dataset."""
    keep = set(questions)
    cached_questions = cache["questions"]
    stale = [question for question in cached_questions if question not in keep]

    for question in stale:
        del cached_questions[question]

    return bool(stale)
