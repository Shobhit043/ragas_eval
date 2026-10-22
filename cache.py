import json
import os

CACHE_PATH = "eval_cache.json"


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}

    with open(CACHE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2, ensure_ascii=False)
