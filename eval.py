import asyncio
import json
from pathlib import Path

import pandas as pd
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    ContextUtilization,
    FactualCorrectness,
    Faithfulness,
    NoiseSensitivity,
    ResponseGroundedness,
    SemanticSimilarity,
)

from cache import load_cache, prune_cache, save_cache
from db.chroma import DB
from document_loader import DEFAULT_DOCUMENTS_DIR, load_chunks
from embeddings import create_embeddings
from llm import generate_answer, ragas_llm, with_retries
from models.model import EmbeddModel
from plots import PLOTS_DIR, generate_plots

PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = DEFAULT_DOCUMENTS_DIR
DATASET_PATH = PROJECT_ROOT / "eval_dataset.json"
RESULTS_PATH = PROJECT_ROOT / "ragas_results.csv"


def load_evaluation_dataset(path=DATASET_PATH):
    """Load question/reference pairs and fail early on malformed input."""
    with open(path, encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"Evaluation dataset must be a non-empty list: {path}")

    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict) or not item.get("question") or not item.get("reference"):
            raise ValueError(
                "Each dataset item needs a non-empty 'question' and 'reference' "
                f"(item {index} in {path})"
            )

    return dataset


METRIC_INPUTS = {
    "answer_relevancy": ("user_input", "response"),
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "answer_correctness": ("user_input", "response", "reference"),
    "factual_correctness": ("response", "reference"),
    "semantic_similarity": ("reference", "response"),
    "context_precision": ("user_input", "reference", "retrieved_contexts"),
    "context_recall": ("user_input", "reference", "retrieved_contexts"),
    "context_entity_recall": ("reference", "retrieved_contexts"),
    "response_groundedness": ("response", "retrieved_contexts"),
    "context_utilization": ("user_input", "response", "retrieved_contexts"),
    "noise_sensitivity": ("user_input", "response", "reference", "retrieved_contexts"),
}

LOWER_IS_BETTER = {"noise_sensitivity"}


def build_metrics(llm, embeddings):
    """Create the metric scorers and declare the sample fields each one needs."""
    scorers = {
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "faithfulness": Faithfulness(llm=llm),
        "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
        "factual_correctness": FactualCorrectness(llm=llm),
        "semantic_similarity": SemanticSimilarity(embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "context_entity_recall": ContextEntityRecall(llm=llm),
        "response_groundedness": ResponseGroundedness(llm=llm),
        "context_utilization": ContextUtilization(llm=llm),
        "noise_sensitivity": NoiseSensitivity(llm=llm),
    }

    metrics = {}

    for name, inputs in METRIC_INPUTS.items():
        spec = {"scorer": scorers[name], "inputs": inputs}

        if name in LOWER_IS_BETTER:
            spec["lower_is_better"] = True

        metrics[name] = spec

    return metrics


def retrieve_contexts(db_client, question):
    results = db_client.search(question)

    if not results:
        return []

    result = results[0]

    if isinstance(result, str):
        return [result]

    return list(result)


async def ensure_answer(cache, db_client, entry, question, save=None):
    save = save or save_cache
    contexts = retrieve_contexts(db_client, question)

    if entry.get("response") and entry.get("retrieved_contexts") == contexts:
        return

    entry["retrieved_contexts"] = contexts
    entry["response"] = await generate_answer("\n\n".join(contexts), question)

    # The previous scores belong to a different answer.
    entry["metrics"] = {}
    save(cache)


def prepare_entry(cache, question, reference):
    entry = cache["questions"].setdefault(question, {})
    entry.setdefault("metrics", {})

    updated = entry.get("reference") != reference

    # A changed reference answer invalidates all metric scores for this question.
    # Entries migrated from the old cache format have no stored reference.
    if updated and entry.get("reference") is not None:
        entry["metrics"] = {}

    entry["reference"] = reference

    return entry, updated


async def score_metric(spec, sample):
    inputs = {field: sample[field] for field in spec["inputs"]}
    result = await with_retries(spec["scorer"].ascore, **inputs)
    return float(result.value)


async def evaluate_dataset(dataset, db_client, metrics, cache, verbose=False, save=None):
    """Answer and score every dataset item, returning one row per question."""
    save = save or save_cache
    rows = []

    for item in dataset:
        question = item["question"]
        reference = item["reference"]
        entry, entry_updated = prepare_entry(cache, question, reference)

        if entry_updated:
            save(cache)

        await ensure_answer(cache, db_client, entry, question, save=save)

        sample = {
            "user_input": question,
            "response": entry["response"],
            "retrieved_contexts": entry["retrieved_contexts"],
            "reference": reference,
        }

        if verbose:
            print(f"\nEvaluating: {question}")

        for name, spec in metrics.items():
            cached_score = entry["metrics"].get(name)

            if cached_score is not None:
                if verbose:
                    print(f"  {name}: cached ({cached_score:.4f})")
                continue

            entry["metrics"][name] = await score_metric(spec, sample)
            save(cache)

            if verbose:
                print(f"  {name}: {entry['metrics'][name]:.4f}")

        row = {
            "user_input": question,
            "response": entry["response"],
            "retrieved_contexts": entry["retrieved_contexts"],
            "reference": reference,
        }
        row.update({name: entry["metrics"][name] for name in metrics})
        rows.append(row)

    return pd.DataFrame(rows)


async def main():
    dataset = load_evaluation_dataset()
    chunks = load_chunks(DOCUMENTS_DIR)

    model = EmbeddModel().load_model()
    embeddings = create_embeddings(model)
    db_client = DB(model=model)
    metrics = build_metrics(ragas_llm, embeddings)

    db_client.replace_documents(chunks)

    cache = load_cache()
    questions = [item["question"] for item in dataset]

    if prune_cache(cache, questions):
        save_cache(cache)

    df = await evaluate_dataset(dataset, db_client, metrics, cache, verbose=True)

    print("\n=== RAGAS Evaluation Results ===")
    print(df.to_string(index=False))

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")

    plot_paths = generate_plots(df, metrics, PLOTS_DIR)
    print(f"\nSaved {len(plot_paths)} plots to {PLOTS_DIR}/")
    for path in plot_paths:
        print(f"  {path}")

    print("\n=== Average Metric Scores ===")
    averages = df[list(metrics)].mean()
    for name, value in averages.items():
        suffix = " (lower is better)" if metrics[name].get("lower_is_better") else ""
        print(f"{name}: {value:.4f}{suffix}")


if __name__ == "__main__":
    asyncio.run(main())
