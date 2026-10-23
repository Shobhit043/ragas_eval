import asyncio

import pandas as pd

from cache import load_cache, prune_cache, save_cache
from data import docs, references, user_inputs
from db.chroma import DB
from embeddings import embeddings, model
from llm import generate_answer, ragas_llm, with_retries
from plots import PLOTS_DIR, generate_plots
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

RESULTS_PATH = "ragas_results.csv"

db_client = DB(model=model)

# Each metric lists the sample fields it needs. Those names match both the
# metric's ascore() arguments and the keys of the sample dict in main().
METRICS = {
    "answer_relevancy": {
        "scorer": AnswerRelevancy(llm=ragas_llm, embeddings=embeddings),
        "inputs": ("user_input", "response"),
    },
    "faithfulness": {
        "scorer": Faithfulness(llm=ragas_llm),
        "inputs": ("user_input", "response", "retrieved_contexts"),
    },
    "answer_correctness": {
        "scorer": AnswerCorrectness(llm=ragas_llm, embeddings=embeddings),
        "inputs": ("user_input", "response", "reference"),
    },
    "factual_correctness": {
        "scorer": FactualCorrectness(llm=ragas_llm),
        "inputs": ("response", "reference"),
    },
    "semantic_similarity": {
        "scorer": SemanticSimilarity(embeddings=embeddings),
        "inputs": ("reference", "response"),
    },
    "context_precision": {
        "scorer": ContextPrecision(llm=ragas_llm),
        "inputs": ("user_input", "reference", "retrieved_contexts"),
    },
    "context_recall": {
        "scorer": ContextRecall(llm=ragas_llm),
        "inputs": ("user_input", "reference", "retrieved_contexts"),
    },
    "context_entity_recall": {
        "scorer": ContextEntityRecall(llm=ragas_llm),
        "inputs": ("reference", "retrieved_contexts"),
    },
    "response_groundedness": {
        "scorer": ResponseGroundedness(llm=ragas_llm),
        "inputs": ("response", "retrieved_contexts"),
    },
    "context_utilization": {
        "scorer": ContextUtilization(llm=ragas_llm),
        "inputs": ("user_input", "response", "retrieved_contexts"),
    },
    "noise_sensitivity": {
        "scorer": NoiseSensitivity(llm=ragas_llm),
        "inputs": ("user_input", "response", "reference", "retrieved_contexts"),
        "lower_is_better": True,
    },
}


def retrieve_contexts(question):
    result = db_client.semanticSearch(question)[0]

    if isinstance(result, str):
        return [result]

    return list(result)


async def ensure_answer(cache, entry, question):
    contexts = retrieve_contexts(question)

    if entry.get("response") and entry.get("retrieved_contexts") == contexts:
        return

    entry["retrieved_contexts"] = contexts
    entry["response"] = await generate_answer("\n\n".join(contexts), question)

    # The previous scores belong to a different answer.
    entry["metrics"] = {}
    save_cache(cache)


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


async def main():
    db_client.clear()
    db_client.addDocs(docs=docs)

    cache = load_cache()

    if prune_cache(cache, user_inputs):
        save_cache(cache)

    rows = []

    for index, question in enumerate(user_inputs):
        reference = references[index]
        entry, entry_updated = prepare_entry(cache, question, reference)

        if entry_updated:
            save_cache(cache)

        await ensure_answer(cache, entry, question)

        sample = {
            "user_input": question,
            "response": entry["response"],
            "retrieved_contexts": entry["retrieved_contexts"],
            "reference": reference,
        }

        print(f"\nEvaluating: {question}")

        for name, spec in METRICS.items():
            cached_score = entry["metrics"].get(name)

            if cached_score is not None:
                print(f"  {name}: cached ({cached_score:.4f})")
                continue

            entry["metrics"][name] = await score_metric(spec, sample)
            save_cache(cache)
            print(f"  {name}: {entry['metrics'][name]:.4f}")

        row = {
            "user_input": question,
            "response": entry["response"],
            "retrieved_contexts": entry["retrieved_contexts"],
            "reference": reference,
        }
        row.update({name: entry["metrics"][name] for name in METRICS})
        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n=== RAGAS Evaluation Results ===")
    print(df.to_string(index=False))

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")

    plot_paths = generate_plots(df, METRICS, PLOTS_DIR)
    print(f"\nSaved {len(plot_paths)} plots to {PLOTS_DIR}/")
    for path in plot_paths:
        print(f"  {path}")

    print("\n=== Average Metric Scores ===")
    averages = df[list(METRICS)].mean()
    for name, value in averages.items():
        suffix = " (lower is better)" if METRICS[name].get("lower_is_better") else ""
        print(f"{name}: {value:.4f}{suffix}")


if __name__ == "__main__":
    asyncio.run(main())
