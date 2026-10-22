import asyncio

import pandas as pd

from cache import load_cache, save_cache
from data import docs, references, user_inputs
from db.chroma import DB
from embeddings import embeddings, model
from llm import generate_answer, ragas_llm, with_retries
from ragas.metrics.collections import (
    AnswerRelevancy,
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)

RESULTS_PATH = "ragas_results.csv"

dbClient = DB(model=model)

metric_objects = {
    "answer_relevancy": AnswerRelevancy(
        llm=ragas_llm,
        embeddings=embeddings,
    ),
    "faithfulness": Faithfulness(llm=ragas_llm),
    "context_precision": ContextPrecision(llm=ragas_llm),
    "context_recall": ContextRecall(llm=ragas_llm),
}


async def main():
    dbClient.clear()
    dbClient.addDocs(docs=docs)

    cache = load_cache()
    rows = []

    for index, question in enumerate(user_inputs):
        entry = cache.setdefault(question, {})
        cached_metrics = entry.setdefault("metrics", {})

        if "response" not in entry or "retrieved_contexts" not in entry:
            search_result = dbClient.semanticSearch(question)[0]

            if isinstance(search_result, str):
                context_chunks = [search_result]
            else:
                context_chunks = list(search_result)

            context_text = "\n\n".join(context_chunks)

            entry["retrieved_contexts"] = context_chunks
            entry["response"] = await generate_answer(context_text, question)
            save_cache(cache)

        sample = {
            "user_input": question,
            "response": entry["response"],
            "retrieved_contexts": entry["retrieved_contexts"],
            "reference": references[index],
        }

        print(f"\nEvaluating: {question}")

        metric_inputs = {
            "answer_relevancy": {
                "user_input": sample["user_input"],
                "response": sample["response"],
            },
            "faithfulness": {
                "user_input": sample["user_input"],
                "response": sample["response"],
                "retrieved_contexts": sample["retrieved_contexts"],
            },
            "context_precision": {
                "user_input": sample["user_input"],
                "reference": sample["reference"],
                "retrieved_contexts": sample["retrieved_contexts"],
            },
            "context_recall": {
                "user_input": sample["user_input"],
                "reference": sample["reference"],
                "retrieved_contexts": sample["retrieved_contexts"],
            },
        }

        for name, metric in metric_objects.items():
            if name in cached_metrics:
                print(f"  {name}: cached")
                continue

            result = await with_retries(
                metric.ascore,
                **metric_inputs[name],
            )
            cached_metrics[name] = float(result.value)
            save_cache(cache)
            print(f"  {name}: {cached_metrics[name]:.4f}")

        rows.append({
            "user_input": sample["user_input"],
            "response": sample["response"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "reference": sample["reference"],
            **cached_metrics,
        })

    df = pd.DataFrame(rows)

    print("\n=== RAGAS Evaluation Results ===")
    print(df.to_string(index=False))

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
