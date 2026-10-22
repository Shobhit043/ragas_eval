# RAGAS Evaluation Project

A small, end-to-end example of evaluating a Retrieval-Augmented Generation (RAG) pipeline with [RAGAS](https://docs.ragas.io/).

It indexes a set of short documents in ChromaDB, answers questions using a hosted LLM with retrieved context, and scores the answers with four RAGAS metrics.

## What it does

```
documents ──> embeddings ──> ChromaDB
                                 │
question ──> top-5 retrieval ────┘
                    │
                    ▼
           LLM answer (NVIDIA NIM)
                    │
                    ▼
       RAGAS metrics ──> ragas_results.csv
```

For every question the project:

1. Embeds the documents with `all-MiniLM-L6-v2`.
2. Stores them in a local ChromaDB collection.
3. Retrieves the 5 most similar chunks.
4. Generates an answer from the retrieved context.
5. Scores the result and writes a CSV.

## Requirements

- Python **3.13+**
- An [NVIDIA API key](https://build.nvidia.com/)
- ~1 GB of free disk space (dependencies) plus ~90 MB for the embedding model

## Setup

Clone the repository, then install the dependencies.

Using `uv` (recommended — resolves from `uv.lock`):

```bash
uv sync
```

Using `pip`:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Create your `.env` file from the template and add your key:

```bash
cp .env.example .env
```

```dotenv
NVIDIA_API_KEY=your-nvidia-api-key-here
```

## Run

```bash
uv run eval.py
```

or, with the venv activated:

```bash
python eval.py
```

The first run downloads the embedding model to `models/checkpoints/` (about 90 MB) and is slower than subsequent runs.

## Output

- `ragas_results.csv` — one row per question with the answer, retrieved contexts, reference answer, and metric scores.
- `eval_cache.json` — intermediate cache so a run can resume without paying for the same LLM calls twice.
- `db/chroma_db/` — the persistent ChromaDB store.

All three are environment-specific/generated and are ignored by git.

## Metrics

The evaluation in `eval.py` uses the RAGAS `collections` metrics:

| Metric | What it checks |
| --- | --- |
| `answer_relevancy` | Does the answer actually address the question? |
| `faithfulness` | Is the answer supported by the retrieved context? |
| `context_precision` | Is the retrieved context relevant? |
| `context_recall` | Does the context cover what the reference answer needs? |

Each score is stored as a float between 0 and 1.

## Project layout

```
.
├── data.py                 # 20 documents, 30 questions, 30 reference answers
├── eval.py                 # main pipeline + RAGAS evaluation
├── llm.py                  # NVIDIA client, retry logic, RAGAS LLM wrapper
├── embeddings.py           # Ragas adapter around the shared SentenceTransformer
├── cache.py                # JSON cache load/save helpers
├── db/
│   ├── chroma.py           # ChromaDB wrapper (add, search, clear)
│   └── embedding_func.py   # Chroma embedding function
├── models/
│   └── model.py            # Downloads/loads all-MiniLM-L6-v2
├── pyproject.toml          # Project metadata and direct dependencies
├── requirements.txt        # Same dependencies for pip users
└── uv.lock                 # Fully resolved dependency lockfile
```

## How it works

**Embeddings are shared.** `models/model.py` loads one `SentenceTransformer`, and both ChromaDB (`db/embedding_func.py`) and RAGAS (`embeddings.py`) use it. This keeps retrieval and evaluation embeddings consistent.

**The LLM client is async and retries.** `llm.py` talks to the NVIDIA OpenAI-compatible endpoint and retries rate-limit, timeout, connection, and 5xx errors with exponential backoff, honoring the `retry-after` header.

**Caching is per question and per metric.** `eval.py` stores the generated answer, retrieved contexts, and each metric score in `eval_cache.json`. To force a full re-evaluation, delete that file.

**The database is rebuilt on each run.** `eval.py` clears the collection and re-adds the documents, so the CSV always reflects the current `data.py`.

## Troubleshooting

- **`ValueError: NVIDIA_API_KEY is missing from your .env file`** — create `.env` from `.env.example` and add your key.
- **First run seems stuck** — it is downloading the embedding model. Later runs load it from `models/checkpoints/`.
- **Frequent rate limits** — lower `max_attempts` or increase `base_wait` in `llm.py:53`, or use a different model.
- **Stale results** — delete `eval_cache.json` (and optionally `ragas_results.csv`) and run again.

## Notes

- `langchain-community` is pinned to `0.3.31` because RAGAS 0.4.3 imports a module that newer versions removed.
- `models/checkpoints/`, `.venv/`, `.env`, and the generated evaluation files are git-ignored.
