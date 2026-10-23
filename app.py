import asyncio
import json
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from db.chroma import DB
from document_loader import load_chunks_from_uploads
from embeddings import create_embeddings
from models.model import EmbeddModel

DEFAULT_METRICS = ["answer_relevancy", "faithfulness", "response_groundedness"]
EXTRA_METRICS_WITH_REFERENCE = ["answer_correctness", "semantic_similarity"]
JSON_TEMPLATE = [
    {
        "question": "What is the main finding?",
        "reference": "The main finding is ...",
    },
    {
        "question": "Which method was used?",
        "reference": "The method was ...",
    },
]


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_model():
    return EmbeddModel().load_model()


@st.cache_resource
def get_embeddings():
    return create_embeddings(get_model())


def get_database():
    """Return the in-memory database for this Streamlit session."""
    database = st.session_state.get("database")

    if database is None:
        database = DB.in_memory(
            get_model(),
            collection_name=f"session_{uuid.uuid4().hex}",
        )
        st.session_state["database"] = database

    return database


def get_evaluation():
    import eval as evaluation

    return evaluation


def metrics_for_reference(metric_inputs, has_reference):
    """Hide reference-based metrics until a reference answer is provided."""
    if has_reference:
        return list(metric_inputs)

    return [name for name, fields in metric_inputs.items() if "reference" not in fields]


def build_index(files, chunk_size, chunk_overlap):
    """Index uploaded files in memory without writing them to disk."""
    chunks = load_chunks_from_uploads(
        files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    database = get_database()
    database.replace_documents(chunks)
    sources = sorted({chunk["metadata"]["source"] for chunk in chunks})
    st.session_state["index_stats"] = {"chunks": len(chunks), "sources": sources}
    st.session_state["eval_cache"] = {"questions": {}}
    st.session_state.pop("last_result", None)
    st.session_state.pop("batch_result", None)

    return st.session_state["index_stats"]


def _clean_cell(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def dataset_from_editor(frame):
    """Convert editable rows into questions, skipping completely blank rows."""
    dataset = []

    for row_number, row in frame.iterrows():
        question = _clean_cell(row.get("question"))
        reference = _clean_cell(row.get("reference"))

        if not question and not reference:
            continue
        if not question:
            raise ValueError(f"Row {row_number + 1} needs a question.")

        dataset.append({"question": question, "reference": reference})

    return dataset


def parse_batch_json(payload):
    """Validate a batch JSON file and return questions in the evaluation format."""
    if hasattr(payload, "getvalue"):
        payload = payload.getvalue()

    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Batch JSON must be UTF-8 encoded.") from error

    try:
        items = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"Batch JSON is not valid JSON: {error.msg}.") from error

    if not isinstance(items, list) or not items:
        raise ValueError("Batch JSON must be a non-empty list of question objects.")

    dataset = []

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Batch item {index} must be an object.")

        question = item.get("question")
        reference = item.get("reference") or ""

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Batch item {index} needs a non-empty 'question' string.")
        if not isinstance(reference, str):
            raise ValueError(f"Batch item {index} has a non-string 'reference'.")

        dataset.append({"question": question.strip(), "reference": reference.strip()})

    return dataset


def has_references(dataset):
    return bool(dataset) and all(item["reference"] for item in dataset)


def run_batch_evaluation(dataset, metric_names, cache=None):
    evaluation = get_evaluation()
    database = get_database()

    if database.num_docs == 0:
        raise RuntimeError("No documents are indexed. Upload documents and build the index first.")
    if not dataset:
        raise ValueError("Add at least one question.")
    if not metric_names:
        raise ValueError("Select at least one metric.")

    all_metrics = evaluation.build_metrics(evaluation.ragas_llm, get_embeddings())
    metrics = {name: all_metrics[name] for name in metric_names}

    if cache is None:
        cache = st.session_state.setdefault("eval_cache", {"questions": {}})

    questions = [item["question"] for item in dataset]

    evaluation.prune_cache(cache, questions)
    df = asyncio.run(
        evaluation.evaluate_dataset(
            dataset,
            database,
            metrics,
            cache,
            save=lambda _: None,
        )
    )
    csv_data = df.to_csv(index=False).encode("utf-8")

    with tempfile.TemporaryDirectory() as output_dir:
        plot_paths = evaluation.generate_plots(df, metrics, output_dir)
        plots = [{"name": Path(path).name, "data": Path(path).read_bytes()} for path in plot_paths]

    return {"df": df, "csv": csv_data, "plots": plots, "metrics": list(metrics)}


async def evaluate_question(question, reference, metric_names):
    evaluation = get_evaluation()
    missing_reference = [
        name
        for name in metric_names
        if "reference" in evaluation.METRIC_INPUTS[name] and not reference.strip()
    ]

    if missing_reference:
        raise ValueError(
            "Reference-based metrics need a reference answer: " + ", ".join(missing_reference)
        )

    contexts = evaluation.retrieve_contexts(get_database(), question)

    if not contexts:
        raise RuntimeError("No documents are indexed. Upload documents and build the index first.")

    answer = await evaluation.generate_answer("\n\n".join(contexts), question)
    sample = {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": reference,
    }
    metrics = evaluation.build_metrics(evaluation.ragas_llm, get_embeddings())
    scores = {}

    for name in metric_names:
        scores[name] = await evaluation.score_metric(metrics[name], sample)

    return {
        "question": question,
        "reference": reference,
        "answer": answer,
        "contexts": contexts,
        "scores": scores,
    }


def render_sidebar():
    st.sidebar.header("Knowledge base")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="document_uploads",
    )
    chunk_size = st.sidebar.number_input(
        "Chunk size (characters)",
        min_value=100,
        max_value=4000,
        value=800,
        step=100,
    )
    chunk_overlap = st.sidebar.number_input(
        "Chunk overlap (characters)",
        min_value=0,
        max_value=1000,
        value=150,
        step=50,
    )

    st.sidebar.caption("Uploaded files stay in memory and are cleared when the session ends.")

    if st.sidebar.button("Build index", type="primary"):
        if not uploaded_files:
            st.sidebar.warning("Upload at least one PDF or text file first.")
        else:
            try:
                with st.sidebar.spinner("Indexing documents..."):
                    stats = build_index(
                        uploaded_files,
                        int(chunk_size),
                        int(chunk_overlap),
                    )
            except Exception as error:
                st.sidebar.error(f"Could not build the index: {error}")
            else:
                st.sidebar.success(
                    f"Indexed {stats['chunks']} chunks from {len(stats['sources'])} file(s)."
                )

    stats = st.session_state.get("index_stats")

    if stats:
        st.sidebar.metric("Indexed chunks", stats["chunks"])

        with st.sidebar.expander("Indexed sources"):
            for source in stats["sources"]:
                st.write(source)


def render_result():
    result = st.session_state.get("last_result")

    if not result:
        return

    evaluation = get_evaluation()
    st.divider()
    st.subheader("Answer")
    st.markdown(result["answer"])

    st.subheader("RAGAS scores")
    rows = [
        {
            "Metric": name.replace("_", " "),
            "Score": score,
            "Better": "Lower" if name in evaluation.LOWER_IS_BETTER else "Higher",
        }
        for name, score in result["scores"].items()
    ]
    scores = pd.DataFrame(rows)

    if not scores.empty:
        st.dataframe(scores, hide_index=True)
        st.bar_chart(scores.set_index("Metric")[["Score"]])

    st.subheader("Retrieved context")

    for index, context in enumerate(result["contexts"], start=1):
        with st.expander(f"Context {index}"):
            st.write(context)


def render_evaluate_tab():
    try:
        evaluation = get_evaluation()
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"Could not load the evaluation setup: {error}")
        return

    if not st.session_state.get("index_stats"):
        st.info("Upload documents in the sidebar and build the index before asking a question.")

    question = st.text_input("Ask a question about your documents")
    reference = st.text_area(
        "Reference answer (optional)",
        height=100,
        help="Required for answer correctness, context precision/recall, and similar metrics.",
    )
    has_reference = bool(reference.strip())
    available_metrics = metrics_for_reference(evaluation.METRIC_INPUTS, has_reference)
    defaults = [name for name in DEFAULT_METRICS if name in available_metrics]

    if has_reference:
        defaults += [name for name in EXTRA_METRICS_WITH_REFERENCE if name in available_metrics]

    metric_names = st.multiselect(
        "Metrics",
        options=available_metrics,
        default=defaults,
    )

    if st.button("Ask and evaluate", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
            return
        if not metric_names:
            st.warning("Select at least one metric.")
            return

        with st.spinner("Retrieving context, generating an answer, and scoring..."):
            try:
                st.session_state["last_result"] = asyncio.run(
                    evaluate_question(question, reference, metric_names)
                )
            except Exception as error:
                st.error(f"Evaluation failed: {error}")

    render_result()


def _batch_input():
    mode = st.radio(
        "How do you want to provide the batch?",
        ["Enter questions", "Upload JSON"],
        horizontal=True,
    )

    if mode == "Enter questions":
        rows = st.data_editor(
            pd.DataFrame(
                {
                    "question": ["", "", ""],
                    "reference": ["", "", ""],
                }
            ),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "question": st.column_config.TextColumn("Question", required=False),
                "reference": st.column_config.TextColumn("Reference answer", required=False),
            },
            key="batch_rows",
        )

        return dataset_from_editor(rows)

    uploaded = st.file_uploader("Upload batch JSON", type=["json"], key="batch_json")

    with st.expander("JSON format"):
        st.code(json.dumps(JSON_TEMPLATE, indent=2), language="json")
        st.download_button(
            "Download JSON template",
            data=json.dumps(JSON_TEMPLATE, indent=2).encode("utf-8"),
            file_name="questions.json",
            mime="application/json",
        )

    if uploaded is None:
        return []

    return parse_batch_json(uploaded)


def render_batch_result():
    result = st.session_state.get("batch_result")

    if not result:
        return

    st.divider()
    st.subheader("Results")
    st.dataframe(result["df"], hide_index=True)
    st.download_button(
        "Download CSV",
        data=result["csv"],
        file_name="ragas_results.csv",
        mime="text/csv",
        key="download-results-csv",
    )

    st.subheader("Plots")

    for plot in result["plots"]:
        st.image(plot["data"], caption=plot["name"])
        st.download_button(
            f"Download {plot['name']}",
            data=plot["data"],
            file_name=plot["name"],
            mime="image/png",
            key=f"download-{plot['name']}",
        )


def render_batch_tab():
    try:
        evaluation = get_evaluation()
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"Could not load the evaluation setup: {error}")
        return

    if not st.session_state.get("index_stats"):
        st.info("Upload documents in the sidebar and build the index before running a batch.")

    st.caption(
        "Provide questions and reference answers directly, or upload a JSON list. "
        "Questions are evaluated against the documents in this session. "
        "Reference-based metrics appear only when every question has a reference."
    )

    try:
        dataset = _batch_input()
    except ValueError as error:
        st.error(str(error))
        dataset = []

    if dataset:
        with st.expander("Batch preview"):
            st.dataframe(pd.DataFrame(dataset), hide_index=True)

    has_reference = has_references(dataset)
    available_metrics = metrics_for_reference(evaluation.METRIC_INPUTS, has_reference)
    defaults = (
        list(available_metrics)
        if has_reference
        else [name for name in DEFAULT_METRICS if name in available_metrics]
    )
    metric_names = st.multiselect(
        "Metrics",
        options=available_metrics,
        default=defaults,
        key=f"batch_metrics_{'reference' if has_reference else 'no_reference'}",
    )
    st.caption(f"{len(dataset)} questions x {len(metric_names)} metrics per run.")

    if st.button("Run batch evaluation", type="primary"):
        if not dataset:
            st.warning("Add at least one question or upload a JSON file.")
            return
        if not metric_names:
            st.warning("Select at least one metric.")
            return

        with st.spinner("Answering questions, scoring metrics, and generating plots..."):
            try:
                st.session_state["batch_result"] = run_batch_evaluation(dataset, metric_names)
            except Exception as error:
                st.error(f"Batch evaluation failed: {error}")

    render_batch_result()


def main():
    st.set_page_config(page_title="RAGAS Evaluation", layout="wide")
    st.title("RAGAS Evaluation")
    st.caption(
        "Upload documents to build a private, in-memory index, then ask a question "
        "or evaluate a batch you provide."
    )

    render_sidebar()

    ask_tab, batch_tab = st.tabs(["Ask a question", "Batch evaluation"])

    with ask_tab:
        render_evaluate_tab()

    with batch_tab:
        render_batch_tab()


if __name__ == "__main__":
    main()
