import asyncio
from pathlib import Path

import pandas as pd
import streamlit as st

from db.chroma import DB
from document_loader import DEFAULT_DOCUMENTS_DIR, load_chunks
from embeddings import create_embeddings
from models.model import EmbeddModel

UPLOAD_DIR = DEFAULT_DOCUMENTS_DIR / "uploads"
DEFAULT_METRICS = ["answer_relevancy", "faithfulness", "response_groundedness"]
EXTRA_METRICS_WITH_REFERENCE = ["answer_correctness", "semantic_similarity"]


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_model():
    return EmbeddModel().load_model()


@st.cache_resource
def get_embeddings():
    return create_embeddings(get_model())


@st.cache_resource
def get_database():
    return DB(model=get_model())


def get_evaluation():
    import eval as evaluation

    return evaluation


def save_uploads(files):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    for uploaded in files or []:
        name = Path(uploaded.name).name
        suffix = Path(name).suffix.lower()

        if suffix not in {".pdf", ".txt"}:
            continue

        path = UPLOAD_DIR / name
        path.write_bytes(uploaded.getbuffer())
        saved.append(path)

    return saved


def metrics_for_reference(metric_inputs, has_reference):
    """Hide reference-based metrics until a reference answer is provided."""
    if has_reference:
        return list(metric_inputs)

    return [name for name, fields in metric_inputs.items() if "reference" not in fields]


def rebuild_index(chunk_size, chunk_overlap):
    chunks = load_chunks(
        DEFAULT_DOCUMENTS_DIR,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    db = get_database()
    db.replace_documents(chunks)
    sources = sorted({chunk["metadata"]["source"] for chunk in chunks})
    st.session_state["index_stats"] = {"chunks": len(chunks), "sources": sources}


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
        raise RuntimeError("No documents are indexed. Add files and rebuild the index first.")

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

    if st.sidebar.button("Save and rebuild index", type="primary"):
        try:
            saved = save_uploads(uploaded_files)
            rebuild_index(int(chunk_size), int(chunk_overlap))
        except Exception as error:
            st.sidebar.error(f"Could not rebuild the index: {error}")
        else:
            if saved:
                st.sidebar.success(f"Saved {len(saved)} uploaded file(s).")
            st.sidebar.success("Index rebuilt.")

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
        st.info("Add documents in the sidebar and rebuild the index before asking a question.")

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


def main():
    st.set_page_config(page_title="RAGAS Evaluation", layout="wide")
    st.title("RAGAS Evaluation")
    st.caption(
        "Upload documents, ask a question, and inspect the retrieved context and RAGAS scores."
    )

    render_sidebar()
    render_evaluate_tab()


if __name__ == "__main__":
    main()
