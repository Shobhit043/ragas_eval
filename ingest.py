import argparse
import logging
from pathlib import Path

from db.chroma import DB
from document_loader import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DOCUMENTS_DIR,
    load_chunks,
)
from models.model import EmbeddModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load PDF and text files into ChromaDB for the RAG pipeline."
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=DEFAULT_DOCUMENTS_DIR,
        help="Directory to scan recursively for .pdf and .txt files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Characters shared between neighboring chunks.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing chunks instead of replacing the collection.",
    )
    parser.add_argument("--db-path", default="chroma_db", help="ChromaDB storage path.")
    parser.add_argument("--collection", default="data", help="ChromaDB collection name.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    chunks = load_chunks(
        args.documents_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    sources = {chunk["metadata"]["source"] for chunk in chunks}

    model = EmbeddModel().load_model()
    db = DB(db_path=args.db_path, collection_name=args.collection, model=model)

    if args.append:
        db.add_documents(chunks)
    else:
        db.replace_documents(chunks)

    action = "Added" if args.append else "Indexed"
    print(f"{action} {len(chunks)} chunks from {len(sources)} files.")
    print(f"Collection now contains {db.num_docs} chunks.")


if __name__ == "__main__":
    main()
