import hashlib
import logging
from pathlib import Path
from typing import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
SUPPORTED_SUFFIXES = {".pdf", ".txt"}


class DocumentChunk(TypedDict):
    id: str
    text: str
    metadata: dict[str, str | int]


def load_chunks(
    documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Load every .txt and .pdf file under a directory and split it into chunks."""
    directory = Path(documents_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Documents path is not a directory: {directory}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")

    files = _find_source_files(directory)

    if not files:
        raise ValueError(f"No .pdf or .txt files found in {directory}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    chunks: list[DocumentChunk] = []

    for path in files:
        source = path.relative_to(directory).as_posix()

        if path.suffix.lower() == ".txt":
            text = _read_text_file(path)
            chunks.extend(_split_text(text, source, page=None, splitter=splitter))
            continue

        for page_number, text in _read_pdf_pages(path):
            chunks.extend(_split_text(text, source, page=page_number, splitter=splitter))

    if not chunks:
        raise ValueError(f"No readable text found in {directory}")

    return chunks


def _find_source_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not _is_hidden(path, directory)
    )


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("File is not valid UTF-8; replacing invalid characters: %s", path)
        return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(path)
    except Exception:
        logger.warning("Skipping unreadable PDF: %s", path, exc_info=True)
        return []

    if reader.is_encrypted and not reader.decrypt(""):
        logger.warning("Skipping encrypted PDF: %s", path)
        return []

    pages: list[tuple[int, str]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            logger.warning(
                "Could not extract page %s from %s",
                page_number,
                path,
                exc_info=True,
            )
            continue

        if text.strip():
            pages.append((page_number, text))

    return pages


def _split_text(
    text: str,
    source: str,
    page: int | None,
    splitter: RecursiveCharacterTextSplitter,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for piece in splitter.split_text(text):
        piece = piece.strip()

        if not piece:
            continue

        metadata: dict[str, str | int] = {
            "source": source,
            "chunk_index": chunk_index,
        }

        if page is not None:
            metadata["page"] = page

        identifier = f"{source}:{page or 0}:{chunk_index}"
        chunks.append(
            {
                "id": hashlib.sha256(identifier.encode("utf-8")).hexdigest(),
                "text": piece,
                "metadata": metadata,
            }
        )
        chunk_index += 1

    return chunks
