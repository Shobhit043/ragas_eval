import hashlib
import io
import logging
from pathlib import Path
from typing import BinaryIO, TypedDict

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

    _validate_chunking(chunk_size, chunk_overlap)
    files = _find_source_files(directory)

    if not files:
        raise ValueError(f"No .pdf or .txt files found in {directory}")

    splitter = _create_splitter(chunk_size, chunk_overlap)
    chunks: list[DocumentChunk] = []

    for path in files:
        source = path.relative_to(directory).as_posix()

        if path.suffix.lower() == ".txt":
            text = _read_text_file(path)
            chunks.extend(_split_text(text, source, page=None, splitter=splitter))
            continue

        for page_number, text in _read_pdf_pages(path, source):
            chunks.extend(_split_text(text, source, page=page_number, splitter=splitter))

    if not chunks:
        raise ValueError(f"No readable text found in {directory}")

    return chunks


def load_chunks_from_uploads(
    files,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Load uploaded .txt and .pdf files without writing them to disk."""
    _validate_chunking(chunk_size, chunk_overlap)

    if not files:
        raise ValueError("Upload at least one .pdf or .txt file.")

    splitter = _create_splitter(chunk_size, chunk_overlap)
    chunks: list[DocumentChunk] = []

    for uploaded in files:
        source = Path(uploaded.name).name
        suffix = Path(source).suffix.lower()

        if suffix not in SUPPORTED_SUFFIXES:
            continue

        data = uploaded.getvalue()
        source_id = f"{source}:{hashlib.sha256(data).hexdigest()[:16]}"

        if suffix == ".txt":
            text = _decode_text(data, source)
            chunks.extend(
                _split_text(text, source, page=None, splitter=splitter, source_id=source_id)
            )
            continue

        for page_number, text in _read_pdf_pages(io.BytesIO(data), source):
            chunks.extend(
                _split_text(
                    text,
                    source,
                    page=page_number,
                    splitter=splitter,
                    source_id=source_id,
                )
            )

    if not chunks:
        raise ValueError("No readable text found in the uploaded files.")

    return chunks


def _validate_chunking(chunk_size: int, chunk_overlap: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")


def _create_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )


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
    return _decode_text(path.read_bytes(), str(path))


def _decode_text(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("File is not valid UTF-8; replacing invalid characters: %s", label)
        return data.decode("utf-8", errors="replace")


def _read_pdf_pages(source: Path | BinaryIO, label: str) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(source)
    except Exception:
        logger.warning("Skipping unreadable PDF: %s", label, exc_info=True)
        return []

    if reader.is_encrypted and not reader.decrypt(""):
        logger.warning("Skipping encrypted PDF: %s", label)
        return []

    pages: list[tuple[int, str]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            logger.warning(
                "Could not extract page %s from %s",
                page_number,
                label,
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
    source_id: str | None = None,
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

        identifier = f"{source_id or source}:{page or 0}:{chunk_index}"
        chunks.append(
            {
                "id": hashlib.sha256(identifier.encode("utf-8")).hexdigest(),
                "text": piece,
                "metadata": metadata,
            }
        )
        chunk_index += 1

    return chunks
