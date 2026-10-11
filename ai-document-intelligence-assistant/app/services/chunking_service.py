import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of characters shared between chunks.

    Returns:
        List of text chunks.
    """

    if not text or not text.strip():
        return []

    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = (
        chunk_overlap
        if chunk_overlap is not None
        else settings.chunk_overlap
    )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text = text.strip()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - chunk_overlap

    logger.info(
        "Created %d chunks | size=%d | overlap=%d",
        len(chunks),
        chunk_size,
        chunk_overlap,
    )

    return chunks

def chunk_document(
    pages: list[dict],
    document_id: str,
    source: str,
) -> list[dict]:
    """
    Convert page-level document data into RAG chunks
    while preserving source metadata.
    """

    chunks = []

    chunk_counter = 0

    for page in pages:

        page_number = page["page_number"]
        page_text = page.get("text", "")

        page_chunks = chunk_text(page_text)

        for page_chunk_index, text in enumerate(
            page_chunks,
            start=1,
        ):
            chunk_counter += 1

            chunks.append(
                {
                    "document_id": document_id,
                    "source": source,
                    "page": page_number,
                    "chunk_id": f"{document_id}_chunk_{chunk_counter}",
                    "text": text,
                }
            )

    logger.info(
        "Document chunking completed | document=%s | chunks=%d",
        document_id,
        len(chunks),
    )

    return chunks