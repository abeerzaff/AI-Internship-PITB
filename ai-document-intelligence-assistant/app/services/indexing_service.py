
import logging
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.services.chunking_service import chunk_document
from app.services.document_service import process_document
from app.services.embedding_service import generate_embeddings
from app.services.vector_service import (
    add_chunks,
    count_chunks,
    delete_all_except,
    delete_document_chunks,
    get_indexed_documents,
)

logger = logging.getLogger(__name__)


def cleanup_old_uploads(
    old_documents: list[dict],
    new_filename: str,
) -> None:
    """Remove old managed PDFs after replacement succeeds."""

    upload_dir = settings.upload_directory.resolve()

    for document in old_documents:
        filename = document.get("source")

        if not filename or filename == new_filename:
            continue

        # Never delete arbitrary paths.
        if Path(filename).name != filename:
            logger.warning("Skipping unsafe filename: %s", filename)
            continue

        old_path = upload_dir / filename

        # Do not follow symlinks or escape upload directory.
        if old_path.is_symlink():
            logger.warning("Skipping symlink: %s", old_path)
            continue

        if old_path.resolve().parent != upload_dir:
            logger.warning("Skipping unsafe path: %s", old_path)
            continue

        try:
            old_path.unlink(missing_ok=True)
            logger.info("Deleted old uploaded PDF: %s", filename)
        except OSError:
            logger.exception(
                "Could not delete old PDF: %s",
                filename,
            )


def index_document(file_path: str) -> dict:
    """Index the new PDF and replace the old searchable PDF."""

    path = Path(file_path).resolve()
    upload_dir = settings.upload_directory.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.parent != upload_dir:
        raise ValueError(
            "PDF must be inside the configured upload directory."
        )

    document_id = uuid4().hex

    # Remember old documents for later cleanup.
    old_documents = get_indexed_documents()

    logger.info("Indexing document: %s", path.name)

    # Extract text (OCR fallback handled internally).
    processed = process_document(str(path))

    # Chunk the pages.
    chunks = chunk_document(
        pages=processed["pages"],
        document_id=document_id,
        source=path.name,
    )

    if not chunks:
        raise ValueError("No text chunks were generated.")

    # Generate embeddings.
    embeddings = generate_embeddings(
        [chunk["text"] for chunk in chunks]
    )

    # Add new chunks without first deleting the old ones.
     
    
    # Step 1: Insert the new document chunks into ChromaDB.
    try:
        add_chunks(chunks, embeddings)

    except Exception:
        logger.exception("Failed to store new document chunks.")

        # Remove any chunks partially added for this document.
        try:
            delete_document_chunks(document_id)

        except Exception as cleanup_error:
            logger.exception(
                "Partial indexing cleanup failed for %s",
                document_id,
            )

            raise RuntimeError(
                "Indexing failed and partial chunks could not "
                "be cleaned up. Database recovery is required."
            ) from cleanup_error

        raise

    # Step 2: Diagnostic check after insertion.
    logger.info(
        "AFTER INSERT: %s",
        get_indexed_documents(),
    )

    logger.info(
        "Total chunks after insert: %d",
         count_chunks(),
    )

    # Step 3: TEMPORARILY disable deletion while debugging.
    # delete_all_except(document_id)


    # Remove old chunks after new chunks are indexed.
    # If this fails, preserve the new file for recovery.
    delete_all_except(document_id)

    remaining = get_indexed_documents()

    remaining_ids = {
        item["document_id"]
        for item in remaining
}

    if remaining_ids != {document_id}:
        raise RuntimeError(
        "Document replacement was not completed correctly. "
        "Check ChromaDB before retrying."
    )

    if any(
        d["document_id"] != document_id
        for d in remaining
    ):
        raise RuntimeError(
            "Old document chunks were not fully removed."
        )

    # Clean up only the previous indexed PDFs.
    cleanup_old_uploads(
        old_documents=old_documents,
        new_filename=path.name,
    )

    logger.info(
        "Indexed %d chunks for document %s",
        len(chunks),
        document_id,
    )

    return {
        "document_id": document_id,
        "source": path.name,
        "page_count": processed["page_count"],
        "chunk_count": len(chunks),
        "extraction_method": processed["extraction_method"],
    }
