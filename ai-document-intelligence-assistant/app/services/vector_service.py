
import logging

import chromadb

from app.core.config import settings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "document_chunks"

_client = None
_collection = None


def get_chroma_client():
    """Create or reuse the persistent ChromaDB client."""

    global _client

    if _client is None:
        settings.chroma_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Initializing ChromaDB: %s",
            settings.chroma_directory,
        )

        _client = chromadb.PersistentClient(
            path=str(settings.chroma_directory)
        )

    return _client


def get_collection():
    """Get or create the document chunk collection."""

    global _collection

    if _collection is None:
        client = get_chroma_client()

        _collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={
                "description": "Document chunks for RAG retrieval",
                "hnsw:space": "cosine",
                
            },
        )

    return _collection


def add_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Store chunks, embeddings and metadata."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunks and embeddings must have the same length."
        )

    if not chunks:
        return

    collection = get_collection()

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "document_id": chunk["document_id"],
            "source": chunk["source"],
            "page": chunk["page"],
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(
        "Indexed %d chunks into ChromaDB.",
        len(chunks),
    )


def count_chunks() -> int:
    """Return the number of stored chunks."""

    return get_collection().count()


def search_chunks(
    query_embedding: list[float],
    top_k: int | None = None,
    document_id: str | None = None,
) -> list[dict]:
    """Retrieve semantically similar document chunks."""

    document_id = validate_active_document(document_id)

    top_k = settings.top_k if top_k is None else top_k

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    collection = get_collection()

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    query_kwargs["where"] = {
    "document_id": document_id
}

    results = collection.query(**query_kwargs)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
                "similarity": 1 - distance,
            }
        )

    logger.info(
        "Retrieved %d chunks for document_id=%s.",
        len(retrieved),
        document_id,
    )

    return retrieved

def get_overview_chunks(
    document_id: str,
    head: int = 3,
    tail: int = 2,
) -> list[dict]:

    document_id = validate_active_document(document_id)
    """Return the first and last chunks of the document (overview context)."""
    get_kwargs = {"include": ["documents", "metadatas"]}
    if document_id:
        get_kwargs["where"] = {"document_id": document_id}
    data = get_collection().get(**get_kwargs)
    

    rows = sorted(
        zip(data["ids"], data["documents"], data["metadatas"]),
        key=lambda row: int(row[0].rsplit("_", 1)[-1]),
    )

    selected = rows[:head] + [r for r in rows[-tail:] if r not in rows[:head]]

    return [
        {"text": text, "metadata": meta, "distance": None, "similarity": None}
        for _, text, meta in selected
    ]



def get_all_chunks(document_id: str | None = None) -> list[dict]:
   
    """Return every chunk of the document in reading order."""
    document_id = validate_active_document(document_id)
    get_kwargs = {"include": ["documents", "metadatas"]}
    if document_id:
        get_kwargs["where"] = {"document_id": document_id}
    data = get_collection().get(**get_kwargs)

    rows = sorted(
        zip(data["ids"], data["documents"], data["metadatas"]),
        key=lambda row: int(row[0].rsplit("_", 1)[-1]),
    )
    return [
        {"text": text, "metadata": meta, "distance": None, "similarity": None}
        for _, text, meta in rows
    ]


def clear_collection() -> None:
    """Remove all indexed chunks so only the next upload is searchable."""
    collection = get_collection()
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)
        logger.info("Cleared %d old chunks.", len(ids))


def delete_all_except(document_id: str) -> None:
    """Remove every chunk that does NOT belong to this document."""
    get_collection().delete(where={"document_id": {"$ne": document_id}})
    logger.info("Removed chunks of older documents.")



def get_indexed_documents() -> list[dict]:
    """List unique documents currently stored in ChromaDB."""

    data = get_collection().get(include=["metadatas"])

    documents = {}

    for metadata in data["metadatas"] or []:
        if not metadata:
            continue

        document_id = metadata.get("document_id")

        if document_id:
            documents[document_id] = {
                "document_id": document_id,
                "source": metadata.get("source"),
            }

    return list(documents.values())


def validate_active_document(document_id: str | None) -> str:
    """Require an existing document ID."""

    if not isinstance(document_id, str):
        raise ValueError("Please upload or select a PDF first.")

    document_id = document_id.strip()

    if not document_id:
        raise ValueError("A document ID is required.")

    result = get_collection().get(
        where={"document_id": document_id},
        limit=1,
    )

    if not result["ids"]:
        raise ValueError(
            "This PDF is no longer active. "
            "Please upload the document again."
        )

    return document_id



def delete_document_chunks(document_id: str) -> None:
    """Delete chunks for one specified document."""

    if not document_id:
        raise ValueError("Document ID is required.")

    get_collection().delete(
        where={"document_id": document_id}
    )

