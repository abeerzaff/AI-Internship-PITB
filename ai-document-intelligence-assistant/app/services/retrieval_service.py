
from app.services.embedding_service import (
    generate_query_embedding,
)
from app.services.vector_service import search_chunks


def retrieve(
    query: str,
    document_id: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Embed a query and retrieve relevant chunks from one document."""

    query_embedding = generate_query_embedding(query)

    return search_chunks(
        query_embedding,
        top_k=top_k,
        document_id=document_id,
    )