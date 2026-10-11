import logging

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

_model = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""

    global _model

    if _model is None:
        logger.info(
            "Loading embedding model: %s",
            settings.embedding_model,
        )

        _model = SentenceTransformer(
            settings.embedding_model
        )

        logger.info("Embedding model loaded successfully.")

    return _model


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding for one text."""

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    model = get_embedding_model()

    embedding = model.encode(
        text,
        convert_to_numpy=True,
    )

    return embedding.tolist()


def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """Generate embeddings for multiple texts."""

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError(
            "Embedding input contains empty text."
        )

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
    )

    return embeddings.tolist()

def generate_query_embedding(
    query: str,
) -> list[float]:
    """Generate an embedding for a user question."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    return generate_embedding(query)