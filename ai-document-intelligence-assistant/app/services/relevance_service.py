
from app.core.config import settings


def distance_to_similarity(distance: float) -> float:
    """Convert cosine distance to a similarity-like score."""

    return 1.0 - distance


def filter_relevant_results(
    results: list[dict],
) -> list[dict]:
    """Keep only results above the configured threshold."""

    relevant = []

    for result in results:

        similarity = distance_to_similarity(
            result["distance"]
        )

        result["similarity"] = similarity

        if similarity >= settings.similarity_threshold:
            relevant.append(result)

    return relevant