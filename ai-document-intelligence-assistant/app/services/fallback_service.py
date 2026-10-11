
import logging

logger = logging.getLogger(__name__)


FALLBACK_MESSAGE = (
    "I could not find enough relevant information in "
    "the provided document to answer this question."
)


def fallback_response() -> dict:

    logger.info("Fallback response returned.")

    return {
        "answer": FALLBACK_MESSAGE,
        "grounded": False,
        "sources": [],
    }