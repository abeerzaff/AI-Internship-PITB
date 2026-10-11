import logging

from app.core.config import settings
from app.services.context_service import build_context
from app.services.fallback_service import fallback_response
from app.services.llm_service import generate_answer
from app.services.prompt_service import NOT_IN_DOCUMENT, build_grounded_prompt
from app.services.relevance_service import filter_relevant_results
from app.services.retrieval_service import retrieve
from app.services.source_service import build_sources
from app.services.vector_service import get_overview_chunks ,count_chunks
from app.services.vector_service import validate_active_document
import re

logger = logging.getLogger(__name__)

OVERVIEW_PATTERN = re.compile(
    r"\b(objective|purpose|summar\w*|overview|goals?|aims?"
    r"|main (idea|topic)|what is this"
    r"|(explain|describe) (this|the) (document|report|file)"
    r"|this (document|report|file))\b",
    re.IGNORECASE,
)
GENERAL_NOTICE = (
    "Note: this is not from the uploaded document. "
    "Answer from general knowledge:\n\n"
)


def is_overview_question(question: str) -> bool:
    return bool(OVERVIEW_PATTERN.search(question))


def _not_in_document(question: str) -> dict:
    """Fallback, or a clearly labelled general answer if enabled."""
    if not settings.allow_general_knowledge:
        return fallback_response()

    prompt = (
        "Answer this general-knowledge question briefly and factually. "
        "If you are not sure, say so.\n\nQUESTION: " + question
    )
    answer = generate_answer(prompt)
    return {
        "answer": GENERAL_NOTICE + answer,
        "grounded": False,
        "sources": [],
    }


def answer_question(question:str ,document_id: str| None = None) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if count_chunks() == 0:
        raise ValueError("Upload a document first.")

    logger.info("Processing question: %s", question)
    document_id = validate_active_document(document_id)

    results = retrieve(question,  document_id,)
    relevant_results = filter_relevant_results(results)

    if is_overview_question(question):
        overview = get_overview_chunks(document_id)
        seen = {r["text"] for r in relevant_results}
        relevant_results = [
        o for o in overview if o["text"] not in seen
    ] + relevant_results

    if not relevant_results:
        logger.info("No relevant context found.")
        return _not_in_document(question)

    context = build_context(relevant_results)
    prompt = build_grounded_prompt(question=question, context=context)
    answer = generate_answer(prompt)

    if answer.strip().startswith(NOT_IN_DOCUMENT):
        logger.info("Model reported the answer is not in the document.")
        return _not_in_document(question)

    return {
        "answer": answer,
        "grounded": True,
        "sources": build_sources(relevant_results,answer),
    }
