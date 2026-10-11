
import logging

from app.services.llm_service import generate_answer

logger = logging.getLogger(__name__)


def summarize_document(context: str) -> str:

    if not context.strip():
        raise ValueError("Context cannot be empty.")

    prompt = f"""
You are summarizing a public-sector document.

Use ONLY the provided document content.

Provide a concise professional summary.

DOCUMENT:
{context}

SUMMARY:
"""

    summary = generate_answer(prompt)

    logger.info("Document summary generated.")

    return summary