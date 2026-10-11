
import logging
import re

from app.services.llm_service import generate_answer

logger = logging.getLogger(__name__)

NO_ACTIONS = "No explicit action items were identified."
LIST_PREFIX = re.compile(r"^\s*(\d+[.)]|[-*•])\s+")


def extract_action_items(context: str) -> list[str]:

    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    prompt = f"""
Identify action items, recommendations and next steps that the document
explicitly states.

Rules:
- Use ONLY information explicitly stated in the document.
- Return ONLY a numbered list (1., 2., 3., ...), one sentence per item.
- No introduction and no headings.
- If the document states none, reply with exactly:
{NO_ACTIONS}

DOCUMENT:
{context}

ACTION ITEMS:
"""

    response = generate_answer(prompt).strip()

    if not response:
        return []

    if response.lower().rstrip(".") == NO_ACTIONS.lower().rstrip("."):
        logger.info("No explicit action items found.")
        return []

    actions = []

    for line in response.splitlines():
        cleaned = LIST_PREFIX.sub(
            "", line
        ).replace("**", "").strip()

        if not cleaned:
            continue

        if cleaned.lower().rstrip(".") == NO_ACTIONS.lower().rstrip("."):
            continue

        actions.append(cleaned)

    if not actions:
        logger.info("No valid action items found.")

    logger.info("Extracted %d action items.", len(actions))

    return actions
