import logging
import re

from app.services.llm_service import generate_answer

logger = logging.getLogger(__name__)

NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s+(.*\S)\s*$")


def extract_key_points(context: str) -> list[str]:

    if not context.strip():
        raise ValueError("Context cannot be empty.")

    prompt = f"""
Extract 6 to 10 key points from the document.

Rules:
- Use ONLY the provided content.
- Return ONLY a numbered list (1., 2., 3., ...).
- One sentence per point. No introduction, no headings, no sub-bullets.

DOCUMENT:
{context}

KEY POINTS:
"""

    response = generate_answer(prompt)

    points = []

    for line in response.splitlines():
        match = NUMBERED_LINE.match(line)
        if match:
            points.append(match.group(1).replace("**", "").strip())

    # Fallback if the model ignored the numbered format
    if not points:
        points = [
            line.strip("-*• ").strip()
            for line in response.splitlines()
            if line.strip()
        ]

    logger.info(
        "Extracted %d key points.",
        len(points),
    )

    return points