import logging
from pathlib import Path

import pymupdf as fitz


logger = logging.getLogger(__name__)


MINIMUM_PAGE_CHARACTERS = 30


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    pages = []

    with fitz.open(path) as pdf:

        logger.info(
            "Opened PDF: %s | pages=%d",
            path.name,
            len(pdf),
        )

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "character_count": len(text),
                }
            )

    logger.info(
        "Text extraction completed: %s",
        path.name,
    )

    return pages


def is_scanned_pdf(
    pages: list[dict],
    minimum_characters: int = MINIMUM_PAGE_CHARACTERS,
) -> bool:
    """
    Determine whether a PDF likely requires OCR.
    """

    if not pages:
        return True

    pages_with_insufficient_text = sum(
        page["character_count"] < minimum_characters
        for page in pages
    )

    ratio = pages_with_insufficient_text / len(pages)

    logger.info(
        "Scanned PDF detection: %.2f%% pages "
        "have insufficient text.",
        ratio * 100,
    )

    return ratio >= 0.5