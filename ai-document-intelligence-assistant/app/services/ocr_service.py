import logging
from io import BytesIO

import fitz
import pytesseract
from PIL import Image, ImageOps


logger = logging.getLogger(__name__)


def preprocess_image(
    image: Image.Image,
) -> Image.Image:
    """
    Preprocess a rendered PDF page before OCR.
    """

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Improve contrast
    image = ImageOps.autocontrast(image)

    return image


def ocr_pdf(
    file_path: str,
    page_numbers: list[int] | None = None,
) -> list[dict]:
    """
    Render selected PDF pages as images and extract
    text using Tesseract OCR.

    If page_numbers is None, OCR all pages.
    Otherwise, OCR only the requested pages.
    """

    pages = []

    logger.info(
        "Starting OCR processing: %s",
        file_path,
    )

    with fitz.open(file_path) as pdf:

        logger.info(
            "PDF contains %d pages.",
            len(pdf),
        )

        for page_number, page in enumerate(
            pdf,
            start=1,
        ):

            # Skip pages that do not require OCR
            if (
                page_numbers is not None
                and page_number not in page_numbers
            ):
                continue

            # Render page at higher resolution
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False,
            )

            # Convert rendered PDF page to PIL image
            image = Image.open(
                BytesIO(
                    pixmap.tobytes("png")
                )
            )

            # Preprocess image
            processed_image = preprocess_image(
                image
            )

            # Run Tesseract OCR
            text = pytesseract.image_to_string(
                processed_image,
                lang="eng",
            )

            text = text.strip()

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "character_count": len(text),
                    "extraction_method": "ocr",
                }
            )

            logger.info(
                "OCR completed for page %d | characters=%d",
                page_number,
                len(text),
            )

    logger.info(
        "OCR processing completed successfully."
    )

    return pages