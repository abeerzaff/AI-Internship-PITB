
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings
from app.services.ocr_service import ocr_pdf
from app.services.pdf_service import (
    MINIMUM_PAGE_CHARACTERS,
    extract_text_from_pdf,
)
from app.utils.text_cleaner import clean_text
from app.utils.validators import (
    validate_file_size,
    validate_pdf_file,
)

logger = logging.getLogger(__name__)


async def save_uploaded_document(file: UploadFile) -> dict:
    """Validate and save an uploaded PDF."""

    validate_pdf_file(file)

    file_content = await file.read()
    validate_file_size(len(file_content))

    unique_filename = (
        f"{uuid4().hex}_{Path(file.filename).name}"
    )

    settings.upload_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = settings.upload_directory / unique_filename
    destination.write_bytes(file_content)

    logger.info("Document uploaded: %s", destination.name)

    return {
        "filename": file.filename,
        "stored_filename": unique_filename,
        "file_size": len(file_content),
        "content_type": file.content_type,
        "file_path": str(destination),
    }


def process_document(file_path: str) -> dict:
    """
    Extract and clean PDF text.

    1. Try native PyMuPDF extraction.
    2. Identify pages with insufficient native text.
    3. Apply OCR only to weak pages.
    4. Preserve native text when OCR does not improve
       the amount of extracted text.
    """

    logger.info("Processing document: %s", file_path)

    # Step 1: Native PDF text extraction
    pages = extract_text_from_pdf(file_path)

    for page in pages:
        page["extraction_method"] = "native"

    # Step 2: Identify pages potentially requiring OCR
    weak_pages = [
        page["page_number"]
        for page in pages
        if page["character_count"] < MINIMUM_PAGE_CHARACTERS
    ]

    # Step 3: Apply OCR only to weak pages
    ocr_by_page = {}

    if weak_pages:
        logger.info("OCR needed for pages: %s", weak_pages)

        ocr_pages = ocr_pdf(
            file_path,
            weak_pages,
        )

        ocr_by_page = {
            page["page_number"]: page
            for page in ocr_pages
        }

    # Step 4: Choose native or OCR text for each page
    updated_pages = []

    for page in pages:
        page_number = page["page_number"]
        ocr_page = ocr_by_page.get(page_number)

        if ocr_page is None:
            updated_pages.append(page)
            continue

        native_text = page.get("text", "").strip()
        ocr_text = ocr_page.get("text", "").strip()

        if ocr_text and (
            not native_text
            or len(ocr_text) > len(native_text)
        ):
            updated_pages.append(ocr_page)
        else:
            logger.info(
                "Keeping native text for page %d",
                page_number,
            )
            updated_pages.append(page)

    pages = updated_pages

    # Step 5: Determine document extraction method
    methods = {
        page["extraction_method"]
        for page in pages
    }

    if methods == {"ocr"}:
        extraction_method = "ocr"
    elif "ocr" in methods:
        extraction_method = "mixed"
    else:
        extraction_method = "native"

    # Step 6: Clean extracted text
    for page in pages:
        page["text"] = clean_text(page["text"])
        page["character_count"] = len(page["text"])

    # Step 7: Calculate extracted character count
    total_characters = sum(
        page["character_count"]
        for page in pages
    )

    logger.info(
        "Document processed: %d pages, %d characters, method=%s",
        len(pages),
        total_characters,
        extraction_method,
    )

    return {
        "file_path": file_path,
        "page_count": len(pages),
        "total_characters": total_characters,
        "extraction_method": extraction_method,
        "pages": pages,
    }
