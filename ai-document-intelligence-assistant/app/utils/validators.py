
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.config import settings


ALLOWED_CONTENT_TYPES = {
    "application/pdf",
}

ALLOWED_EXTENSIONS = {
    ".pdf",
}


def validate_pdf_file(file: UploadFile) -> None:
    """
    Validate the uploaded file's name and content type.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename was provided.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a PDF.",
        )


def validate_file_size(file_size: int) -> None:
    """
    Validate the uploaded file size.
    """

    if file_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if file_size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size exceeds the maximum allowed size "
                f"of {settings.max_file_size_mb} MB."
            ),
        )