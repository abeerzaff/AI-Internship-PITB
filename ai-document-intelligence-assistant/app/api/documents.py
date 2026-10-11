
from pathlib import Path
import logging

import pytesseract

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas.document import DocumentUploadResponse
from app.services.document_service import save_uploaded_document
from app.services.indexing_service import index_document

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
):
    # Step 1: Validate and save the PDF
    saved = await save_uploaded_document(file)

    try:
        # Step 2: Extract, chunk, embed and index
        indexed = await run_in_threadpool(
            index_document,
            saved["file_path"],
        )

    except ValueError as exc:
        Path(saved["file_path"]).unlink(missing_ok=True)

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except pytesseract.TesseractNotFoundError as exc:
        logger.exception("Tesseract OCR is not installed.")

        Path(saved["file_path"]).unlink(missing_ok=True)

        raise HTTPException(
            status_code=503,
            detail="The OCR engine (Tesseract) is not installed.",
        ) from exc

    except Exception as exc:
        logger.exception("Document indexing failed.")

        # Preserve the new file because some chunks may already
        # have been indexed before this exception occurred.
        logger.error(
            "Preserving uploaded file for recovery: %s",
            saved["file_path"],
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Document indexing did not complete safely. "
                "Please check server logs before retrying."
            ),
        ) from exc

    # Step 3: Return new document information
    return DocumentUploadResponse(
        filename=saved["filename"],
        file_size=saved["file_size"],
        content_type=saved["content_type"],
        message="PDF uploaded and indexed.",
        document_id=indexed["document_id"],
        page_count=indexed["page_count"],
        chunk_count=indexed["chunk_count"],
        extraction_method=indexed["extraction_method"],
    )
