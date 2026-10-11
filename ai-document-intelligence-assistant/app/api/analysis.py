
import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.ai_response import ItemsResponse, SummaryResponse
from app.services.action_service import extract_action_items
from app.services.context_service import build_context
from app.services.keypoint_service import extract_key_points
from app.services.llm_service import (
    LLMQuotaExceededError,
    LLMUnavailableError,
)
from app.services.summary_service import summarize_document
from app.services.vector_service import (
    get_all_chunks,
    validate_active_document,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Document Analysis"],
)


def _load(document_id: str | None):
    try:
        document_id = validate_active_document(document_id)
        chunks = get_all_chunks(document_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Upload a document first.",
        )

    selected = []
    total = 0

    for chunk in chunks:
        length = len(chunk["text"])

        if total + length > settings.max_analysis_chars:
            break

        selected.append(chunk)
        total += length

    if not selected:
        raise HTTPException(
            status_code=400,
            detail="The document content could not fit the analysis limit.",
        )

    pages = {
        (
            chunk["metadata"]["source"],
            chunk["metadata"]["page"],
        )
        for chunk in selected
    }

    sources = [
        {"source": source, "page": page}
        for source, page in sorted(pages)
    ]

    return build_context(selected), sources


def _run(function, context: str):
    try:
        return function(context)

    except LLMQuotaExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail="AI service quota has been reached. Please try again later.",
        ) from exc

    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="The AI service is busy. Please try again shortly.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Document analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the request.",
        ) from exc


@router.post(
    "/summary",
    response_model=SummaryResponse,
)
def summary(document_id: str):
    context, sources = _load(document_id)

    return {
        "summary": _run(summarize_document, context),
        "sources": sources,
    }


@router.post(
    "/key-points",
    response_model=ItemsResponse,
)
def key_points(document_id: str):
    context, sources = _load(document_id)

    return {
        "items": _run(extract_key_points, context),
        "sources": sources,
    }


@router.post(
    "/action-items",
    response_model=ItemsResponse,
)
def action_items(document_id: str):
    context, sources = _load(document_id)

    return {
        "items": _run(extract_action_items, context),
        "sources": sources,
    }
