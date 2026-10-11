


import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.schemas.ai_response import DocumentAIResponse
from app.services.llm_service import (
    LLMQuotaExceededError,
    LLMUnavailableError,
)
from app.services.rag_service import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Document ID returned by the upload endpoint.",
    )


@router.post(
    "/ask",
    response_model=DocumentAIResponse,
)
def ask_question(request: ChatRequest):
    try:
        return answer_question(
            request.question,
            request.document_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

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

    except Exception as exc:
        logger.exception("Failed to answer question")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the request.",
        ) from exc
