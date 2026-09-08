
from pydantic import BaseModel, Field, field_validator


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to summarize")
    max_length: int = Field(150, ge=20, le=1000, description="Approx. max words in summary")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be empty or whitespace only")
        return v

class SummarizeResponse(BaseModel):
    summary: str
    status: str = "success"


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about uploaded documents")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be empty or whitespace only")
        return v

class QAResponse(BaseModel):
    answer: str
    sources: list[str]
    answer_source: str 
    status: str = "success"


class UploadResponse(BaseModel):
    filename: str
    chunks_stored: int
    extraction_method: str  
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    service: str
