
from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    source: str
    page: int | str
    similarity: float | None = None


class DocumentAIResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[SourceReference] = Field(
        default_factory=list
    )