from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    filename: str
    file_size: int
    content_type: str
    message: str
    document_id: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    extraction_method: str | None = None
