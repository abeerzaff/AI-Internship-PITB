# API Documentation — AI Assistant Backend API

Base URL: `http://127.0.0.1:8000`


## GET /
Description:Home endpoint. Confirms the API is running and lists available routes.
Response 200
json
{
  "message": "AI Assistant Backend API is running.",
  "endpoints": ["/health", "/summarize", "/ask", "/upload", "/docs"]
}

## GET /health
Description:Health check endpoint for monitoring/testing tools.
Response 200
json
{ "status": "ok", "service": "ai-assistant-backend-api" }

## POST /summarize
Description:Summarizes input text using Gemini.
Request Body
json
{ "text": "Long text to summarize...", "max_length": 100 }
Response 200
json
{ "summary": "Short summary text.", "status": "success" }

Errors
1. 422 — validation error: empty text, whitespace-only text, missing field, wrong type, or `max_length` outside the 20–1000 range
2. 500— Gemini API failure after retries

## POST /upload
Description:Uploads a PDF, extracts text, chunks it, generates embeddings, and stores them in ChromaDB for later retrieval. Automatically falls back to OCR pymupdf for rendering + easyocr for text recognition.
Request: multipart/form-data, field name=file, PDF only.

Response 200 (normal PDF)
json
{ "filename": "sample.pdf", "chunks_stored": 12, "extraction_method": "native", "status": "success" }

Response 200 (scanned PDF)
json
{ "filename": "scanned.pdf", "chunks_stored": 5, "extraction_method": "ocr", "status": "success" }
Errors
1. 400 = file is not a PDF
2. 422 = no file provided, or no extractable text found via native extraction *or* OCR (e.g. blank/corrupted PDF)
3. 500 — unexpected processing error

## POST /ask
Description: Answers a question. Prefers RAG over uploaded documents when relevant chunks exist; otherwise (including when no document has been uploaded) falls back to the LLM's own general knowledge instead of refusing to answer.

Request Body
json
{ "question": "What is this document about?" }

Response 200 (answered from a document)
json
{
  "answer": "The document discusses...",
  "sources": ["sample.pdf"],
  "answer_source": "document",
  "status": "success"
}

Response 200 (answered from general knowledge — no relevant document content, or none uploaded)
json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "sources": [],
  "answer_source": "general_knowledge",
  "status": "success"
}
Errors
1. 422=validation error: empty or whitespace-only question, or missing field
2. 500=Gemini API failure after retries

## Interactive Docs
FastAPI auto-generates a Swagger UI at `/docs` and ReDoc at `/redoc` — use
either for live testing alongside Postman.