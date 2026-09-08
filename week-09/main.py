"""
AI Assistant Backend API
Assignment 9 — AI Product Development and Final Delivery

Endpoints:
  GET  /                -> Home
  GET  /health           -> Health check
  POST /summarize        -> Text summarization
  POST /ask               -> Question answering (RAG-based, over uploaded docs)
  POST /upload            -> Document upload (PDF ingestion into ChromaDB)

Run with:
  uvicorn main:app --reload
"""

import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
import rag_utils
import ai_client
from schemas import (
    SummarizeRequest,
    SummarizeResponse,
    QARequest,
    QAResponse,
    UploadResponse,
    HealthResponse,
)

app = FastAPI(
    title="AI Assistant Backend API",
    description="Backend API exposing text summarization, RAG-based Q&A, "
                 "and document upload as AI-powered endpoints.",
    version="1.0.0",
)

# Allow local frontend/Postman testing without CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)


@app.get("/", tags=["General"])
def home():
    """Home endpoint — confirms the API is running and lists available routes."""
    return {
        "message": "AI Assistant Backend API is running.",
        "endpoints": ["/health", "/summarize", "/ask", "/upload", "/docs"],
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint — used by monitoring tools / Postman smoke tests."""
    return {"status": "ok", "service": "ai-assistant-backend-api"}


@app.post("/summarize", response_model=SummarizeResponse, tags=["AI"])
def summarize(request: SummarizeRequest):
    """Summarize arbitrary input text using Gemini."""

    try:
        summary = ai_client.summarize_text(
            request.text,
            request.max_length
        )

        return {
            "summary": summary,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )

@app.post("/ask", response_model=QAResponse, tags=["AI"])
def ask_question(request: QARequest):
    """
    Answer a question. Retrieves relevant chunks from previously uploaded
    documents (ChromaDB) if any exist and are relevant; otherwise (or if no
    documents were uploaded at all) answers using the LLM's own general
    knowledge instead of refusing. The response reports which source was
    actually used via `answer_source`.
    """
    try:
        chunks = rag_utils.retrieve_relevant_chunks(request.question)
        context = rag_utils.build_context(chunks)
        result = ai_client.answer_question(request.question, context)
        sources = list({c["source"] for c in chunks}) if result["answer_source"] == "document" else []
        return {
            "answer": result["answer"],
            "sources": sources,
            "answer_source": result["answer_source"],
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.post("/upload", response_model=UploadResponse, tags=["AI"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document. The file is saved, text is extracted (falling
    back to OCR automatically if the PDF is scanned/image-based), chunked,
    embedded, and stored in ChromaDB so it can later be queried via /ask.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = rag_utils.process_and_store_document(file_path, file.filename)
    except ValueError as e:
        # Raised only when NEITHER native extraction NOR OCR find any text
        # (e.g. blank/corrupted PDF) — a genuine dead end, not a fixable case.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

    return {
        "filename": file.filename,
        "chunks_stored": result["chunks_stored"],
        "extraction_method": result["extraction_method"],
        "status": "success",
    }
