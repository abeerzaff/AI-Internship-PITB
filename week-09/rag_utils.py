"""
RAG utilities: PDF text extraction, chunking, embeddings, ChromaDB storage,
and retrieval. This reuses the pipeline built in Assignment 8
(Professional RAG PDF Assistant), now exposed for API use instead of a
Gradio interface.
"""

import os
import uuid
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import pymupdf  # PDF page rendering — pure pip install, no poppler/brew needed
import easyocr  # OCR engine — pure pip install (PyTorch-based), no tesseract/brew needed

import config

# Load embedding model once at import time (expensive to reload per request)
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

# Persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)


# EasyOCR reader is loaded once, lazily, on first use — it's the slow/heavy
# part (loads a neural network), so we don't want to pay that cost on every
# single upload, only once per server run.
_ocr_reader = None


def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def _extract_text_native(file_path: str) -> str:
    """Direct text-layer extraction via pypdf. Works on normal, text-based PDFs."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def _extract_text_ocr(file_path: str) -> str:
    """
    OCR fallback for scanned/image-based PDFs — no system binaries required.

    PyMuPDF renders each PDF page directly to a pixel image in-process (it
    has its own built-in rendering engine, so unlike pdf2image it does not
    need poppler installed on the machine). EasyOCR then reads the text out
    of each rendered image; it ships its own neural OCR model via pip, so it
    does not need the Tesseract binary installed either. Net effect: the
    whole OCR pipeline is `pip install` only, nothing to build with Homebrew.
    """
    doc = pymupdf.open(file_path)
    reader = _get_ocr_reader()
    text = ""
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        results = reader.readtext(img_array, detail=0)
        text += " ".join(results) + "\n"
    doc.close()
    return text


def extract_text_from_pdf(file_path: str) -> tuple[str, str]:
    """
    Extract text from a PDF file, automatically falling back to OCR when the
    PDF has no real text layer (scanned/image-based).

    Returns (text, method) where method is "native" or "ocr", so callers/
    API responses can report which path was used.
    """
    text = _extract_text_native(file_path)

    if len(text.strip()) >= 20:
        return text, "native"

    # Native extraction found effectively nothing -> likely scanned. Try OCR.
    ocr_text = _extract_text_ocr(file_path)

    if len(ocr_text.strip()) < 20:
        raise ValueError(
            "No extractable text found even after OCR. The PDF may be "
            "blank, corrupted, or too low-quality to read."
        )
    return ocr_text, "ocr"


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def process_and_store_document(file_path: str, doc_name: str) -> dict:
    """
    Full ingestion pipeline: extract (native or OCR) -> chunk -> embed ->
    store in ChromaDB. Returns a dict with chunk count and which extraction
    method was used, so the API can report it back to the caller.
    """
    text, method = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{doc_name}_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": doc_name, "chunk_index": i, "extraction_method": method}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return {"chunks_stored": len(chunks), "extraction_method": method}


def retrieve_relevant_chunks(query: str, top_k: int = None) -> list[dict]:
    """Retrieve the most relevant chunks for a query from ChromaDB."""
    top_k = top_k or config.TOP_K_RESULTS
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    retrieved = []
    if results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            retrieved.append({"text": doc, "source": meta.get("source"), "distance": dist})
    return retrieved


def build_context(chunks: list[dict]) -> str:
    """Combine retrieved chunks into a single context block for the LLM prompt."""
    return "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
