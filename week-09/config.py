"""
Central configuration for the AI Assistant Backend API.

Per mentor feedback on Assignment 8: model names are kept here as
configurable values instead of being hardcoded inside the logic files.
Change GEMINI_MODEL in one place if a model becomes unavailable/deprecated.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Model configuration (easy to swap) ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Storage paths ---
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

# --- Chunking settings ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Retrieval settings ---
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
