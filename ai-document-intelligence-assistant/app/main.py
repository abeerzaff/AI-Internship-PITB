
from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
from app.api.analysis import router as analysis_router

from app.core.config import settings
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered document intelligence assistant "
        "for public-sector documents."
    ),
)

# Register each API router only once.
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(analysis_router)


@app.get("/")
def root():
    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
    }
