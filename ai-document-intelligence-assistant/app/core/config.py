
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_DIR = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    """

    app_name: str = "AI-Powered Document Intelligence Assistant"
    app_version: str = "1.0.0"
    environment: str = "development"

    gemini_api_key: str

    model_name: str = "gemini-3.6-flash"
    allow_general_knowledge: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 800
    chunk_overlap: int = 150

    top_k: int = 15
    similarity_threshold: float = 0.25
    max_analysis_chars: int = 200000

    max_file_size_mb: int = 50

    chroma_path: str = "data/chroma"
    upload_dir: str = "data/uploads"

    model_config = SettingsConfigDict(
        env_file=(ENV_DIR / ".env", BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API configuration
    api_url: str = "http://127.0.0.1:8000"

    @property
    def max_file_size_bytes(self) -> int:
        """
        Convert configured file size from MB to bytes.
        """
        return self.max_file_size_mb * 1024 * 1024

    @property
    def chroma_directory(self) -> Path:
        """
        Return the absolute ChromaDB directory.
        """
        return BASE_DIR / self.chroma_path

    @property
    def upload_directory(self) -> Path:
        """
        Return the absolute upload directory.
        """
        return BASE_DIR / self.upload_dir


settings = Settings()