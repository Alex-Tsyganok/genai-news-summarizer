"""
Configuration settings for the AI News Summarizer.

Best practice: avoid import-time side effects. This module lazily loads
environment variables on first access.
"""
import os
from typing import Dict, Any, Optional

class Settings:
    """Application settings and configuration (lazy-loaded)."""
    
    # Supported OpenAI embedding models
    SUPPORTED_EMBEDDING_MODELS = [
        "text-embedding-ada-002",
        "text-embedding-3-small", 
        "text-embedding-3-large"
    ]
    
    def __init__(self) -> None:
        # Mark as not yet loaded
        self._loaded = False

    # ----- Lazy loader internals -----
    def _get_int(self, name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _get_float(self, name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _get_bool(self, name: str, default: bool = False) -> bool:
        return os.getenv(name, str(default)).lower() in ("true", "1", "yes")

    def _read_env(self) -> None:
        """Read environment variables into instance attributes."""
        # OpenAI Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

        # LangSmith Configuration
        self.LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
        self.LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "genai-news-summarizer")
        self.LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        self.LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
        self.ENABLE_LANGSMITH = self._get_bool("ENABLE_LANGSMITH", False)

        # ChromaDB Configuration
        self.CHROMADB_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./data/chromadb")
        self.CHROMADB_COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION_NAME", "news_articles")

        # Extraction Configuration
        self.REQUEST_TIMEOUT = self._get_int("REQUEST_TIMEOUT", 30)
        self.MAX_RETRIES = self._get_int("MAX_RETRIES", 3)

        # AI Scoring Configuration
        self.MIN_CONFIDENCE_SCORE = self._get_float("MIN_CONFIDENCE_SCORE", 0.6)
        self.MIN_CONTENT_LENGTH = self._get_int("MIN_CONTENT_LENGTH", 300)

        # Summarization Configuration
        self.MAX_SUMMARY_LENGTH = self._get_int("MAX_SUMMARY_LENGTH", 1000)
        self.MAX_TOPICS = self._get_int("MAX_TOPICS", 5)
        self.SUMMARY_SENTENCES = os.getenv("SUMMARY_SENTENCES", "5-10 sentences")
        self.SUMMARY_WORD_LIMIT = self._get_int("SUMMARY_WORD_LIMIT", 100)
        self.SUMMARY_CHUNK_SIZE = self._get_int("SUMMARY_CHUNK_SIZE", 8000)

        # Search Configuration
        self.DEFAULT_SEARCH_LIMIT = self._get_int("DEFAULT_SEARCH_LIMIT", 10)
        self.SIMILARITY_THRESHOLD = self._get_float("SIMILARITY_THRESHOLD", 0.55)

        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

        # Fallback Mode
        self.USE_FALLBACK_ONLY = self._get_bool("USE_FALLBACK_ONLY", False)

        self._loaded = True

    def ensure_loaded(self) -> None:
        """Ensure settings are loaded from the environment."""
        if not getattr(self, "_loaded", False):
            # Remove dotenv auto-load here; keep loading in CLI/Streamlit
            self._read_env()

    # Deprecated: Settings.load_env removed in favor of standard dotenv in entry points.

    @classmethod
    def reset(cls) -> None:
        """Reset the lazy-loaded singleton to force re-read of environment (useful in tests)."""
        try:
            from .settings import settings as _singleton  # type: ignore
            _singleton._loaded = False
        except Exception:
            pass

    # Dynamically resolve attributes on first access
    def __getattr__(self, name: str):  # noqa: D401
        self.ensure_loaded()
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"Settings has no attribute '{name}'")

    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        s = settings
        s.ensure_loaded()
        return {
            "api_key": s.OPENAI_API_KEY,
            "model": s.OPENAI_MODEL,
            "embedding_model": s.OPENAI_EMBEDDING_MODEL,
        }
    
    @classmethod
    def get_chromadb_config(cls) -> Dict[str, Any]:
        """Get ChromaDB configuration dictionary."""
        if isinstance(cls, type):
            _s = settings
        else:
            _s = cls
        _s.ensure_loaded()
        return {
            "persist_directory": _s.CHROMADB_PERSIST_DIRECTORY,
            "collection_name": _s.CHROMADB_COLLECTION_NAME,
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration settings."""
        if isinstance(cls, type):
            _s = settings
        else:
            _s = cls
        _s.ensure_loaded()

        if not _s.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        
        # Validate OpenAI embedding model
        if _s.OPENAI_EMBEDDING_MODEL not in cls.SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"OPENAI_EMBEDDING_MODEL '{_s.OPENAI_EMBEDDING_MODEL}' is not supported. "
                f"Supported models: {', '.join(cls.SUPPORTED_EMBEDDING_MODELS)}"
            )
            
        # Validate LangSmith configuration if enabled
        if _s.ENABLE_LANGSMITH:
            if not _s.LANGCHAIN_API_KEY:
                raise ValueError("LANGCHAIN_API_KEY is required when ENABLE_LANGSMITH is true")
            if not _s.LANGCHAIN_PROJECT:
                raise ValueError("LANGCHAIN_PROJECT is required when ENABLE_LANGSMITH is true")
                
        return True

# Global settings instance
settings = Settings()
