"""
Configuration settings for the AI News Summarizer.
"""
import os
from typing import Dict, Any

class Settings:
    """Application settings and configuration."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # ChromaDB Configuration
    CHROMADB_PERSIST_DIRECTORY: str = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./data/chromadb")
    CHROMADB_COLLECTION_NAME: str = os.getenv("CHROMADB_COLLECTION_NAME", "news_articles")
    
    # Extraction Configuration
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    
    # AI Scoring Configuration
    MIN_CONFIDENCE_SCORE: float = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.6"))  # Minimum score to consider article as valid news
    MIN_CONTENT_LENGTH: int = int(os.getenv("MIN_CONTENT_LENGTH", "300"))  # Minimum length for AI analysis
    
    # Summarization Configuration
    MAX_SUMMARY_LENGTH: int = int(os.getenv("MAX_SUMMARY_LENGTH", "200"))
    MAX_TOPICS: int = int(os.getenv("MAX_TOPICS", "5"))
    
    # Search Configuration
    DEFAULT_SEARCH_LIMIT: int = int(os.getenv("DEFAULT_SEARCH_LIMIT", "10"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Fallback Mode (Development/Testing)
    # When enabled, bypasses all AI calls and uses basic text processing instead
    # Useful for: development, testing, CI/CD, offline demos, cost control
    USE_FALLBACK_ONLY: bool = os.getenv("USE_FALLBACK_ONLY", "false").lower() in ("true", "1", "yes")
    
    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        """Get OpenAI configuration dictionary."""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "embedding_model": cls.OPENAI_EMBEDDING_MODEL
        }
    
    @classmethod
    def get_chromadb_config(cls) -> Dict[str, Any]:
        """Get ChromaDB configuration dictionary."""
        return {
            "persist_directory": cls.CHROMADB_PERSIST_DIRECTORY,
            "collection_name": cls.CHROMADB_COLLECTION_NAME
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration settings."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        return True

# Global settings instance
settings = Settings()
