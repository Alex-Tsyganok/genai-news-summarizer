"""
Logging configuration for the AI News Summarizer.
"""
import logging
import sys
from typing import Optional
from config.settings import settings

def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Set up application logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    if level is None:
        level = settings.LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger("news_summarizer")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger

# Global logger instance
logger = setup_logging()
