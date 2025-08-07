"""Main source package."""

from .models import Article, SearchResult, ProcessingResult
from .pipeline import NewsPipeline

__all__ = ['Article', 'SearchResult', 'ProcessingResult', 'NewsPipeline']
