"""
News article data models and structures.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class Article:
    """
    Data model for a news article.
    
    Attributes:
        title: Article headline/title
        body: Full article text content
        source_url: Original URL of the article
        summary: AI-generated summary (optional)
        topics: List of identified topics/keywords (optional)
        extracted_at: Timestamp when article was extracted
        metadata: Additional metadata about the article
    """
    title: str
    body: str
    source_url: str
    summary: Optional[str] = None
    topics: Optional[List[str]] = None
    extracted_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.extracted_at is None:
            self.extracted_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if self.topics is None:
            self.topics = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary representation."""
        return {
            'title': self.title,
            'body': self.body,
            'source_url': self.source_url,
            'summary': self.summary,
            'topics': self.topics,
            'extracted_at': self.extracted_at.isoformat() if self.extracted_at else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Article':
        """Create article from dictionary representation."""
        extracted_at = None
        if data.get('extracted_at'):
            extracted_at = datetime.fromisoformat(data['extracted_at'])
        
        return cls(
            title=data['title'],
            body=data['body'],
            source_url=data['source_url'],
            summary=data.get('summary'),
            topics=data.get('topics', []),
            extracted_at=extracted_at,
            metadata=data.get('metadata', {})
        )

@dataclass
class SearchResult:
    """
    Data model for search results.
    
    Attributes:
        article: The matching article
        score: Similarity score (0.0 to 1.0)
        rank: Position in search results
    """
    article: Article
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary representation."""
        return {
            'article': self.article.to_dict(),
            'score': self.score,
            'rank': self.rank
        }

@dataclass
class ProcessingResult:
    """
    Data model for article processing results.
    
    Attributes:
        success: Whether processing was successful
        article: The processed article (if successful)
        error: Error message (if failed)
        processing_time: Time taken to process in seconds
    """
    success: bool
    article: Optional[Article] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert processing result to dictionary representation."""
        return {
            'success': self.success,
            'article': self.article.to_dict() if self.article else None,
            'error': self.error,
            'processing_time': self.processing_time
        }
