"""
Vector database storage and retrieval using ChromaDB.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
import uuid
from typing import List, Dict, Any, Optional
import json
import os

from ..models import Article
from config import settings, logger

class VectorStorage:
    """
    Vector database storage using ChromaDB for article embeddings and retrieval.
    """
    
    def __init__(self):
        """Initialize the vector storage."""
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure persist directory exists
            os.makedirs(settings.CHROMADB_PERSIST_DIRECTORY, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.CHROMADB_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMADB_COLLECTION_NAME,
                metadata={"description": "News articles with AI-generated summaries and topics"}
            )
            
            logger.info(f"Initialized ChromaDB with collection: {settings.CHROMADB_COLLECTION_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def store_article(self, article: Article) -> bool:
        """
        Store an article in the vector database.
        
        Args:
            article: Article object to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate unique ID for the article
            article_id = self._generate_article_id(article)
            
            # Prepare document text (combination of title, summary, and topics)
            document_text = self._prepare_document_text(article)
            
            # Prepare metadata
            metadata = self._prepare_metadata(article)
            
            # Add to collection
            self.collection.add(
                documents=[document_text],
                metadatas=[metadata],
                ids=[article_id]
            )
            
            logger.info(f"Stored article: {article.title[:50]}... (ID: {article_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store article {article.title}: {e}")
            return False
    
    def store_articles(self, articles: List[Article]) -> Dict[str, bool]:
        """
        Store multiple articles in batch.
        
        Args:
            articles: List of articles to store
            
        Returns:
            Dictionary mapping article URLs to success status
        """
        results = {}
        
        try:
            # Prepare batch data
            ids = []
            documents = []
            metadatas = []
            
            for article in articles:
                article_id = self._generate_article_id(article)
                document_text = self._prepare_document_text(article)
                metadata = self._prepare_metadata(article)
                
                ids.append(article_id)
                documents.append(document_text)
                metadatas.append(metadata)
                
                results[article.source_url] = True
            
            # Batch insert
            if ids:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Stored {len(ids)} articles in batch")
            
        except Exception as e:
            logger.error(f"Failed to store articles in batch: {e}")
            # Mark all as failed
            for article in articles:
                results[article.source_url] = False
        
        return results
    
    def search_articles(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Search for articles using semantic similarity.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of search results with metadata
        """
        if limit is None:
            limit = settings.DEFAULT_SEARCH_LIMIT
        
        try:
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process and format results
            formatted_results = self._format_search_results(results, query)
            
            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search articles: {e}")
            return []
    
    def get_article_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific article by its URL.
        
        Args:
            url: Source URL of the article
            
        Returns:
            Article data or None if not found
        """
        try:
            # Query by URL in metadata
            results = self.collection.get(
                where={"source_url": url},
                include=['documents', 'metadatas']
            )
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get article by URL {url}: {e}")
            return None
    
    def delete_article(self, url: str) -> bool:
        """
        Delete an article by its URL.
        
        Args:
            url: Source URL of the article to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find article by URL
            results = self.collection.get(
                where={"source_url": url}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted article: {url}")
                return True
            
            logger.warning(f"Article not found for deletion: {url}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete article {url}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the article collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            return {
                'total_articles': count,
                'collection_name': settings.CHROMADB_COLLECTION_NAME,
                'persist_directory': settings.CHROMADB_PERSIST_DIRECTORY
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {'total_articles': 0}
    
    def _generate_article_id(self, article: Article) -> str:
        """
        Generate a unique ID for an article.
        
        Args:
            article: Article object
            
        Returns:
            Unique identifier string
        """
        # Use URL-based ID for consistency
        return f"article_{hash(article.source_url) % 1000000}"
    
    def _prepare_document_text(self, article: Article) -> str:
        """
        Prepare the text content for embedding.
        
        Args:
            article: Article object
            
        Returns:
            Combined text for embedding
        """
        # Combine title, summary, and topics for better search
        parts = [article.title]
        
        if article.summary:
            parts.append(article.summary)
        
        if article.topics:
            parts.append(" ".join(article.topics))
        
        # Add a portion of the body text
        if article.body:
            # Use first 500 characters of body for context
            body_excerpt = article.body[:500]
            parts.append(body_excerpt)
        
        return " | ".join(parts)
    
    def _prepare_metadata(self, article: Article) -> Dict[str, Any]:
        """
        Prepare metadata for storage.
        
        Args:
            article: Article object
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'title': article.title,
            'source_url': article.source_url,
            'summary': article.summary or '',
            'topics': json.dumps(article.topics or []),
            'extracted_at': article.extracted_at.isoformat() if article.extracted_at else '',
        }
        
        # Add original metadata if available
        if article.metadata:
            for key, value in article.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[f"meta_{key}"] = value
        
        return metadata
    
    def _format_search_results(self, results: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """
        Format search results into a standardized format.
        
        Args:
            results: Raw search results from ChromaDB
            query: Original search query
            
        Returns:
            Formatted search results
        """
        formatted_results = []
        
        if not results['ids'] or not results['ids'][0]:
            return formatted_results
        
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
            # Convert distance to similarity score (ChromaDB uses cosine distance)
            similarity_score = 1 - distance
            
            # Parse topics from JSON
            topics = []
            if metadata.get('topics'):
                try:
                    topics = json.loads(metadata['topics'])
                except:
                    topics = []
            
            formatted_result = {
                'id': doc_id,
                'title': metadata.get('title', ''),
                'summary': metadata.get('summary', ''),
                'source_url': metadata.get('source_url', ''),
                'topics': topics,
                'similarity_score': similarity_score,
                'rank': i + 1,
                'query': query
            }
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def reset_collection(self) -> bool:
        """
        Reset the collection (delete all articles).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(settings.CHROMADB_COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=settings.CHROMADB_COLLECTION_NAME,
                metadata={"description": "News articles with AI-generated summaries and topics"}
            )
            logger.info("Reset ChromaDB collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
