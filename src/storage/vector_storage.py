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
    
    def __init__(self, chromadb_config=None):
        """
        Initialize the vector storage.
        
        Args:
            chromadb_config: Optional custom ChromaDB configuration
        """
        self.config = chromadb_config or settings.get_chromadb_config()
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure persist directory exists
            os.makedirs(self.config['persist_directory'], exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.config['persist_directory'],
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # OpenAI embeddings are mandatory for this system
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required - OpenAI embeddings are mandatory for this system")
            
            # Initialize with OpenAI embeddings
            import chromadb.utils.embedding_functions as embedding_functions
            
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_EMBEDDING_MODEL
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.config['collection_name'],
                embedding_function=openai_ef,
                metadata={
                    "description": "News articles with OpenAI embeddings",
                    "embedding_function": "openai",
                    "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
                    "embedding_dimensions": 3072 if "3-large" in settings.OPENAI_EMBEDDING_MODEL else 1536
                }
            )
            logger.info(f"Initialized ChromaDB with OpenAI embeddings: {settings.OPENAI_EMBEDDING_MODEL}")
            
            logger.info(f"Initialized ChromaDB with collection: {self.config['collection_name']}")
            
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
            # Check if article already exists
            existing_article = self.get_article_by_url(article.source_url)
            if existing_article:
                logger.info(f"Article already exists, skipping: {article.title[:50]}...")
                return True  # Consider this a success since the article is already stored
            
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
            # Prepare batch data, filtering out duplicates
            ids = []
            documents = []
            metadatas = []
            new_articles = []
            
            for article in articles:
                # Check if article already exists
                existing_article = self.get_article_by_url(article.source_url)
                if existing_article:
                    logger.info(f"Article already exists, skipping: {article.title[:50]}...")
                    results[article.source_url] = True  # Consider this a success
                    continue
                
                article_id = self._generate_article_id(article)
                document_text = self._prepare_document_text(article)
                metadata = self._prepare_metadata(article)
                
                ids.append(article_id)
                documents.append(document_text)
                metadatas.append(metadata)
                new_articles.append(article)
                
                results[article.source_url] = True
            
            # Batch insert only new articles
            if ids:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Stored {len(ids)} new articles in batch")
            else:
                logger.info("No new articles to store - all were duplicates")
            
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
        Generate a unique ID for an article based on its URL.
        
        Args:
            article: Article object
            
        Returns:
            Unique identifier string based on URL hash
        """
        import hashlib
        # Use SHA-256 hash of URL for consistent, deterministic ID generation
        url_hash = hashlib.sha256(article.source_url.encode('utf-8')).hexdigest()
        # Use first 8 characters for a shorter, readable ID
        return f"article_{url_hash[:8]}"
    
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
            'body': article.body or '',  # Include original article body
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
            # Convert distance to similarity score
            # ChromaDB uses cosine distance. For cosine distance:
            # distance = 0 means identical vectors (similarity = 1.0)
            # distance = 1 means orthogonal vectors (similarity = 0.0)
            # distance = 2 means opposite vectors (similarity = -1.0)
            # Formula: cosine_similarity = 1 - (cosine_distance / 2)
            # This maps distance [0,2] to similarity [1,0]
            similarity_score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            
            # Debug: log the raw distance and calculated similarity
            logger.debug(f"Raw distance: {distance:.4f}, Calculated similarity: {similarity_score:.4f}")
            
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
                'query': query,
                'extracted_at': metadata.get('extracted_at', ''),
                'body': metadata.get('body', ''),  # Include original article body
                'document_text': document  # Include the prepared document text used for search
            }
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def reset_collection(self) -> bool:
        """
        Reset the collection (delete all articles) and recreate with OpenAI embeddings.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing collection
            self.client.delete_collection(settings.CHROMADB_COLLECTION_NAME)
            
            # Recreate with OpenAI embeddings (mandatory)
            import chromadb.utils.embedding_functions as embedding_functions
            
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_EMBEDDING_MODEL
            )
            
            self.collection = self.client.create_collection(
                name=settings.CHROMADB_COLLECTION_NAME,
                embedding_function=openai_ef,
                metadata={
                    "description": "News articles with OpenAI embeddings",
                    "embedding_function": "openai",
                    "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
                    "embedding_dimensions": 3072 if "3-large" in settings.OPENAI_EMBEDDING_MODEL else 1536
                }
            )
            logger.info("Reset ChromaDB collection with OpenAI embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
    def get_all_articles(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve all articles from storage without search/embedding operations.
        
        Args:
            limit: Maximum number of articles to retrieve
            
        Returns:
            List of article dictionaries with metadata
        """
        try:
            # Get data directly without embeddings/search to avoid API calls
            query_limit = limit if limit else 1000  # ChromaDB default max
            
            results = self.collection.get(
                limit=query_limit,
                include=['documents', 'metadatas']  # Exclude embeddings to avoid API calls
            )
            
            if not results['ids']:
                logger.info("No articles found in collection")
                return []
            
            articles = []
            for i, article_id in enumerate(results['ids']):
                try:
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    document = results['documents'][i] if results['documents'] else ''
                    
                    # Parse topics from JSON if needed
                    topics = metadata.get('topics', [])
                    if isinstance(topics, str):
                        try:
                            import json
                            topics = json.loads(topics)
                        except:
                            topics = topics.split(',') if topics else []
                    
                    # Create article dictionary
                    article_data = {
                        'id': article_id,
                        'title': metadata.get('title', 'Unknown Title'),
                        'summary': metadata.get('summary', ''),
                        'topics': topics,
                        'source_url': metadata.get('source_url', ''),
                        'body': metadata.get('body', ''),
                        'extracted_at': metadata.get('extracted_at', ''),
                        'document_text': document,
                        # Include any additional metadata fields
                        'meta_extraction_method': metadata.get('meta_extraction_method', ''),
                        'meta_ai_confidence_score': metadata.get('meta_ai_confidence_score', 0),
                        'meta_top_image': metadata.get('meta_top_image', '')
                    }
                    
                    articles.append(article_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse article {article_id}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(articles)} articles from storage")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to retrieve articles: {e}")
            return []

    def remove_duplicates(self) -> Dict[str, Any]:
        """
        Remove duplicate articles based on source URL, keeping the latest version.
        
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            # Get all articles
            all_articles = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            if not all_articles['ids']:
                return {'total_articles': 0, 'duplicates_removed': 0, 'unique_articles': 0}
            
            # Group by URL to find duplicates
            url_groups = {}
            for i, (article_id, metadata) in enumerate(zip(all_articles['ids'], all_articles['metadatas'])):
                url = metadata.get('source_url', '')
                if url:
                    if url not in url_groups:
                        url_groups[url] = []
                    url_groups[url].append({
                        'id': article_id,
                        'extracted_at': metadata.get('extracted_at', ''),
                        'metadata': metadata,
                        'document': all_articles['documents'][i]
                    })
            
            # Find duplicates and identify which to remove
            duplicates_to_remove = []
            unique_count = 0
            
            for url, articles in url_groups.items():
                if len(articles) > 1:
                    # Sort by extracted_at timestamp, keep the latest
                    articles.sort(key=lambda x: x['extracted_at'], reverse=True)
                    # Mark all but the first (latest) for removal
                    duplicates_to_remove.extend([article['id'] for article in articles[1:]])
                    logger.info(f"Found {len(articles)} duplicates for URL: {url}")
                
                unique_count += 1
            
            # Remove duplicates
            if duplicates_to_remove:
                self.collection.delete(ids=duplicates_to_remove)
                logger.info(f"Removed {len(duplicates_to_remove)} duplicate articles")
            
            return {
                'total_articles': len(all_articles['ids']),
                'duplicates_removed': len(duplicates_to_remove),
                'unique_articles': unique_count
            }
            
        except Exception as e:
            logger.error(f"Failed to remove duplicates: {e}")
            return {'error': str(e)}
