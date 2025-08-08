"""
Semantic search functionality for finding relevant articles.
"""
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

from ..models import Article, SearchResult
from ..storage import VectorStorage
from config import settings, logger

class SemanticSearcher:
    """
    Semantic search engine for finding relevant news articles.
    
    Provides natural language search capabilities over stored articles
    using vector similarity and additional filtering options.
    """
    
    def __init__(self, vector_storage: VectorStorage, similarity_threshold: float = None):
        """
        Initialize the semantic searcher.
        
        Args:
            vector_storage: Vector storage instance for search operations
            similarity_threshold: Optional similarity threshold for search results
        """
        self.vector_storage = vector_storage
        self.similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
    
    def search(self, 
               query: str, 
               limit: int = None,
               min_score: float = None,
               topics_filter: List[str] = None) -> List[SearchResult]:
        """
        Perform semantic search for articles.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results to return
            min_score: Minimum similarity score threshold
            topics_filter: Filter results by specific topics
            
        Returns:
            List of SearchResult objects ranked by relevance
        """
        if limit is None:
            limit = settings.DEFAULT_SEARCH_LIMIT
        if min_score is None:
            min_score = self.similarity_threshold
        
        try:
            logger.info(f"Performing semantic search: '{query}'")
            
            # Clean and enhance the query
            enhanced_query = self._enhance_query(query)
            
            # Perform vector search
            raw_results = self.vector_storage.search_articles(
                enhanced_query, 
                limit=limit * 2  # Get more results for filtering
            )
            
            # Convert to SearchResult objects
            search_results = self._convert_to_search_results(raw_results)
            
            # Apply filters
            filtered_results = self._apply_filters(
                search_results, 
                min_score=min_score,
                topics_filter=topics_filter
            )
            
            # Limit results
            final_results = filtered_results[:limit]
            
            logger.info(f"Found {len(final_results)} relevant articles")
            return final_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_topics(self, topics: List[str], limit: int = None) -> List[SearchResult]:
        """
        Search articles by specific topics.
        
        Args:
            topics: List of topics to search for
            limit: Maximum number of results
            
        Returns:
            List of matching articles
        """
        if not topics:
            return []
        
        # Create a query from topics
        query = " ".join(topics)
        return self.search(query, limit=limit, topics_filter=topics)
    
    def find_similar_articles(self, article_url: str, limit: int = 5) -> List[SearchResult]:
        """
        Find articles similar to a given article.
        
        Args:
            article_url: URL of the reference article
            limit: Maximum number of similar articles to return
            
        Returns:
            List of similar articles
        """
        try:
            # Get the reference article
            reference_article = self.vector_storage.get_article_by_url(article_url)
            if not reference_article:
                logger.warning(f"Reference article not found: {article_url}")
                return []
            
            # Use the article's summary and topics as query
            metadata = reference_article['metadata']
            summary = metadata.get('summary', '')
            topics = metadata.get('topics', '[]')
            
            # Create search query
            query_parts = []
            if summary:
                query_parts.append(summary)
            
            try:
                import json
                topics_list = json.loads(topics)
                if topics_list:
                    query_parts.extend(topics_list)
            except:
                pass
            
            if not query_parts:
                return []
            
            search_query = " ".join(query_parts)
            
            # Perform search
            results = self.search(search_query, limit=limit + 1)
            
            # Remove the reference article from results
            similar_articles = [
                result for result in results 
                if result.article.source_url != article_url
            ]
            
            return similar_articles[:limit]
            
        except Exception as e:
            logger.error(f"Failed to find similar articles: {e}")
            return []
    
    def get_trending_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending topics from stored articles.
        
        Args:
            limit: Maximum number of topics to return
            
        Returns:
            List of trending topics with counts
        """
        try:
            # This is a simplified implementation
            # In a real system, you might analyze recent articles
            
            # Get recent articles (simplified approach)
            recent_results = self.vector_storage.search_articles(
                "news today recent", 
                limit=100
            )
            
            # Count topic frequencies
            topic_counts = {}
            for result in recent_results:
                metadata = result.get('metadata', {})
                topics_str = metadata.get('topics', '[]')
                
                try:
                    import json
                    topics = json.loads(topics_str)
                    for topic in topics:
                        if isinstance(topic, str) and len(topic) > 2:
                            topic_counts[topic] = topic_counts.get(topic, 0) + 1
                except:
                    continue
            
            # Sort by frequency
            trending_topics = [
                {'topic': topic, 'count': count}
                for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
            ]
            
            return trending_topics[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get trending topics: {e}")
            return []
    
    def _enhance_query(self, query: str) -> str:
        """
        Enhance the search query for better results.
        
        Args:
            query: Original search query
            
        Returns:
            Enhanced query string
        """
        # Clean the query
        enhanced = query.strip().lower()
        
        # Add common news-related terms for context
        if len(enhanced.split()) < 3:
            # For short queries, add context
            enhanced = f"{enhanced} news article report"
        
        return enhanced
    
    def _convert_to_search_results(self, raw_results: List[Dict[str, Any]]) -> List[SearchResult]:
        """
        Convert raw search results to SearchResult objects.
        
        Args:
            raw_results: Raw results from vector storage
            
        Returns:
            List of SearchResult objects
        """
        search_results = []
        
        for i, result in enumerate(raw_results):
            try:
                # Parse topics
                topics = []
                topics_str = result.get('topics', [])
                if isinstance(topics_str, list):
                    topics = topics_str
                elif isinstance(topics_str, str):
                    try:
                        import json
                        topics = json.loads(topics_str)
                    except:
                        topics = []
                
                # Create Article object
                article = Article(
                    title=result.get('title', ''),
                    body='',  # Not stored in search results
                    source_url=result.get('source_url', ''),
                    summary=result.get('summary', ''),
                    topics=topics
                )
                
                # Create SearchResult
                search_result = SearchResult(
                    article=article,
                    score=result.get('similarity_score', 0.0),
                    rank=i + 1
                )
                
                search_results.append(search_result)
                
            except Exception as e:
                logger.warning(f"Failed to convert search result: {e}")
                continue
        
        return search_results
    
    def _apply_filters(self, 
                      results: List[SearchResult],
                      min_score: float = None,
                      topics_filter: List[str] = None) -> List[SearchResult]:
        """
        Apply filters to search results.
        
        Args:
            results: List of search results to filter
            min_score: Minimum similarity score
            topics_filter: Topics that must be present
            
        Returns:
            Filtered search results
        """
        filtered_results = results
        
        # Apply score filter
        if min_score is not None:
            filtered_results = [
                result for result in filtered_results 
                if result.score >= min_score
            ]
        
        # Apply topics filter
        if topics_filter:
            topics_filter_lower = [topic.lower() for topic in topics_filter]
            filtered_results = [
                result for result in filtered_results
                if any(
                    any(filter_topic in article_topic.lower() 
                        for article_topic in result.article.topics or [])
                    for filter_topic in topics_filter_lower
                )
            ]
        
        return filtered_results
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            
        Returns:
            List of suggested queries
        """
        try:
            # This is a simplified implementation
            # In a real system, you might use a more sophisticated approach
            
            suggestions = []
            
            # Get trending topics as suggestions
            trending = self.get_trending_topics(limit=5)
            for topic_info in trending:
                topic = topic_info['topic']
                if partial_query.lower() in topic.lower():
                    suggestions.append(topic)
            
            # Add some common news categories
            common_categories = [
                "technology", "politics", "business", "science", "health",
                "sports", "entertainment", "world news", "economy"
            ]
            
            for category in common_categories:
                if partial_query.lower() in category.lower():
                    suggestions.append(category)
            
            return suggestions[:5]
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []
