"""
Main pipeline for news processing and search.
"""
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime

from .models import Article, ProcessingResult, SearchResult
from .extractors import NewsExtractor
from .summarizers import AIsummarizer
from .storage import VectorStorage
from .search import SemanticSearcher
from .scoring import AIConfidenceScorer
from config import settings, logger

class NewsPipeline:
    """
    Main pipeline that orchestrates news extraction, summarization, and search.
    
    This class provides a high-level interface for the entire news processing workflow.
    """
    
    def __init__(self):
        """Initialize the news pipeline with all components."""
        logger.info("Initializing News Pipeline...")
        
        # Initialize components
        self.extractor = NewsExtractor()
        self.ai_scorer = AIConfidenceScorer()
        self.summarizer = AIsummarizer()
        self.storage = VectorStorage()
        self.searcher = SemanticSearcher(self.storage)
        
        logger.info("News Pipeline initialized successfully")
    
    def process_articles(self, urls: List[str]) -> Dict[str, Any]:
        """
        Process a list of article URLs through the complete pipeline.
        
        Args:
            urls: List of article URLs to process
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        logger.info(f"Processing {len(urls)} articles...")
        
        results = {
            'total_urls': len(urls),
            'successful': 0,
            'failed': 0,
            'errors': [],
            'processed_articles': [],
            'processing_time': 0
        }
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing article {i}/{len(urls)}: {url}")
            
            try:
                # Step 1: Extract article content
                extraction_result = self.extractor.extract_article(url)
                
                if not extraction_result.success:
                    results['failed'] += 1
                    error_details = {
                        'url': url,
                        'step': 'extraction',
                        'error': extraction_result.error
                    }
                    
                    # Add confidence score if available
                    if extraction_result.article and 'confidence_score' in extraction_result.article.metadata:
                        error_details['confidence_score'] = extraction_result.article.metadata['confidence_score']
                    
                    results['errors'].append(error_details)
                    continue
                
                article = extraction_result.article
                
                # Step 2: Calculate AI confidence score
                try:
                    logger.info(f"Calculating AI confidence score for article: {article.title[:60]}...")
                    
                    scoring_result = self.ai_scorer.score_article(article)
                    if not scoring_result.success:
                        logger.error(f"AI scoring failed: {scoring_result.error}")
                        results['failed'] += 1
                        results['errors'].append({
                            'url': url,
                            'step': 'ai_scoring',
                            'error': scoring_result.error
                        })
                        continue
                        
                    # Check if AI confidence score is too low
                    ai_score = article.metadata.get('ai_confidence_score', 0.0)
                    analysis = article.metadata.get('ai_analysis', {})
                    
                    logger.info(f"AI confidence score: {ai_score:.2f} (threshold: {settings.MIN_CONFIDENCE_SCORE})")
                    if analysis:
                        logger.info(f"Analysis - Style: {analysis.get('style_score', 'N/A')}, "
                                  f"Content: {analysis.get('content_quality_score', 'N/A')}, "
                                  f"Structure: {analysis.get('structure_score', 'N/A')}")
                        if analysis.get('flags'):
                            logger.warning(f"Analysis flags: {', '.join(analysis['flags'])}")
                    
                    if ai_score < settings.MIN_CONFIDENCE_SCORE:
                        logger.warning(f"Article rejected: AI confidence score {ai_score:.2f} below threshold {settings.MIN_CONFIDENCE_SCORE}")
                        if analysis.get('reasons'):
                            logger.warning(f"Rejection reasons: {', '.join(analysis['reasons'])}")
                            
                        results['failed'] += 1
                        results['errors'].append({
                            'url': url,
                            'step': 'ai_scoring',
                            'error': f'AI confidence score too low: {ai_score:.2f}',
                            'ai_score': ai_score,
                            'analysis': analysis
                        })
                        continue
                    
                    logger.info(f"Article passed AI confidence check with score {ai_score:.2f}")
                        
                except Exception as e:
                    logger.warning(f"AI scoring failed for {url}: {e}")
                    # Continue with pipeline even if AI scoring fails
                
                # Step 3: Generate summary and topics
                try:
                    summary, topics = self.summarizer.summarize_article(article)
                    article.summary = summary
                    article.topics = topics
                except Exception as e:
                    logger.warning(f"Summarization failed for {url}: {e}")
                    article.summary = self.summarizer._create_fallback_summary(article)
                    article.topics = []
                
                # Step 3: Store in vector database
                storage_success = self.storage.store_article(article)
                
                if storage_success:
                    results['successful'] += 1
                    results['processed_articles'].append({
                        'url': url,
                        'title': article.title,
                        'summary': article.summary,
                        'topics': article.topics
                    })
                else:
                    results['failed'] += 1
                    results['errors'].append({
                        'url': url,
                        'step': 'storage',
                        'error': 'Failed to store in vector database'
                    })
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'url': url,
                    'step': 'general',
                    'error': str(e)
                })
                logger.error(f"Failed to process {url}: {e}")
        
        results['processing_time'] = time.time() - start_time
        
        logger.info(f"Pipeline processing completed: {results['successful']}/{len(urls)} successful")
        return results
    
    def search(self, query: str, limit: int = None) -> List[SearchResult]:
        """
        Search for articles using natural language query.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        logger.info(f"Searching for: '{query}'")
        return self.searcher.search(query, limit=limit)
    
    def search_by_topics(self, topics: List[str], limit: int = None) -> List[SearchResult]:
        """
        Search articles by specific topics.
        
        Args:
            topics: List of topics to search for
            limit: Maximum number of results
            
        Returns:
            List of SearchResult objects
        """
        return self.searcher.search_by_topics(topics, limit=limit)
    
    def find_similar_articles(self, article_url: str, limit: int = 5) -> List[SearchResult]:
        """
        Find articles similar to a given article.
        
        Args:
            article_url: URL of the reference article
            limit: Number of similar articles to return
            
        Returns:
            List of similar articles
        """
        return self.searcher.find_similar_articles(article_url, limit=limit)
    
    def get_trending_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending topics from stored articles.
        
        Args:
            limit: Maximum number of topics to return
            
        Returns:
            List of trending topics with counts
        """
        return self.searcher.get_trending_topics(limit=limit)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics and information.
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = self.storage.get_collection_stats()
        
        stats.update({
            'pipeline_components': {
                'extractor': 'NewsExtractor',
                'summarizer': 'AIsummarizer',
                'storage': 'VectorStorage (ChromaDB)',
                'searcher': 'SemanticSearcher'
            },
            'configuration': {
                'openai_model': settings.OPENAI_MODEL,
                'embedding_model': settings.OPENAI_EMBEDDING_MODEL,
                'collection_name': settings.CHROMADB_COLLECTION_NAME,
                'similarity_threshold': settings.SIMILARITY_THRESHOLD,
                'fallback_only_mode': settings.USE_FALLBACK_ONLY
            }
        })
        
        return stats
    
    def process_single_article(self, url: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a single article through the pipeline.
        
        Args:
            url: Article URL to process
            
        Returns:
            Tuple of (success, result_info)
        """
        try:
            result = self.process_articles([url])
            
            success = result['successful'] > 0
            info = {
                'url': url,
                'processing_time': result['processing_time'],
                'success': success
            }
            
            if success and result['processed_articles']:
                info.update(result['processed_articles'][0])
            elif result['errors']:
                info['error'] = result['errors'][0]['error']
            
            return success, info
            
        except Exception as e:
            return False, {'url': url, 'error': str(e)}
    
    def export_articles(self, format_type: str = 'json', to_file: bool = False, filename: str = None) -> str:
        """
        Export all stored articles in the specified format.
        
        Args:
            format_type: Export format ('json', 'csv', 'txt')
            to_file: Whether to write to file in data folder
            filename: Custom filename (optional, auto-generated if not provided)
            
        Returns:
            Exported data as string, or file path if written to file
        """
        import os
        import json
        import csv
        import io
        from datetime import datetime
        
        try:
            # Get all articles (simplified approach)
            all_results = self.storage.search_articles("", limit=1000)
            
            if not all_results:
                logger.warning("No articles found to export")
                return ""
            
            # Generate content based on format
            content = ""
            file_extension = ""
            
            if format_type.lower() == 'json':
                content = json.dumps(all_results, indent=2, default=str)
                file_extension = "json"
            
            elif format_type.lower() == 'csv':
                output = io.StringIO()
                fieldnames = ['title', 'summary', 'source_url', 'topics', 'extracted_at', 'body_excerpt', 'similarity_score']
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in all_results:
                    # Truncate body for CSV to keep it manageable
                    body_text = result.get('body', '')
                    body_excerpt = body_text[:500] + "..." if len(body_text) > 500 else body_text
                    
                    row = {
                        'title': result.get('title', ''),
                        'summary': result.get('summary', ''),
                        'source_url': result.get('source_url', ''),
                        'topics': ', '.join(result.get('topics', [])),
                        'extracted_at': result.get('extracted_at', ''),
                        'body_excerpt': body_excerpt.replace('\n', ' ').replace('\r', ''),
                        'similarity_score': result.get('similarity_score', 0)
                    }
                    writer.writerow(row)
                
                content = output.getvalue()
                file_extension = "csv"
            
            elif format_type.lower() == 'txt':
                # Plain text format for readability
                lines = []
                lines.append(f"News Articles Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append("=" * 80)
                lines.append(f"Total Articles: {len(all_results)}")
                lines.append("=" * 80)
                lines.append("")
                
                for i, result in enumerate(all_results, 1):
                    lines.append(f"Article {i}:")
                    lines.append(f"Title: {result.get('title', 'N/A')}")
                    lines.append(f"URL: {result.get('source_url', 'N/A')}")
                    lines.append(f"Summary: {result.get('summary', 'N/A')}")
                    lines.append(f"Topics: {', '.join(result.get('topics', []))}")
                    lines.append(f"Extracted: {result.get('extracted_at', 'N/A')}")
                    lines.append("")
                    lines.append("Original Article Text:")
                    lines.append("-" * 40)
                    body_text = result.get('body', 'N/A')
                    lines.append(body_text)
                    lines.append("-" * 80)
                    lines.append("")
                
                content = "\n".join(lines)
                file_extension = "txt"
            
            else:
                raise ValueError(f"Unsupported format: {format_type}. Supported: json, csv, txt")
            
            # Write to file if requested
            if to_file:
                # Ensure data/exports directory exists
                exports_dir = "data/exports"
                os.makedirs(exports_dir, exist_ok=True)
                
                # Generate filename if not provided
                if not filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"articles_export_{timestamp}.{file_extension}"
                elif not filename.endswith(f".{file_extension}"):
                    filename = f"{filename}.{file_extension}"
                
                file_path = os.path.join(exports_dir, filename)
                
                # Write content to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"Articles exported to: {file_path}")
                return file_path
            
            return content
                
        except Exception as e:
            logger.error(f"Failed to export articles: {e}")
            return ""
    
    def reset_storage(self) -> bool:
        """
        Reset the vector storage (delete all articles).
        
        Returns:
            True if successful, False otherwise
        """
        logger.warning("Resetting article storage...")
        return self.storage.reset_collection()
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all pipeline components.
        
        Returns:
            Dictionary with component health status
        """
        health = {}
        
        try:
            # Check OpenAI connection (not required in fallback-only mode)
            if settings.USE_FALLBACK_ONLY:
                health['openai'] = True  # Not needed in fallback mode
            else:
                # Check if API key exists and is not empty
                api_key = settings.OPENAI_API_KEY
                health['openai'] = bool(api_key and api_key.strip())
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            health['openai'] = False
        
        try:
            # Check ChromaDB
            stats = self.storage.get_collection_stats()
            health['chromadb'] = stats['total_articles'] >= 0
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            health['chromadb'] = False
        
        try:
            # Check extractor
            health['extractor'] = hasattr(self.extractor, 'extract_article')
        except Exception as e:
            logger.error(f"Extractor health check failed: {e}")
            health['extractor'] = False
            
        try:
            # Check AI scorer
            health['ai_scorer'] = hasattr(self.ai_scorer, 'score_article')
        except Exception as e:
            logger.error(f"AI scorer health check failed: {e}")
            health['ai_scorer'] = False
        
        try:
            # Check summarizer
            health['summarizer'] = hasattr(self.summarizer, 'summarize_article')
        except Exception as e:
            logger.error(f"Summarizer health check failed: {e}")
            health['summarizer'] = False
        
        health['overall'] = all(health.values())
        
        return health
