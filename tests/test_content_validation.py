"""
Test pipeline handling of non-news URLs and invalid content.
"""
import os
import json
import pytest
from src.pipeline import NewsPipeline
from src.models import Article

@pytest.fixture
def test_pipeline():
    """Create a pipeline instance for testing."""
    return NewsPipeline()

@pytest.fixture
def mixed_articles():
    """Load mixed test articles data."""
    test_data_path = os.path.join("test_data", "mixed_articles.json")
    with open(test_data_path, 'r') as f:
        return json.load(f)

def test_non_news_url_handling(test_pipeline, mixed_articles):
    """Test handling of non-news URLs."""
    # Get only non-news URLs
    non_news_urls = [article["url"] for article in mixed_articles 
                    if not article.get("is_valid_news", True)]
    
    # Process the non-news URLs
    results = test_pipeline.process_articles(non_news_urls)
    
    # Verify all were rejected
    assert results['failed'] == len(non_news_urls), \
        f"Expected all {len(non_news_urls)} non-news URLs to fail processing"
    
    # Check error messages
    for error in results['errors']:
        assert error['step'] in ['extraction', 'ai_scoring'], \
            f"Unexpected error step: {error['step']}"
        assert error['url'] in non_news_urls, \
            f"Unexpected URL in errors: {error['url']}"

def test_valid_news_handling(test_pipeline, mixed_articles):
    """Test handling of valid news URLs."""
    # Get only valid news URLs
    news_urls = [article["url"] for article in mixed_articles 
                if article.get("is_valid_news", True)]
    
    # Process the valid news URLs
    results = test_pipeline.process_articles(news_urls)
    
    # Verify most were processed successfully
    # Note: We use a threshold since some valid news might fail for other reasons
    success_threshold = 0.7  # At least 70% should succeed
    success_rate = results['successful'] / len(news_urls)
    
    assert success_rate >= success_threshold, \
        f"Success rate {success_rate:.2%} below threshold {success_threshold:.2%}"
    
    # Check that processed articles have required fields
    for article in results['processed_articles']:
        assert 'url' in article, "Processed article missing URL"
        assert 'summary' in article, "Processed article missing summary"
        assert 'topics' in article, "Processed article missing topics"

def test_mixed_batch_processing(test_pipeline, mixed_articles):
    """Test processing a mixed batch of URLs."""
    # Get all URLs
    all_urls = [article["url"] for article in mixed_articles]
    expected_valid_count = sum(1 for article in mixed_articles 
                             if article.get("is_valid_news", True))
    
    # Process all URLs
    results = test_pipeline.process_articles(all_urls)
    
    # Basic validation checks
    assert results['total_urls'] == len(all_urls), \
        "Not all URLs were processed"
    assert results['successful'] + results['failed'] == len(all_urls), \
        "Total of successes and failures should match URL count"
    
    # Check success rate for valid news
    assert results['successful'] >= expected_valid_count * 0.7, \
        f"Expected at least 70% of valid news ({expected_valid_count}) to succeed"
    
    # Verify all non-news URLs are in errors
    non_news_urls = {article["url"] for article in mixed_articles 
                    if not article.get("is_valid_news", True)}
    error_urls = {error['url'] for error in results['errors']}
    assert non_news_urls.issubset(error_urls), \
        "Not all non-news URLs were rejected"

def test_confidence_scoring(test_pipeline, mixed_articles):
    """Test AI confidence scoring for different content types."""
    # Get a mix of URLs
    urls = [article["url"] for article in mixed_articles[:4]]  # Take first 4 for quick test
    results = test_pipeline.process_articles(urls)
    
    for error in results['errors']:
        if 'ai_score' in error:
            # If AI scoring was performed, verify score is below threshold
            assert error.get('ai_score', 1.0) < test_pipeline.settings.MIN_CONFIDENCE_SCORE, \
                f"Failed article has score above threshold: {error['url']}"
        
        if 'analysis' in error:
            # Check if analysis contains rejection reasons
            analysis = error['analysis']
            assert isinstance(analysis, dict), \
                f"Analysis should be a dictionary: {error['url']}"
            assert 'reasons' in analysis or 'flags' in analysis, \
                f"Analysis missing reasons/flags: {error['url']}"
