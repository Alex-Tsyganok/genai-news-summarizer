"""
Run content validation tests without pytest dependency.
"""
import os
import sys
import json
import time
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Load environment variables
load_dotenv()

from src.pipeline import NewsPipeline
from config import logger

def load_test_data():
    """Load mixed test articles data."""
    test_data_path = os.path.join("test_data", "mixed_articles.json")
    with open(test_data_path, 'r') as f:
        return json.load(f)

def validate_non_news_handling(pipeline, test_data):
    """Test handling of non-news URLs."""
    print("\n🔍 Testing non-news URL handling...")
    
    # Get only non-news URLs
    non_news_urls = [article["url"] for article in test_data 
                    if not article.get("is_valid_news", True)]
    
    print(f"Testing {len(non_news_urls)} non-news URLs...")
    results = pipeline.process_articles(non_news_urls)
    
    # Verify all were rejected
    success = results['failed'] == len(non_news_urls)
    print(f"✅ All non-news URLs rejected: {success}")
    
    # Check error messages
    for error in results['errors']:
        if error['step'] not in ['extraction', 'ai_scoring']:
            print(f"❌ Unexpected error step: {error['step']}")
            success = False
        if error['url'] not in non_news_urls:
            print(f"❌ Unexpected URL in errors: {error['url']}")
            success = False
    
    return success

def validate_news_handling(pipeline, test_data):
    """Test handling of valid news URLs."""
    print("\n🔍 Testing valid news handling...")
    
    # Get only valid news URLs
    news_urls = [article["url"] for article in test_data 
                if article.get("is_valid_news", True)]
    
    print(f"Testing {len(news_urls)} news URLs...")
    results = pipeline.process_articles(news_urls)
    
    # Verify success rate
    success_threshold = 0.7  # At least 70% should succeed
    success_rate = results['successful'] / len(news_urls)
    success = success_rate >= success_threshold
    
    print(f"Success rate: {success_rate:.1%} (threshold: {success_threshold:.1%})")
    if not success:
        print(f"❌ Success rate below threshold")
    else:
        print(f"✅ Success rate above threshold")
    
    # Check required fields
    for article in results['processed_articles']:
        missing_fields = []
        for field in ['url', 'summary', 'topics']:
            if field not in article:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Article missing fields: {', '.join(missing_fields)}")
            success = False
    
    return success

def validate_mixed_batch(pipeline, test_data):
    """Test processing a mixed batch of URLs."""
    print("\n🔍 Testing mixed batch processing...")
    
    # Get all URLs
    all_urls = [article["url"] for article in test_data]
    expected_valid_count = sum(1 for article in test_data 
                             if article.get("is_valid_news", True))
    
    print(f"Testing {len(all_urls)} URLs ({expected_valid_count} valid news)...")
    results = pipeline.process_articles(all_urls)
    
    success = True
    
    # Basic validation checks
    if results['total_urls'] != len(all_urls):
        print(f"❌ Not all URLs were processed")
        success = False
        
    if results['successful'] + results['failed'] != len(all_urls):
        print(f"❌ Total of successes and failures doesn't match URL count")
        success = False
    
    # Check success rate for valid news
    if results['successful'] < expected_valid_count * 0.7:
        print(f"❌ Success rate below threshold for valid news")
        success = False
    
    # Verify non-news URLs are in errors
    non_news_urls = {article["url"] for article in test_data 
                    if not article.get("is_valid_news", True)}
    error_urls = {error['url'] for error in results['errors']}
    
    if not non_news_urls.issubset(error_urls):
        print("❌ Some non-news URLs were not rejected")
        success = False
    
    if success:
        print("✅ Mixed batch processing passed all checks")
    
    return success

def validate_confidence_scoring(pipeline, test_data):
    """Test AI confidence scoring for different content types."""
    print("\n🔍 Testing confidence scoring...")
    
    # Get a mix of URLs (first 4 for quick test)
    urls = [article["url"] for article in test_data[:4]]
    print(f"Testing confidence scoring for {len(urls)} URLs...")
    
    results = pipeline.process_articles(urls)
    success = True
    
    for error in results['errors']:
        if 'ai_score' in error:
            score = error.get('ai_score', 1.0)
            if score >= pipeline.settings.MIN_CONFIDENCE_SCORE:
                print(f"❌ Failed article has score above threshold: {error['url']} (score: {score:.2f})")
                success = False
        
        if 'analysis' in error:
            analysis = error['analysis']
            if not isinstance(analysis, dict):
                print(f"❌ Analysis not a dictionary: {error['url']}")
                success = False
            elif not ('reasons' in analysis or 'flags' in analysis):
                print(f"❌ Analysis missing reasons/flags: {error['url']}")
                success = False
    
    if success:
        print("✅ Confidence scoring passed all checks")
    
    return success

def main():
    """Run all content validation tests."""
    print("🔬 Running Content Validation Tests")
    print("==================================")
    
    try:
        # Load test data
        test_data = load_test_data()
        print(f"✅ Loaded {len(test_data)} test articles")
        
        # Initialize pipeline
        pipeline = NewsPipeline()
        print("✅ Pipeline initialized")
        
        # Run all validations
        results = {
            "non_news_handling": validate_non_news_handling(pipeline, test_data),
            "news_handling": validate_news_handling(pipeline, test_data),
            "mixed_batch": validate_mixed_batch(pipeline, test_data),
            "confidence_scoring": validate_confidence_scoring(pipeline, test_data)
        }
        
        # Show summary
        print("\n📊 Test Results Summary")
        print("=====================")
        passed = 0
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {test_name}")
            if success:
                passed += 1
        
        print(f"\nPassed {passed} of {len(results)} tests")
        
    except Exception as e:
        print(f"❌ Error during validation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
