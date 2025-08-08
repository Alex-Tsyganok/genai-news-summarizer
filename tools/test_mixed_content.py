"""
Demo script for testing pipeline with mixed content (news and non-news).
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

def main():
    """Run mixed content validation demo."""
    print("🔍 Testing News Pipeline with Mixed Content")
    print("==========================================")
    
    # Load test data
    try:
        with open(os.path.join("test_data", "mixed_articles.json"), 'r') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return
    
    # Initialize pipeline
    try:
        print("\n🚀 Initializing pipeline...")
        pipeline = NewsPipeline()
        print("✅ Pipeline initialized")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Process valid news articles
    print("\n📰 Testing valid news articles...")
    news_urls = [article["url"] for article in test_data if article.get("is_valid_news", True)]
    process_urls(pipeline, news_urls, "Valid News")
    
    # Process non-news URLs
    print("\n🚫 Testing non-news URLs...")
    non_news_urls = [article["url"] for article in test_data if not article.get("is_valid_news", True)]
    process_urls(pipeline, non_news_urls, "Non-News")
    
    # Process mixed batch
    print("\n🔄 Testing mixed content batch...")
    all_urls = [article["url"] for article in test_data]
    process_urls(pipeline, all_urls, "Mixed Content")

def process_urls(pipeline, urls, test_type):
    """Process a batch of URLs and display results."""
    start_time = time.time()
    results = pipeline.process_articles(urls)
    duration = time.time() - start_time
    
    print(f"\n📊 {test_type} Results:")
    print(f"  Total URLs: {len(urls)}")
    print(f"  Successful: {results['successful']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Time: {duration:.2f}s")
    
    if results['processed_articles']:
        print(f"\n✅ Successfully processed articles:")
        for article in results['processed_articles']:
            print(f"\n  📑 {article['url']}")
            print(f"     Summary: {article.get('summary', 'N/A')[:100]}...")
            if article.get('topics'):
                print(f"     Topics: {', '.join(article['topics'])}")
    
    if results['errors']:
        print(f"\n❌ Failed articles:")
        for error in results['errors']:
            print(f"\n  🔗 {error['url']}")
            print(f"     Error: {error['error']}")
            if 'ai_score' in error:
                print(f"     AI Score: {error['ai_score']:.2f}")
            if 'analysis' in error and error['analysis'].get('reasons'):
                print(f"     Reasons: {', '.join(error['analysis']['reasons'])}")

if __name__ == "__main__":
    main()
