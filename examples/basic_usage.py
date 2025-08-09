"""
Example usage of the AI News Summarizer pipeline.
"""
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import NewsPipeline

def main():
    """Demonstrate the news summarizer pipeline."""
    # Load environment variables
    load_dotenv()
    
    # Sample news URLs for testing
    sample_urls = [
        "https://www.bbc.com/news/technology",
        "https://techcrunch.com/",
        "https://www.reuters.com/technology/",
        # Add more URLs as needed for testing
    ]
    
    print("🤖 AI News Summarizer - Example Usage")
    print("=" * 50)
    
    # Initialize the pipeline
    print("\n1. Initializing pipeline...")
    try:
        pipeline = NewsPipeline()
        print("✅ Pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Check pipeline health
    print("\n2. Checking pipeline health...")
    health = pipeline.health_check()
    for component, status in health.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {component}: {'OK' if status else 'Failed'}")
    
    if not health['overall']:
        print("❌ Pipeline health check failed. Please check your configuration.")
        return
    
    # Get current statistics
    print("\n3. Current collection statistics:")
    stats = pipeline.get_statistics()
    print(f"   📚 Total articles: {stats.get('total_articles', 0)}")
    print(f"   🤖 AI model: {stats.get('configuration', {}).get('openai_model', 'N/A')}")
    
    # Process sample articles (optional - comment out if you don't want to add real articles)
    process_articles = input("\n4. Process sample articles? (y/n): ").lower().strip() == 'y'
    
    if process_articles:
        print(f"\n   Processing {len(sample_urls)} sample articles...")
        
        # Use only the first URL for demo to avoid rate limits
        demo_urls = sample_urls[:1]
        
        results = pipeline.process_articles(demo_urls)
        
        print(f"   ✅ Successfully processed: {results['successful']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ⏱️ Processing time: {results['processing_time']:.2f} seconds")
        
        if results['processed_articles']:
            print("\n   📰 Processed articles:")
            for article in results['processed_articles']:
                print(f"   - {article['title'][:60]}...")
                print(f"     Topics: {', '.join(article['topics'][:3])}")
                print(f"     Summary: {article['summary'][:100]}...")
                print()
    
    # Demonstrate search functionality
    print("\n5. Demonstrating search functionality...")
    
    # Sample search queries
    search_queries = [
        "artificial intelligence",
        "technology news",
        "machine learning",
        "software development"
    ]
    
    for query in search_queries:
        print(f"\n   🔍 Searching for: '{query}'")
        results = pipeline.search(query, limit=3)
        
        if results:
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result.article.title[:50]}...")
                print(f"      Score: {result.score:.3f}")
                print(f"      Topics: {', '.join(result.article.topics[:3])}")
        else:
            print("   No results found.")
    
    # Show trending topics
    print("\n6. Trending topics:")
    trending = pipeline.get_trending_topics(limit=10)
    if trending:
        for i, topic_info in enumerate(trending, 1):
            print(f"   {i}. {topic_info['topic']} ({topic_info['count']} articles)")
    else:
        print("   No trending topics available.")
    
    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("- Run the Streamlit app: streamlit run src/ui/Home.py")
    print("- Add your own article URLs")
    print("- Experiment with different search queries")

if __name__ == "__main__":
    main()
