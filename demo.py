"""
Comprehensive demonstration of the AI News Summarizer pipeline.
"""
import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import NewsPipeline

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print('='*60)

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

def demonstrate_pipeline():
    """Run a comprehensive demonstration of the pipeline."""
    
    print_section("AI News Summarizer - Complete Demonstration")
    
    # Initialize pipeline
    print_subsection("1. Pipeline Initialization")
    try:
        pipeline = NewsPipeline()
        print("✅ Pipeline initialized successfully!")
        
        # Health check
        health = pipeline.health_check()
        print("\n🔧 Component Health:")
        for component, status in health.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}: {'OK' if status else 'Failed'}")
        
        if not health['overall']:
            print("❌ Pipeline not healthy. Please check configuration.")
            return False
            
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return False
    
    # Display current statistics
    print_subsection("2. Current Collection Status")
    stats = pipeline.get_statistics()
    print(f"📚 Total articles in collection: {stats.get('total_articles', 0)}")
    print(f"🤖 AI Model: {stats.get('configuration', {}).get('openai_model', 'N/A')}")
    print(f"🔮 Embedding Model: {stats.get('configuration', {}).get('embedding_model', 'N/A')}")
    
    # Load sample data
    print_subsection("3. Sample Data")
    sample_file = "data/sample_articles.json"
    
    if os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            sample_data = json.load(f)
        print(f"📋 Loaded {len(sample_data)} sample articles")
        
        for article in sample_data:
            print(f"   - {article['title']}")
    else:
        print("📋 No sample data file found")
        sample_data = []
    
    # Demonstrate article processing (optional)
    process_demo = input("\n❓ Process sample articles? (y/n): ").lower().strip() == 'y'
    
    if process_demo and sample_data:
        print_subsection("4. Article Processing Demo")
        
        # Ask how many articles to process
        max_articles = len(sample_data)  # Allow processing all available articles
        num_to_process = input(f"\n📝 How many articles to process? (1-{max_articles}, or 'all'): ").strip()
        
        if num_to_process.lower() == 'all':
            articles_to_process = max_articles
        else:
            try:
                articles_to_process = min(int(num_to_process), max_articles)
            except ValueError:
                articles_to_process = 5  # Default to 5 instead of 1
        
        print(f"\n📰 Processing {articles_to_process} sample articles...")
        
        # Process multiple articles
        urls_to_process = [article['url'] for article in sample_data[:articles_to_process]]
        
        start_time = time.time()
        results = pipeline.process_articles(urls_to_process)
        processing_time = time.time() - start_time
        
        # Show results
        print(f"\n📊 Processing Results:")
        print(f"   ✅ Successful: {results['successful']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ⏱️ Total Time: {processing_time:.2f}s")
        print(f"   📈 Average Time: {processing_time/len(urls_to_process):.2f}s per article")
        
        if results['successful'] > 0:
            print(f"\n📰 Successfully processed articles:")
            for article in results['processed_articles']:
                print(f"   - {article['title']}")
                print(f"     Summary: {article['summary'][:100]}...")
                print(f"     Topics: {', '.join(article['topics'])}")
                print()
        
        if results['failed'] > 0:
            print(f"\n❌ Failed articles:")
            for error in results['errors']:
                print(f"   - {error['url']}: {error['error']}")
    
    elif not process_demo:
        print_subsection("4. Article Processing Demo")
        print("⏭️ Skipping article processing demo")
    
    # Demonstrate search functionality
    print_subsection("5. Search Functionality Demo")
    
    # Sample search queries
    search_queries = [
        "artificial intelligence",
        "machine learning technology",
        "data science news",
        "AI ethics and privacy",
        "neural networks deep learning"
    ]
    
    print("🔍 Testing various search queries:")
    for query in search_queries:
        print(f"\n   Query: '{query}'")
        results = pipeline.search(query, limit=3)
        
        if results:
            print(f"   Found {len(results)} results:")
            for result in results:
                print(f"   - {result.article.title[:60]}... (Score: {result.score:.3f})")
        else:
            print("   No results found")
    
    # Trending topics
    print_subsection("6. Trending Topics Analysis")
    trending = pipeline.get_trending_topics(limit=15)
    
    if trending:
        print("🔥 Current trending topics:")
        for i, topic_info in enumerate(trending, 1):
            print(f"   {i:2d}. {topic_info['topic']} ({topic_info['count']} articles)")
    else:
        print("📊 No trending topics available (add more articles to see trends)")
    
    # Similar articles demo
    if stats.get('total_articles', 0) > 1:
        print_subsection("7. Similar Articles Demo")
        
        # Get a random article URL for similarity search
        recent_results = pipeline.storage.search_articles("", limit=5)
        if recent_results:
            sample_url = recent_results[0].get('source_url')
            if sample_url:
                print(f"🔗 Finding articles similar to: {sample_url}")
                similar = pipeline.find_similar_articles(sample_url, limit=3)
                
                if similar:
                    print("📄 Similar articles found:")
                    for article in similar:
                        print(f"   - {article.article.title[:60]}... (Score: {article.score:.3f})")
                else:
                    print("   No similar articles found")
    
    # Export demo
    print_subsection("8. Data Export Demo")
    print("📤 Testing export functionality:")
    
    # Test console exports (brief)
    print("\n📋 Console Export Preview:")
    json_data = pipeline.export_articles('json')
    if json_data:
        print(f"   ✅ JSON: {len(json_data)} characters")
    else:
        print("   ❌ JSON: No data available")
    
    csv_data = pipeline.export_articles('csv')
    if csv_data:
        print(f"   ✅ CSV: {len(csv_data.split('\\n'))} lines")
    else:
        print("   ❌ CSV: No data available")
    
    txt_data = pipeline.export_articles('txt')
    if txt_data:
        print(f"   ✅ TXT: {len(txt_data)} characters")
    else:
        print("   ❌ TXT: No data available")
    
    # Test file exports
    print("\n💾 File Export Test:")
    try:
        # Export to files in data/exports folder
        json_file = pipeline.export_articles('json', to_file=True, filename='demo_export')
        if json_file:
            print(f"   ✅ JSON file: {json_file}")
        
        csv_file = pipeline.export_articles('csv', to_file=True, filename='demo_export')
        if csv_file:
            print(f"   ✅ CSV file: {csv_file}")
            
        txt_file = pipeline.export_articles('txt', to_file=True, filename='demo_export')
        if txt_file:
            print(f"   ✅ TXT file: {txt_file}")
    except Exception as e:
        print(f"   ❌ File export failed: {e}")
    
    print("\n📁 Available via CLI:")
    print("   python cli.py export --format json")
    print("   python cli.py export my_file --format csv")
    print("   python cli.py export --format txt --console")
    
    # Final statistics
    print_subsection("9. Final Statistics")
    final_stats = pipeline.get_statistics()
    print(f"📊 Collection summary:")
    print(f"   📚 Total articles: {final_stats.get('total_articles', 0)}")
    print(f"   💾 Storage location: {final_stats.get('persist_directory', 'N/A')}")
    print(f"   🔧 Collection name: {final_stats.get('collection_name', 'N/A')}")
    
    # Usage recommendations
    print_section("💡 Next Steps & Usage Recommendations")
    print("""
🌐 Web Interface:
   streamlit run src/ui/app.py
   
⌨️  Command Line:
   python cli.py add <urls>
   python cli.py search "your query"
   python cli.py export --format json
   python cli.py export my_file --format csv
   python cli.py stats
   
📚 Python API:
   from src.pipeline import NewsPipeline
   pipeline = NewsPipeline()
   
🔧 Configuration:
   Edit config/settings.py for advanced options
   
📊 Real-world Usage:
   - Add news RSS feeds
   - Integrate with news APIs
   - Set up automated processing
   - Build custom search interfaces
    """)
    
    print_section("🎉 Demonstration Complete!")
    print("The AI News Summarizer pipeline is ready for production use!")
    
    return True

if __name__ == "__main__":
    success = demonstrate_pipeline()
    sys.exit(0 if success else 1)
