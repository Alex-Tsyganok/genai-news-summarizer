"""
Command-line interface for the AI News Summarizer.
"""
import argparse
import sys
import os
import json
from dotenv import load_dotenv

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import NewsPipeline
from config import logger

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI News Summarizer - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py add https://example.com/news-article
  python cli.py search "artificial intelligence"
  python cli.py stats
  python cli.py export articles.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add articles command
    add_parser = subparsers.add_parser('add', help='Add articles from URLs')
    add_parser.add_argument('urls', nargs='+', help='Article URLs to process')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search articles')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Number of results')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show collection statistics')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export articles')
    export_parser.add_argument('filename', help='Output filename')
    export_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Export format')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Check pipeline health')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset all data')
    reset_parser.add_argument('--confirm', action='store_true', help='Confirm reset operation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize pipeline
    try:
        print("Initializing pipeline...")
        pipeline = NewsPipeline()
        print("✅ Pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'add':
            cmd_add_articles(pipeline, args.urls)
        elif args.command == 'search':
            cmd_search(pipeline, args.query, args.limit)
        elif args.command == 'stats':
            cmd_stats(pipeline)
        elif args.command == 'export':
            cmd_export(pipeline, args.filename, args.format)
        elif args.command == 'health':
            cmd_health(pipeline)
        elif args.command == 'reset':
            cmd_reset(pipeline, args.confirm)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def cmd_add_articles(pipeline: NewsPipeline, urls: list):
    """Add articles command."""
    print(f"Processing {len(urls)} articles...")
    
    results = pipeline.process_articles(urls)
    
    print(f"\n📊 Results:")
    print(f"  ✅ Successful: {results['successful']}")
    print(f"  ❌ Failed: {results['failed']}")
    print(f"  ⏱️ Time: {results['processing_time']:.2f}s")
    
    if results['processed_articles']:
        print(f"\n📰 Successfully processed:")
        for article in results['processed_articles']:
            print(f"  - {article['title']}")
    
    if results['errors']:
        print(f"\n❌ Errors:")
        for error in results['errors']:
            print(f"  - {error['url']}: {error['error']}")

def cmd_search(pipeline: NewsPipeline, query: str, limit: int):
    """Search command."""
    print(f"Searching for: '{query}'...")
    
    results = pipeline.search(query, limit=limit)
    
    if not results:
        print("No results found.")
        return
    
    print(f"\n🔍 Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.article.title}")
        print(f"   Score: {result.score:.3f}")
        print(f"   URL: {result.article.source_url}")
        print(f"   Summary: {result.article.summary[:100]}...")
        if result.article.topics:
            print(f"   Topics: {', '.join(result.article.topics[:5])}")

def cmd_stats(pipeline: NewsPipeline):
    """Statistics command."""
    stats = pipeline.get_statistics()
    
    print("📊 Collection Statistics:")
    print(f"  📚 Total articles: {stats.get('total_articles', 0)}")
    print(f"  🤖 AI model: {stats.get('configuration', {}).get('openai_model', 'N/A')}")
    print(f"  💾 Storage: {stats.get('collection_name', 'N/A')}")
    
    # Trending topics
    trending = pipeline.get_trending_topics(limit=10)
    if trending:
        print(f"\n🔥 Top {len(trending)} trending topics:")
        for i, topic_info in enumerate(trending, 1):
            print(f"  {i}. {topic_info['topic']} ({topic_info['count']} articles)")

def cmd_export(pipeline: NewsPipeline, filename: str, format_type: str):
    """Export command."""
    print(f"Exporting articles to {filename} ({format_type})...")
    
    data = pipeline.export_articles(format_type)
    
    if not data:
        print("No data to export.")
        return
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(data)
    
    print(f"✅ Exported to {filename}")

def cmd_health(pipeline: NewsPipeline):
    """Health check command."""
    print("🔧 Checking pipeline health...")
    
    health = pipeline.health_check()
    
    print("\n📋 Health Status:")
    for component, status in health.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component.title()}: {'OK' if status else 'Failed'}")
    
    overall_status = "Healthy" if health['overall'] else "Unhealthy"
    print(f"\n🏥 Overall Status: {overall_status}")

def cmd_reset(pipeline: NewsPipeline, confirm: bool):
    """Reset command."""
    if not confirm:
        print("⚠️ This will delete all stored articles!")
        response = input("Type 'YES' to confirm: ")
        if response != 'YES':
            print("Reset cancelled.")
            return
    
    print("🗑️ Resetting all data...")
    success = pipeline.reset_storage()
    
    if success:
        print("✅ All data has been reset successfully!")
    else:
        print("❌ Failed to reset data.")

if __name__ == "__main__":
    main()
