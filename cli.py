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

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI News Summarizer - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py add https://example.com/news-article
  python cli.py search "artificial intelligence" --limit 5
  python cli.py agent "What are the latest trends in AI?" --model gpt-4
  python cli.py agent "Tell me about renewable energy" --max-results 5 --threshold 0.8
  python cli.py stats
  python cli.py export --format json
  python cli.py export my_articles --format csv
  python cli.py export --format txt --console
  python cli.py health
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
    export_parser = subparsers.add_parser('export', help='Export articles to data/exports folder')
    export_parser.add_argument('filename', nargs='?', help='Output filename (optional, auto-generated if not provided)')
    export_parser.add_argument('--format', choices=['json', 'csv', 'txt'], default='json', help='Export format')
    export_parser.add_argument('--console', action='store_true', help='Print to console instead of file')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Check pipeline health')
    
    # Agent command
    agent_parser = subparsers.add_parser('agent', help='Run AI agent for interactive chat')
    agent_parser.add_argument('query', help='Query to ask the agent')
    agent_parser.add_argument('--model', default='gpt-3.5-turbo', help='AI model to use')
    agent_parser.add_argument('--max-results', type=int, default=10, help='Maximum results to retrieve')
    agent_parser.add_argument('--threshold', type=float, default=0.3, help='Similarity threshold (0.0-1.0)')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset all data')
    reset_parser.add_argument('--confirm', action='store_true', help='Confirm reset operation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load environment variables FIRST using standard approach
    load_dotenv()
    
    # Now import after environment is loaded
    from src.pipeline import NewsPipeline
    from config import logger
    
    # Initialize pipeline (fail fast if config invalid)
    try:
        try:
            from config.settings import Settings
            Settings.validate()
        except Exception:
            pass
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
            cmd_export(pipeline, args.filename, args.format, args.console)
        elif args.command == 'health':
            cmd_health(pipeline)
        elif args.command == 'agent':
            import asyncio
            asyncio.run(cmd_agent(pipeline, args.query, args.model, args.max_results, args.threshold))
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

def cmd_add_articles(pipeline, urls: list):
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

def cmd_search(pipeline, query: str, limit: int):
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

def cmd_stats(pipeline):
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

def cmd_export(pipeline, filename: str, format_type: str, console_output: bool = False):
    """Export command with enhanced functionality."""
    if console_output:
        print(f"📤 Exporting articles to console ({format_type})...")
        data = pipeline.export_articles(format_type, to_file=False)
        
        if not data:
            print("❌ No data to export.")
            return
        
        print("\n" + "="*80)
        print(data)
        print("="*80)
        print(f"✅ Export completed ({format_type} format)")
    else:
        print(f"📤 Exporting articles to data/exports folder ({format_type})...")
        
        # Use the new export_articles with file writing capability
        result = pipeline.export_articles(format_type, to_file=True, filename=filename)
        
        if not result:
            print("❌ No data to export or export failed.")
            return
        
        print(f"✅ Articles exported to: {result}")
        
        # Show summary
        try:
            import os
            file_size = os.path.getsize(result)
            print(f"📊 File size: {file_size:,} bytes")
        except Exception:
            pass

def cmd_health(pipeline):
    """Health check command."""
    print("🔧 Checking pipeline health...")
    
    health = pipeline.health_check()
    
    print("\n📋 Health Status:")
    for component, status in health.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component.title()}: {'OK' if status else 'Failed'}")
    
    overall_status = "Healthy" if health['overall'] else "Unhealthy"
    print(f"\n🏥 Overall Status: {overall_status}")

async def cmd_agent(pipeline, query: str, model: str, max_results: int, threshold: float):
    """Agent command - run AI agent with query."""
    print(f"🤖 Running AI agent...")
    print(f"   Query: {query}")
    print(f"   Model: {model}")
    print(f"   Max results: {max_results}")
    print(f"   Similarity threshold: {threshold}")
    
    try:
        # Import agent components
        from src.agents import agent_graph
        from langchain_core.messages import HumanMessage
        from langchain_core.runnables import RunnableConfig
        
        # Create configuration
        config = RunnableConfig(configurable={
            "response_model": model,
            "query_model": model,
            "max_results": max_results,
            "similarity_threshold": threshold,
        })
        
        # Prepare input
        input_data = {
            "messages": [HumanMessage(content=query)]
        }
        
        print("\n🔍 Processing query...")
        
        # Run the agent
        result = await agent_graph.ainvoke(input_data, config)
        
        # Display results
        print("\n" + "="*80)
        print("🤖 AI Agent Response:")
        print("="*80)
        
        # Show generated queries
        if "queries" in result and result["queries"]:
            print(f"\n🔍 Search Queries Generated:")
            for i, q in enumerate(result["queries"], 1):
                print(f"  {i}. {q}")
        
        # Show retrieved documents
        if "retrieved_docs" in result and result["retrieved_docs"]:
            print(f"\n📚 Retrieved Documents ({len(result['retrieved_docs'])}):")
            for i, doc in enumerate(result["retrieved_docs"][:5], 1):  # Show first 5
                print(f"  {i}. {doc.metadata.get('title', 'Untitled')}")
                print(f"     {doc.page_content[:100]}...")
                if i < len(result["retrieved_docs"]):
                    print()
        
        # Show AI response
        if "messages" in result:
            ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
            if ai_messages:
                print(f"\n💬 Agent Response:")
                print(ai_messages[-1].content)
        
        print("\n" + "="*80)
        print("✅ Agent processing completed!")
        
    except Exception as e:
        print(f"❌ Agent error: {e}")
        import traceback
        print("\nDebug traceback:")
        traceback.print_exc()

def cmd_reset(pipeline, confirm: bool):
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
