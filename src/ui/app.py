"""
Streamlit web interface for the AI News Summarizer.
"""
import streamlit as st
import pandas as pd
import time
from datetime import datetime
import sys
import os
from dotenv import load_dotenv

# Ensure both project root and src/ are on sys.path so imports work when run from any CWD
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_CURRENT_DIR)            # .../src
_ROOT_DIR = os.path.dirname(_SRC_DIR)               # repo root
for _p in (_ROOT_DIR, _SRC_DIR):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env before importing internal modules (standard approach)
try:
    load_dotenv()
except Exception:
    pass

from config.settings import Settings
from src.pipeline import NewsPipeline
from config import settings, logger

# Fail fast on invalid/missing required settings (do not log secrets)
try:
    Settings.validate()
except Exception as _e:
    # Defer showing in UI; Streamlit will present error on first interaction
    pass

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'nav_page' not in st.session_state:
        st.session_state.nav_page = "🏠 Home"
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    if 'refreshing_stats' not in st.session_state:
        st.session_state.refreshing_stats = False
    if 'last_stats_refreshed_at' not in st.session_state:
        st.session_state.last_stats_refreshed_at = None

def initialize_pipeline():
    """Initialize the pipeline with error handling."""
    try:
        if not settings.OPENAI_API_KEY:
            st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables.")
            st.stop()
        
        with st.spinner("Initializing AI News Pipeline..."):
            pipeline = NewsPipeline()
            health = pipeline.health_check()
            
            if not health['overall']:
                st.error("❌ Pipeline health check failed:")
                for component, status in health.items():
                    if not status:
                        st.error(f"- {component}: Failed")
                st.stop()
            
            st.success("✅ Pipeline initialized successfully!")
            return pipeline
            
    except Exception as e:
        st.error(f"❌ Failed to initialize pipeline: {str(e)}")
        st.stop()

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AI News Summarizer",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("📰 AI News Summarizer")
        st.markdown("---")
        
        # Navigation (state-driven): radio list for single-click selection
        _pages = ["🏠 Home", "📥 Add Articles", "🔍 Search Articles", "📊 Analytics", "⚙️ Settings"]
        selected_page = st.radio(
            "Navigate to:",
            options=_pages,
            index=_pages.index(st.session_state.get('nav_page', "🏠 Home"))
        )

        # If user changed selection, update model and rerun immediately
        if st.session_state.get('nav_page') != selected_page:
            st.session_state['nav_page'] = selected_page
            st.experimental_rerun()
        
        st.markdown("---")
        
        # Pipeline status
        st.subheader("🔧 Pipeline Status")
        
        if st.session_state.pipeline is None:
            if st.button("Initialize Pipeline"):
                st.session_state.pipeline = initialize_pipeline()
                st.experimental_rerun()
        else:
            st.success("✅ Pipeline Ready")

            # Quick stats with refresh UX
            try:
                is_refreshing = bool(st.session_state.get('refreshing_stats'))

                # Create a placeholder so the metric appears above the button
                metric_ph = st.empty()

                # Refresh button (visually below the metric)
                clicked = st.button(
                    "🔄 Refresh Stats",
                    key="btn_refresh_stats",
                    type="primary",
                    use_container_width=True,
                    disabled=is_refreshing,
                    help="Refresh collection stats"
                )

                if clicked:
                    st.session_state.refreshing_stats = True
                    is_refreshing = True

                # Fetch stats (show spinner only when refreshing)
                if is_refreshing:
                    with st.spinner("Refreshing stats..."):
                        stats = st.session_state.pipeline.get_statistics()
                        st.session_state.last_stats_refreshed_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.session_state.refreshing_stats = False
                    st.toast("Stats refreshed", icon="✅")
                else:
                    stats = st.session_state.pipeline.get_statistics()

                # Render metric above the button via the placeholder
                with metric_ph.container():
                    st.metric("Total Articles", stats.get('total_articles', 0))

                if st.session_state.last_stats_refreshed_at:
                    st.caption(f"Last refreshed: {st.session_state.last_stats_refreshed_at}")

            except Exception as e:
                st.error(f"Failed to get stats: {e}")
    
    # Main content area
    current_page = st.session_state.get('nav_page')
    if st.session_state.pipeline is None:
        show_welcome_page()
    elif current_page == "🏠 Home":
        show_home_page()
    elif current_page == "📥 Add Articles":
        show_add_articles_page()
    elif current_page == "🔍 Search Articles":
        show_search_page()
    elif current_page == "📊 Analytics":
        show_analytics_page()
    elif current_page == "⚙️ Settings":
        show_settings_page()

def show_welcome_page():
    """Show the welcome page when pipeline is not initialized."""
    st.title("🤖 AI News Summarizer & Semantic Search")
    
    st.markdown("""
    ### Welcome to the AI-Powered News Processing Pipeline!
    
    This application provides:
    
    🔍 **Smart News Extraction**: Automatically extract and parse news articles from URLs
    
    🤖 **AI Summarization**: Generate concise summaries and identify key topics using OpenAI GPT
    
    🔍 **Semantic Search**: Natural language search over processed articles using vector embeddings
    
    📊 **Analytics**: Insights and trends from your article collection
    
    ### Get Started:
    1. Set your OpenAI API key in the environment variables
    2. Click "Initialize Pipeline" in the sidebar
    3. Start adding articles and searching!
    """)
    
    st.info("💡 Click 'Initialize Pipeline' in the sidebar to get started!")

def show_home_page():
    """Show the home page with overview and quick actions."""
    st.title("🏠 Dashboard")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        stats = st.session_state.pipeline.get_statistics()
        
        with col1:
            st.metric("📚 Total Articles", stats.get('total_articles', 0))
        
        with col2:
            trending = st.session_state.pipeline.get_trending_topics(limit=5)
            st.metric("🔥 Trending Topics", len(trending))
        
        with col3:
            st.metric("🤖 AI Model", stats.get('configuration', {}).get('openai_model', 'N/A'))
        
        with col4:
            st.metric("💾 Storage", "ChromaDB")
    
    except Exception as e:
        st.error(f"Failed to load dashboard: {e}")
        return
    
    st.markdown("---")
    
    # Recent activity and trending topics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Trending Topics")
        try:
            trending = st.session_state.pipeline.get_trending_topics(limit=10)
            if trending:
                df = pd.DataFrame(trending)
                st.dataframe(df, hide_index=True)
            else:
                st.info("No trending topics available. Add some articles first!")
        except Exception as e:
            st.error(f"Failed to load trending topics: {e}")
    
    with col2:
        st.subheader("🚀 Quick Actions")
        if st.button("➕ Add New Article", use_container_width=True):
            st.session_state.nav_page = "📥 Add Articles"
            st.experimental_rerun()
        
        if st.button("🔍 Search Articles", use_container_width=True):
            st.session_state.nav_page = "🔍 Search Articles"
            st.experimental_rerun()
        
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.nav_page = "📊 Analytics"
            st.experimental_rerun()

def show_add_articles_page():
    """Show the add articles page."""
    st.title("📥 Add News Articles")
    
    st.markdown("Enter news article URLs to extract, summarize, and add to the database.")
    
    # URL input methods
    tab1, tab2, tab3 = st.tabs(["Single URL", "Multiple URLs", "Sample Data"])
    
    with tab1:
        st.subheader("Add Single Article")
        url = st.text_input("Article URL:", placeholder="https://example.com/news-article")
        
        if st.button("Process Article") and url:
            process_single_url(url)
    
    with tab2:
        st.subheader("Add Multiple Articles")
        urls_text = st.text_area(
            "Article URLs (one per line):",
            placeholder="https://example.com/article1\nhttps://example.com/article2\nhttps://example.com/article3",
            height=150
        )
        
        if st.button("Process All Articles") and urls_text:
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            process_multiple_urls(urls)

    with tab3:
        st.subheader("Use Sample Data")
        st.write("Quickly try the pipeline using built-in sample article URLs.")

        try:
            import json
            sample_path = os.path.join(_ROOT_DIR, 'data', 'sample_articles.json')
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            sample_urls = [item.get('url') for item in sample if isinstance(item, dict) and item.get('url')]

            total_samples = len(sample_urls)
            if total_samples == 0:
                st.info("No sample URLs found in data/sample_articles.json")
            else:
                st.caption(f"{total_samples} sample articles available")
                default_n = 10 if total_samples >= 10 else total_samples
                n = st.slider(
                    "Number of sample articles to process",
                    min_value=1,
                    max_value=total_samples,
                    value=default_n
                )

                if st.button("Process Sample Articles"):
                    process_multiple_urls(sample_urls[:n])

        except Exception as e:
            st.error(f"Failed to load sample data: {e}")
    
    # Show processing results
    if st.session_state.processing_results:
        show_processing_results()

def process_single_url(url: str):
    """Process a single URL."""
    with st.spinner(f"Processing article from {url}..."):
        success, info = st.session_state.pipeline.process_single_article(url)
        
        if success:
            st.success(f"✅ Successfully processed: {info.get('title', 'Article')}")
            st.json(info)
        else:
            st.error(f"❌ Failed to process article: {info.get('error', 'Unknown error')}")

def process_multiple_urls(urls: list):
    """Process multiple URLs."""
    st.write(f"Processing {len(urls)} articles...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Processing articles..."):
        results = st.session_state.pipeline.process_articles(urls)
        st.session_state.processing_results = results
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Show summary
    st.success(f"✅ Processed {results['successful']}/{results['total_urls']} articles successfully")

def show_processing_results():
    """Show detailed processing results."""
    results = st.session_state.processing_results
    
    st.subheader("📊 Processing Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total URLs", results['total_urls'])
    with col2:
        st.metric("Successful", results['successful'])
    with col3:
        st.metric("Failed", results['failed'])
    
    # Successful articles
    if results['processed_articles']:
        st.subheader("✅ Successfully Processed")
        df = pd.DataFrame(results['processed_articles'])
        st.dataframe(df, hide_index=True)
    
    # Errors
    if results['errors']:
        st.subheader("❌ Errors")
        df = pd.DataFrame(results['errors'])
        st.dataframe(df, hide_index=True)

def show_search_page():
    """Show the search page."""
    st.title("🔍 Search Articles")
    
    # Search interface
    search_query = st.text_input(
        "Search for articles:",
        placeholder="Enter your search query (e.g., 'artificial intelligence', 'climate change', 'technology trends')"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_limit = st.slider("Number of results:", 1, 20, 10)
    with col2:
        if st.button("🔍 Search"):
            if search_query:
                perform_search(search_query, search_limit)
            else:
                st.warning("Please enter a search query.")
    
    # Search results
    if st.session_state.search_results:
        show_search_results()
    
    # Topic-based search
    st.markdown("---")
    st.subheader("🏷️ Search by Topics")
    
    try:
        trending = st.session_state.pipeline.get_trending_topics(limit=20)
        if trending:
            available_topics = [t['topic'] for t in trending]
            selected_topics = st.multiselect("Select topics:", available_topics)
            min_score = st.slider("Minimum similarity score", 0.0, 1.0, float(settings.SIMILARITY_THRESHOLD), 0.01)
            
            if selected_topics and st.button("Search by Topics"):
                perform_topic_search(selected_topics, search_limit, min_score)
        else:
            st.info("No topics available. Add some articles first!")
    except Exception as e:
        st.error(f"Failed to load topics: {e}")

def perform_search(query: str, limit: int):
    """Perform a semantic search."""
    with st.spinner(f"Searching for '{query}'..."):
        results = st.session_state.pipeline.search(query, limit=limit)
        st.session_state.search_results = results
        
        if results:
            st.success(f"Found {len(results)} relevant articles!")
        else:
            st.info("No articles found for this query.")

def perform_topic_search(topics: list, limit: int, min_score: float | None = None):
    """Perform a topic-based search with optional similarity threshold filtering."""
    with st.spinner(f"Searching for topics: {', '.join(topics)}..."):
        results = st.session_state.pipeline.search_by_topics(topics, limit=limit)
        if min_score is not None:
            results = [r for r in results if getattr(r, 'score', 0) >= float(min_score)]
        st.session_state.search_results = results
        
        if results:
            st.success(f"Found {len(results)} articles matching selected topics!")
        else:
            st.info("No articles found for selected topics.")

def show_search_results():
    """Display search results."""
    st.subheader(f"📋 Search Results ({len(st.session_state.search_results)} found)")
    
    for i, result in enumerate(st.session_state.search_results, 1):
        with st.expander(f"{i}. {result.article.title}", expanded=i <= 3):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Summary:**")
                st.write(result.article.summary)
                
                if result.article.topics:
                    st.write("**Topics:**")
                    topics_str = ", ".join([f"`{topic}`" for topic in result.article.topics])
                    st.markdown(topics_str)
                
                st.write(f"**Source:** [View Article]({result.article.source_url})")
            
            with col2:
                st.metric("Similarity Score", f"{result.score:.3f}")
                st.metric("Rank", result.rank)
                
                if st.button(f"Find Similar", key=f"similar_{i}"):
                    similar_results = st.session_state.pipeline.find_similar_articles(
                        result.article.source_url, limit=5
                    )
                    if similar_results:
                        st.write("**Similar Articles:**")
                        for similar in similar_results:
                            st.write(f"- [{similar.article.title}]({similar.article.source_url})")

def show_analytics_page():
    """Show analytics and insights."""
    st.title("📊 Analytics & Insights")
    
    try:
        stats = st.session_state.pipeline.get_statistics()
        
        # Collection overview
        st.subheader("📚 Collection Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Articles", stats.get('total_articles', 0))
        with col2:
            st.metric("AI Model", stats.get('configuration', {}).get('openai_model', 'N/A'))
        with col3:
            st.metric("Embedding Model", stats.get('configuration', {}).get('embedding_model', 'N/A'))
        
        # Trending topics
        st.subheader("🔥 Trending Topics")
        trending = st.session_state.pipeline.get_trending_topics(limit=15)
        
        if trending:
            df = pd.DataFrame(trending)
            
            # Bar chart
            st.bar_chart(df.set_index('topic')['count'])
            
            # Data table
            st.dataframe(df, hide_index=True)
        else:
            st.info("No trending topics available.")
        
        # Export functionality
        st.subheader("📤 Export Data")
        
        # Export format selection
        export_format = st.selectbox(
            "Choose export format:",
            ["JSON", "CSV", "TXT"],
            help="Select the format for exporting your articles"
        )
        
        # Export destination
        export_destination = st.radio(
            "Export to:",
            ["Download (Browser)", "Save to data/exports folder"],
            help="Choose where to save the exported file"
        )
        
        # Custom filename for file export
        custom_filename = None
        if export_destination == "Save to data/exports folder":
            custom_filename = st.text_input(
                "Custom filename (optional):",
                placeholder="my_articles",
                help="Leave empty for auto-generated timestamp filename"
            )
        
        col1, col2, col3 = st.columns(3)
        
        # Export buttons
        with col1:
            if st.button(f"📥 Export as {export_format}", use_container_width=True):
                format_lower = export_format.lower()
                
                try:
                    if export_destination == "Download (Browser)":
                        # Console export for download
                        export_data = st.session_state.pipeline.export_articles(format_lower, to_file=False)
                        
                        if export_data:
                            # Determine MIME type
                            mime_types = {
                                'json': 'application/json',
                                'csv': 'text/csv',
                                'txt': 'text/plain'
                            }
                            
                            st.download_button(
                                label=f"📥 Download {export_format}",
                                data=export_data,
                                file_name=f"articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_lower}",
                                mime=mime_types.get(format_lower, 'text/plain'),
                                key=f"download_{format_lower}"
                            )
                            st.success(f"✅ {export_format} export ready for download!")
                        else:
                            st.error("❌ No data available for export")
                    
                    else:
                        # File export to data/exports
                        filename = custom_filename if custom_filename.strip() else None
                        file_path = st.session_state.pipeline.export_articles(
                            format_lower, 
                            to_file=True, 
                            filename=filename
                        )
                        
                        if file_path:
                            st.success(f"✅ Articles exported to: `{file_path}`")
                            
                            # Show file info
                            try:
                                import os
                                file_size = os.path.getsize(file_path)
                                st.info(f"📊 File size: {file_size:,} bytes")
                            except Exception:
                                pass
                        else:
                            st.error("❌ Export failed - no data available")
                            
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")
        
        with col2:
            if st.button("📋 Preview Export", use_container_width=True):
                try:
                    preview_data = st.session_state.pipeline.export_articles(export_format.lower(), to_file=False)
                    
                    if preview_data:
                        if export_format.lower() == 'txt':
                            st.text_area(
                                "Preview:",
                                value=preview_data[:2000] + ("..." if len(preview_data) > 2000 else ""),
                                height=300,
                                disabled=True
                            )
                        else:
                            st.code(
                                preview_data[:2000] + ("..." if len(preview_data) > 2000 else ""),
                                language=export_format.lower()
                            )
                        
                        st.info(f"📊 Preview showing first 2000 characters of {len(preview_data):,} total")
                    else:
                        st.error("❌ No data available for preview")
                        
                except Exception as e:
                    st.error(f"❌ Preview failed: {e}")
        
        with col3:
            if st.button("📁 View Export Files", use_container_width=True):
                try:
                    import os
                    exports_dir = "data/exports"
                    
                    if os.path.exists(exports_dir):
                        files = [f for f in os.listdir(exports_dir) if f.endswith(('.json', '.csv', '.txt'))]
                        
                        if files:
                            st.subheader("📁 Available Export Files:")
                            for file in sorted(files, reverse=True):
                                file_path = os.path.join(exports_dir, file)
                                try:
                                    file_size = os.path.getsize(file_path)
                                    st.write(f"• `{file}` ({file_size:,} bytes)")
                                except Exception:
                                    st.write(f"• `{file}`")
                        else:
                            st.info("No export files found in data/exports folder")
                    else:
                        st.info("Exports folder doesn't exist yet")
                        
                except Exception as e:
                    st.error(f"❌ Failed to list export files: {e}")
    
    except Exception as e:
        st.error(f"Failed to load analytics: {e}")

def show_settings_page():
    """Show settings and configuration."""
    st.title("⚙️ Settings & Configuration")
    
    # Pipeline health check
    st.subheader("🔧 Pipeline Health")
    
    if st.button("Run Health Check"):
        with st.spinner("Checking pipeline health..."):
            health = st.session_state.pipeline.health_check()
            
            for component, status in health.items():
                if status:
                    st.success(f"✅ {component.title()}: OK")
                else:
                    st.error(f"❌ {component.title()}: Failed")
    
    # Configuration display
    st.subheader("🔧 Current Configuration")
    
    config_data = {
        "OpenAI Model": settings.OPENAI_MODEL,
        "Embedding Model": settings.OPENAI_EMBEDDING_MODEL,
        "ChromaDB Directory": settings.CHROMADB_PERSIST_DIRECTORY,
        "Collection Name": settings.CHROMADB_COLLECTION_NAME,
        "Max Summary Length": settings.MAX_SUMMARY_LENGTH,
        "Max Topics": settings.MAX_TOPICS,
        "Similarity Threshold": settings.SIMILARITY_THRESHOLD,
    }
    
    df = pd.DataFrame(list(config_data.items()), columns=['Setting', 'Value'])
    st.dataframe(df, hide_index=True)
    
    # Danger zone
    st.subheader("⚠️ Danger Zone")
    
    st.warning("These actions are irreversible!")
    
    if st.button("🗑️ Reset All Data", type="secondary"):
        if st.session_state.get('confirm_reset', False):
            with st.spinner("Resetting all data..."):
                success = st.session_state.pipeline.reset_storage()
                if success:
                    st.success("✅ All data has been reset successfully!")
                    st.session_state.confirm_reset = False
                else:
                    st.error("❌ Failed to reset data.")
        else:
            st.session_state.confirm_reset = True
            st.warning("Click again to confirm data reset.")

if __name__ == "__main__":
    main()
