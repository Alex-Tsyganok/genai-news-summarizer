"""
Streamlit web interface for the AI News Summarizer.
"""
import streamlit as st
import pandas as pd
import time
from datetime import datetime
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import NewsPipeline
from config import settings, logger
import dotenv

# Load environment variables
dotenv.load_dotenv()

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None

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
        
        # Navigation
        page = st.selectbox(
            "Navigate to:",
            ["🏠 Home", "📥 Add Articles", "🔍 Search Articles", "📊 Analytics", "⚙️ Settings"]
        )
        
        st.markdown("---")
        
        # Pipeline status
        st.subheader("🔧 Pipeline Status")
        
        if st.session_state.pipeline is None:
            if st.button("Initialize Pipeline"):
                st.session_state.pipeline = initialize_pipeline()
                st.experimental_rerun()
        else:
            st.success("✅ Pipeline Ready")
            
            # Quick stats
            try:
                stats = st.session_state.pipeline.get_statistics()
                st.metric("Total Articles", stats.get('total_articles', 0))
                
                if st.button("🔄 Refresh Stats"):
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"Failed to get stats: {e}")
    
    # Main content area
    if st.session_state.pipeline is None:
        show_welcome_page()
    elif page == "🏠 Home":
        show_home_page()
    elif page == "📥 Add Articles":
        show_add_articles_page()
    elif page == "🔍 Search Articles":
        show_search_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "⚙️ Settings":
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
            st.switch_page("pages/add_articles.py")
        
        if st.button("🔍 Search Articles", use_container_width=True):
            st.switch_page("pages/search.py")
        
        if st.button("📊 View Analytics", use_container_width=True):
            st.switch_page("pages/analytics.py")

def show_add_articles_page():
    """Show the add articles page."""
    st.title("📥 Add News Articles")
    
    st.markdown("Enter news article URLs to extract, summarize, and add to the database.")
    
    # URL input methods
    tab1, tab2 = st.tabs(["Single URL", "Multiple URLs"])
    
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
            
            if selected_topics and st.button("Search by Topics"):
                perform_topic_search(selected_topics, search_limit)
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

def perform_topic_search(topics: list, limit: int):
    """Perform a topic-based search."""
    with st.spinner(f"Searching for topics: {', '.join(topics)}..."):
        results = st.session_state.pipeline.search_by_topics(topics, limit=limit)
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
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as JSON"):
                export_data = st.session_state.pipeline.export_articles('json')
                if export_data:
                    st.download_button(
                        label="Download JSON",
                        data=export_data,
                        file_name=f"articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col2:
            if st.button("Export as CSV"):
                export_data = st.session_state.pipeline.export_articles('csv')
                if export_data:
                    st.download_button(
                        label="Download CSV",
                        data=export_data,
                        file_name=f"articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
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
