"""
Streamlit multipage app - Main container and navigation.

This serves as the entry point that sets up the multipage app structure.
All individual pages are in the pages/ subdirectory.
"""
import streamlit as st
import pandas as pd
from _shared import (
    initialize_session_state,
    render_sidebar,
    show_welcome_page,
)


def show_dashboard():
    """Show the main dashboard with stats and quick actions."""
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
        st.info("📌 Use the sidebar navigation to:")
        st.markdown("""
        - **Add Articles**: Process new news URLs
        - **Search**: Find articles by content or topics  
        - **Analytics**: View trends and export data
        - **Settings**: Configure pipeline and check health
        """)
        
        st.markdown("---")
        st.markdown("💡 **Tip**: All pages are available in the sidebar navigation!")

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="🏠 Home",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Common sidebar (below the built-in multipage nav)
    render_sidebar()

    # Main content
    if st.session_state.pipeline is None:
        show_welcome_page()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()
