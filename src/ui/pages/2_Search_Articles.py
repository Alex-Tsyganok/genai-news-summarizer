"""Search Articles page."""
import pandas as pd
import streamlit as st
import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(ui_dir)
root_dir = os.path.dirname(src_dir)

for path in [root_dir, src_dir, ui_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from _shared import initialize_session_state, render_sidebar, settings


def perform_search(query: str, limit: int):
    with st.spinner(f"Searching for '{query}'..."):
        results = st.session_state.pipeline.search(query, limit=limit)
        st.session_state.search_results = results
        if results:
            st.success(f"Found {len(results)} relevant articles!")
        else:
            st.info("No articles found for this query.")


def perform_topic_search(topics: list, limit: int, min_score: float | None = None):
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


def main():
    st.set_page_config(page_title="Search Articles", page_icon="🔍")
    initialize_session_state()
    render_sidebar()

    if st.session_state.pipeline is None:
        st.info("Initialize the pipeline from the sidebar to use this page.")
        return

    st.title("🔍 Search Articles")

    search_query = st.text_input(
        "Search for articles:",
        placeholder="Enter your search query (e.g., 'artificial intelligence', 'climate change', 'technology trends')",
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

    if st.session_state.search_results:
        show_search_results()

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


if __name__ == "__main__":
    main()
