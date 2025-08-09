"""Add Articles page."""
import os
import json
import pandas as pd
import streamlit as st
import sys

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(ui_dir)
root_dir = os.path.dirname(src_dir)

for path in [root_dir, src_dir, ui_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from _shared import initialize_session_state, render_sidebar, ROOT_DIR


def process_single_url(url: str):
    with st.spinner(f"Processing article from {url}..."):
        success, info = st.session_state.pipeline.process_single_article(url)
        if success:
            st.success(f"✅ Successfully processed: {info.get('title', 'Article')}")
            st.json(info)
        else:
            st.error(f"❌ Failed to process article: {info.get('error', 'Unknown error')}")


essential_keys = ['total_urls', 'successful', 'failed', 'processed_articles', 'errors']


def process_multiple_urls(urls: list):
    st.write(f"Processing {len(urls)} articles...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    with st.spinner("Processing articles..."):
        results = st.session_state.pipeline.process_articles(urls)
        st.session_state.processing_results = results
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    st.success(f"✅ Processed {results['successful']}/{results['total_urls']} articles successfully")


def show_processing_results():
    results = st.session_state.processing_results
    if not results:
        return
    st.subheader("📊 Processing Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total URLs", results['total_urls'])
    with col2:
        st.metric("Successful", results['successful'])
    with col3:
        st.metric("Failed", results['failed'])
    if results['processed_articles']:
        st.subheader("✅ Successfully Processed")
        df = pd.DataFrame(results['processed_articles'])
        st.dataframe(df, hide_index=True)
    if results['errors']:
        st.subheader("❌ Errors")
        df = pd.DataFrame(results['errors'])
        st.dataframe(df, hide_index=True)


def main():
    st.set_page_config(page_title="Add Articles", page_icon="📥")
    initialize_session_state()
    render_sidebar()

    if st.session_state.pipeline is None:
        st.info("Initialize the pipeline from the sidebar to use this page.")
        return

    st.title("📥 Add News Articles")
    st.markdown("Enter news article URLs to extract, summarize, and add to the database.")

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
            height=150,
        )
        if st.button("Process All Articles") and urls_text:
            urls = [u.strip() for u in urls_text.split("\n") if u.strip()]
            process_multiple_urls(urls)

    with tab3:
        st.subheader("Use Sample Data")
        st.write("Quickly try the pipeline using built-in sample article URLs.")
        try:
            sample_path = os.path.join(ROOT_DIR, 'data', 'sample_articles.json')
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            sample_urls = [item.get('url') for item in sample if isinstance(item, dict) and item.get('url')]
            total_samples = len(sample_urls)
            if total_samples == 0:
                st.info("No sample URLs found in data/sample_articles.json")
            else:
                st.caption(f"{total_samples} sample articles available")
                default_n = 10 if total_samples >= 10 else total_samples
                n = st.slider("Number of sample articles to process", 1, total_samples, default_n)
                if st.button("Process Sample Articles"):
                    process_multiple_urls(sample_urls[:n])
        except Exception as e:
            st.error(f"Failed to load sample data: {e}")

    if st.session_state.processing_results:
        show_processing_results()


if __name__ == "__main__":
    main()
