"""Add Articles page."""
import os
import json
import pandas as pd
import streamlit as st
import sys
import time
from io import StringIO
import logging

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(ui_dir)
root_dir = os.path.dirname(src_dir)

for path in [root_dir, src_dir, ui_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from _shared import initialize_session_state, render_sidebar, ROOT_DIR


class StreamlitLogHandler(logging.Handler):
    """Custom log handler to capture processing logs for UI display."""
    
    def __init__(self, progress_container):
        super().__init__()
        self.progress_container = progress_container
        self.logs = []
        
    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)
        
        # Display relevant processing steps in the UI
        if "Processing article" in log_entry:
            self.progress_container.text(f"🔄 {log_entry}")
        elif "Calculating AI confidence score" in log_entry:
            self.progress_container.text(f"🤖 {log_entry}")
        elif "AI confidence score:" in log_entry:
            self.progress_container.text(f"📊 {log_entry}")
        elif "Article passed AI confidence check" in log_entry:
            self.progress_container.text(f"✅ {log_entry}")
        elif "Processing completed" in log_entry:
            self.progress_container.text(f"🎉 {log_entry}")


def process_single_url(url: str):
    """Process a single URL with interactive feedback."""
    # Create progress display containers
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with status_placeholder.container():
        st.info(f"🔄 Starting processing for: {url}")
    
    # Set up logging capture
    progress_text = progress_placeholder.empty()
    
    # Get the logger and add our custom handler
    from config import logger
    original_level = logger.level
    logger.setLevel(logging.INFO)
    
    # Create custom handler for this processing session
    handler = StreamlitLogHandler(progress_text)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    
    try:
        with st.spinner(f"Processing article from {url}..."):
            success, info = st.session_state.pipeline.process_single_article(url)
            
        # Remove our handler
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        
        # Clear progress text
        progress_text.empty()
        
        if success:
            with status_placeholder.container():
                st.success(f"✅ Successfully processed: {info.get('title', 'Article')}")
                
                # Show processing details
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Extraction Method", info.get('extraction_method', 'N/A'))
                    st.metric("Word Count", info.get('word_count', 'N/A'))
                
                with col2:
                    if 'confidence_score' in info:
                        st.metric("AI Confidence", f"{info['confidence_score']:.2f}")
                    st.metric("Topics Found", len(info.get('topics', [])))
                
                # Show summary and topics
                if info.get('summary'):
                    st.subheader("📝 Summary")
                    st.write(info['summary'])
                
                if info.get('topics'):
                    st.subheader("🏷️ Topics")
                    st.write(", ".join(info['topics']))
                    
                # Show processing log details in expandable section
                if handler.logs:
                    with st.expander("📋 Processing Details"):
                        for log_entry in handler.logs:
                            st.text(log_entry)
        else:
            with status_placeholder.container():
                st.error(f"❌ Failed to process article: {info.get('error', 'Unknown error')}")
                
                # Show processing log for debugging
                if handler.logs:
                    with st.expander("🔍 Processing Logs (for debugging)"):
                        for log_entry in handler.logs:
                            st.text(log_entry)
                            
    except Exception as e:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        progress_text.empty()
        with status_placeholder.container():
            st.error(f"❌ Processing failed with exception: {str(e)}")


essential_keys = ['total_urls', 'successful', 'failed', 'processed_articles', 'errors']


def process_multiple_urls(urls: list):
    """Process multiple URLs with detailed progress tracking."""
    st.subheader(f"🔄 Processing {len(urls)} articles...")
    
    # Create UI elements for progress tracking
    overall_progress = st.progress(0)
    current_status = st.empty()
    detailed_progress = st.empty()
    
    # Create columns for real-time stats
    stats_container = st.container()
    with stats_container:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            processed_metric = st.empty()
        with col2:
            success_metric = st.empty()
        with col3:
            failed_metric = st.empty()
        with col4:
            current_metric = st.empty()
    
    # Set up logging capture
    from config import logger
    original_level = logger.level
    logger.setLevel(logging.INFO)
    
    handler = StreamlitLogHandler(detailed_progress)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    
    # Initialize metrics
    processed_metric.metric("Processed", "0")
    success_metric.metric("Successful", "0")
    failed_metric.metric("Failed", "0")
    current_metric.metric("Current", "Initializing...")
    
    try:
        start_time = time.time()
        
        # Process articles with real-time updates
        results = {'successful': 0, 'failed': 0, 'total_urls': len(urls), 'processed_articles': [], 'errors': []}
        
        for i, url in enumerate(urls):
            # Update overall progress
            progress = (i) / len(urls)
            overall_progress.progress(progress)
            
            # Update current status
            current_status.text(f"🔄 Processing article {i+1}/{len(urls)}: {url[:50]}...")
            current_metric.metric("Current", f"Article {i+1}/{len(urls)}")
            
            # Process the article
            success, info = st.session_state.pipeline.process_single_article(url)
            
            if success:
                results['successful'] += 1
                results['processed_articles'].append(info)
                success_metric.metric("Successful", str(results['successful']))
            else:
                results['failed'] += 1
                results['errors'].append({
                    'url': url,
                    'error': info.get('error', 'Unknown error')
                })
                failed_metric.metric("Failed", str(results['failed']))
            
            # Update processed count
            processed_metric.metric("Processed", str(i + 1))
            
            # Small delay to make progress visible
            time.sleep(0.1)
        
        # Complete progress
        overall_progress.progress(1.0)
        processing_time = time.time() - start_time
        
        # Final status
        current_status.text(f"✅ Processing complete! Processed {len(urls)} articles in {processing_time:.1f}s")
        current_metric.metric("Status", "Complete!")
        
        # Store results in session state
        st.session_state.processing_results = results
        
        # Remove logging handler
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        
        # Show success summary
        st.success(f"🎉 Processed {results['successful']}/{len(urls)} articles successfully in {processing_time:.1f} seconds")
        
        # Show processing logs in expandable section
        if handler.logs:
            with st.expander("📋 Detailed Processing Logs"):
                for log_entry in handler.logs:
                    st.text(log_entry)
        
    except Exception as e:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        detailed_progress.empty()
        st.error(f"❌ Batch processing failed: {str(e)}")
        
        # Show processing logs for debugging
        if handler.logs:
            with st.expander("🔍 Processing Logs (for debugging)"):
                for log_entry in handler.logs:
                    st.text(log_entry)


def show_processing_results():
    """Display detailed processing results with enhanced UI."""
    results = st.session_state.processing_results
    if not results:
        return
        
    st.subheader("📊 Processing Results Summary")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total URLs", results['total_urls'])
    with col2:
        success_rate = (results['successful'] / results['total_urls'] * 100) if results['total_urls'] > 0 else 0
        st.metric("Successful", results['successful'], f"{success_rate:.1f}%")
    with col3:
        st.metric("Failed", results['failed'])
    with col4:
        processing_time = results.get('processing_time', 0)
        avg_time = processing_time / results['total_urls'] if results['total_urls'] > 0 else 0
        st.metric("Avg Time/Article", f"{avg_time:.1f}s")
    
    # Success/failure breakdown - Simple version
    if results['total_urls'] > 0:
        st.subheader("📈 Processing Breakdown")
        
        # Simple metrics display instead of complex table
        success_rate = (results['successful'] / results['total_urls']) * 100
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col2:
            st.metric("Successful", results['successful'])
        with col3:
            st.metric("Failed", results['failed'])
        
        # Simple bar chart
        breakdown_data = pd.DataFrame({
            'Status': ['Successful', 'Failed'],
            'Count': [results['successful'], results['failed']]
        })
        st.bar_chart(breakdown_data.set_index('Status')['Count'])
    
    # Successfully processed articles
    if results['processed_articles']:
        st.subheader("✅ Successfully Processed Articles")
        
        # Enhanced display of processed articles
        processed_df = pd.DataFrame(results['processed_articles'])
        
        # Add summary columns for better overview
        if 'summary' in processed_df.columns:
            processed_df['summary_preview'] = processed_df['summary'].str[:100] + '...'
        
        if 'topics' in processed_df.columns:
            processed_df['topics_count'] = processed_df['topics'].apply(len)
            processed_df['topics_preview'] = processed_df['topics'].apply(lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else ''))
        
        # Select display columns
        display_columns = ['title', 'url']
        if 'word_count' in processed_df.columns:
            display_columns.append('word_count')
        if 'confidence_score' in processed_df.columns:
            display_columns.append('confidence_score')
        if 'topics_count' in processed_df.columns:
            display_columns.append('topics_count')
        if 'extraction_method' in processed_df.columns:
            display_columns.append('extraction_method')
        
        # Show overview table with compact formatting for small spaces
        if display_columns:
            # Create a copy for display with better formatting
            display_df = processed_df[display_columns].copy()
            
            # Format confidence score if present
            if 'confidence_score' in display_df.columns:
                display_df['confidence_score'] = display_df['confidence_score'].round(2)
            
            # Truncate long URLs for better table display
            if 'url' in display_df.columns:
                display_df['url_display'] = display_df['url'].str[:30] + '...'
                display_columns_formatted = [col if col != 'url' else 'url_display' for col in display_columns]
                display_df = display_df[display_columns_formatted]
            
            # Truncate long titles even more for small spaces
            if 'title' in display_df.columns:
                display_df['title'] = display_df['title'].str[:40] + '...'
            
            st.dataframe(
                display_df, 
                hide_index=True,
                use_container_width=True,
                column_config={
                    "title": st.column_config.TextColumn(
                        "Title",
                        width=200
                    ),
                    "url_display": st.column_config.TextColumn(
                        "URL",
                        width=120
                    ),
                    "word_count": st.column_config.NumberColumn(
                        "Words",
                        width=60
                    ),
                    "confidence_score": st.column_config.NumberColumn(
                        "Score",
                        width=60,
                        format="%.2f"
                    ),
                    "topics_count": st.column_config.NumberColumn(
                        "Topics",
                        width=60
                    ),
                    "extraction_method": st.column_config.TextColumn(
                        "Method",
                        width=80
                    )
                }
            )
        
        # Detailed article view
        with st.expander("🔍 View Article Details"):
            for i, article in enumerate(results['processed_articles']):
                st.markdown(f"### {i+1}. {article.get('title', 'Untitled')}")
                
                # Article metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Word Count", article.get('word_count', 'N/A'))
                with col2:
                    if 'confidence_score' in article:
                        st.metric("AI Confidence", f"{article['confidence_score']:.2f}")
                    else:
                        st.metric("AI Confidence", "N/A")
                with col3:
                    st.metric("Topics", len(article.get('topics', [])))
                with col4:
                    st.metric("Method", article.get('extraction_method', 'N/A'))
                
                # Article content
                if article.get('summary'):
                    st.markdown("**Summary:**")
                    st.write(article['summary'])
                
                if article.get('topics'):
                    st.markdown("**Topics:**")
                    st.write(", ".join(article['topics']))
                
                st.markdown("**URL:**")
                st.write(article.get('url', 'N/A'))
                
                if i < len(results['processed_articles']) - 1:
                    st.divider()
    
    # Processing errors
    if results['errors']:
        st.subheader("❌ Processing Errors")
        
        errors_df = pd.DataFrame(results['errors'])
        
        # Group errors by type for better analysis - Simplified view
        if 'step' in errors_df.columns:
            error_summary = errors_df.groupby('step').size().reset_index(name='count')
            
            st.markdown("**Error Types:**")
            # Show as simple metrics instead of table + chart for small spaces
            for _, row in error_summary.iterrows():
                st.write(f"• **{row['step']}**: {row['count']} error{'s' if row['count'] > 1 else ''}")
        else:
            st.markdown("**Error Count:**")
            st.write(f"• **Total errors**: {len(results['errors'])}")
        
        # Detailed error table with compact formatting
        st.markdown("**Detailed Errors:**")
        
        # Format errors dataframe for better display
        errors_display = errors_df.copy()
        
        # Truncate even more for small spaces
        if 'url' in errors_display.columns:
            errors_display['url_short'] = errors_display['url'].str[:25] + '...'
        if 'error' in errors_display.columns:
            errors_display['error_short'] = errors_display['error'].str[:40] + '...'
        
        # Select columns for display
        display_error_cols = []
        if 'url_short' in errors_display.columns:
            display_error_cols.append('url_short')
        if 'step' in errors_display.columns:
            display_error_cols.append('step')
        if 'error_short' in errors_display.columns:
            display_error_cols.append('error_short')
        
        st.dataframe(
            errors_display[display_error_cols] if display_error_cols else errors_display, 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "url_short": st.column_config.TextColumn(
                    "URL",
                    width=150
                ),
                "step": st.column_config.TextColumn(
                    "Step",
                    width=100
                ),
                "error_short": st.column_config.TextColumn(
                    "Error",
                    width=250
                )
            }
        )
        
        # Show common error patterns
        if len(results['errors']) > 1:
            with st.expander("🔍 Error Analysis"):
                error_messages = [error.get('error', '') for error in results['errors']]
                unique_errors = list(set(error_messages))
                
                st.markdown("**Unique Error Messages:**")
                for i, error_msg in enumerate(unique_errors, 1):
                    count = error_messages.count(error_msg)
                    st.markdown(f"{i}. **{error_msg}** (occurred {count} time{'s' if count > 1 else ''})")
    
    # Processing insights
    if results['total_urls'] > 1:
        st.subheader("💡 Processing Insights")
        
        insights = []
        
        # Success rate insight
        success_rate = (results['successful'] / results['total_urls']) * 100
        if success_rate >= 90:
            insights.append("🎉 Excellent success rate! Your URLs are high quality.")
        elif success_rate >= 70:
            insights.append("👍 Good success rate. Some URLs may need attention.")
        else:
            insights.append("⚠️ Low success rate. Check URL quality and network connection.")
        
        # Error pattern insights
        if results['errors']:
            error_steps = [error.get('step', 'unknown') for error in results['errors']]
            most_common_error = max(set(error_steps), key=error_steps.count) if error_steps else None
            
            if most_common_error == 'extraction':
                insights.append("🔍 Most errors occurred during extraction. Check URL accessibility.")
            elif most_common_error == 'ai_scoring':
                insights.append("🤖 Most errors occurred during AI analysis. Check API configuration.")
            elif most_common_error == 'summarization':
                insights.append("📝 Most errors occurred during summarization. Content may be too short/long.")
        
        # Display insights
        for insight in insights:
            st.info(insight)


def main():
    st.set_page_config(page_title="📥 Add Articles", page_icon="📥")
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
