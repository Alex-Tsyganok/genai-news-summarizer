"""Analytics page."""
import os
from datetime import datetime
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

from _shared import initialize_session_state, render_sidebar


def main():
    st.set_page_config(page_title="📊 Analytics", page_icon="📊")
    initialize_session_state()
    render_sidebar()

    if st.session_state.pipeline is None:
        st.info("Initialize the pipeline from the sidebar to use this page.")
        return

    st.title("📊 Analytics & Insights")
    try:
        stats = st.session_state.pipeline.get_statistics()
        st.subheader("📚 Collection Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", stats.get('total_articles', 0))
        with col2:
            st.metric("AI Model", stats.get('configuration', {}).get('openai_model', 'N/A'))
        with col3:
            st.metric("Embedding Model", stats.get('configuration', {}).get('embedding_model', 'N/A'))

        st.subheader("🔥 Trending Topics")
        trending = st.session_state.pipeline.get_trending_topics(limit=15)
        if trending:
            df = pd.DataFrame(trending)
            # Ensure data types are consistent for Arrow compatibility
            df['topic'] = df['topic'].astype(str)
            df['count'] = df['count'].astype(int)
            st.bar_chart(df.set_index('topic')['count'])
            st.dataframe(df, hide_index=True)
        else:
            st.info("No trending topics available.")

        st.subheader("📤 Export Data")
        export_format = st.selectbox(
            "Choose export format:",
            ["JSON", "CSV", "TXT"],
            help="Select the format for exporting your articles",
        )
        export_destination = st.radio(
            "Export to:",
            ["Download (Browser)", "Save to data/exports folder"],
            help="Choose where to save the exported file",
        )

        custom_filename = None
        if export_destination == "Save to data/exports folder":
            custom_filename = st.text_input(
                "Custom filename (optional):",
                placeholder="my_articles",
                help="Leave empty for auto-generated timestamp filename",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(f"📥 Export as {export_format}", use_container_width=True):
                format_lower = export_format.lower()
                try:
                    if export_destination == "Download (Browser)":
                        export_data = st.session_state.pipeline.export_articles(format_lower, to_file=False)
                        if export_data:
                            mime_types = {
                                'json': 'application/json',
                                'csv': 'text/csv',
                                'txt': 'text/plain',
                            }
                            st.download_button(
                                label=f"📥 Download {export_format}",
                                data=export_data,
                                file_name=f"articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_lower}",
                                mime=mime_types.get(format_lower, 'text/plain'),
                                key=f"download_{format_lower}",
                            )
                            st.success(f"✅ {export_format} export ready for download!")
                        else:
                            st.error("❌ No data available for export")
                    else:
                        filename = custom_filename if (custom_filename or '').strip() else None
                        file_path = st.session_state.pipeline.export_articles(
                            format_lower, to_file=True, filename=filename
                        )
                        if file_path:
                            st.success(f"✅ Articles exported to: `{file_path}`")
                            try:
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
                                disabled=True,
                            )
                        else:
                            st.code(
                                preview_data[:2000] + ("..." if len(preview_data) > 2000 else ""),
                                language=export_format.lower(),
                            )
                        st.info(f"📊 Preview showing first 2000 characters of {len(preview_data):,} total")
                    else:
                        st.error("❌ No data available for preview")
                except Exception as e:
                    st.error(f"❌ Preview failed: {e}")

        with col3:
            if st.button("📁 View Export Files", use_container_width=True):
                try:
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


if __name__ == "__main__":
    main()
