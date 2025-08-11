"""Settings page."""
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


def main():
    st.set_page_config(page_title="⚙️ Settings", page_icon="⚙️")
    initialize_session_state()
    render_sidebar()

    if st.session_state.pipeline is None:
        st.info("Initialize the pipeline from the sidebar to use this page.")
        return

    st.title("⚙️ Settings & Configuration")

    st.subheader("🔧 Pipeline Health")
    if st.button("Run Health Check"):
        with st.spinner("Checking pipeline health..."):
            health = st.session_state.pipeline.health_check()
            for component, status in health.items():
                if status:
                    st.success(f"✅ {component.title()}: OK")
                else:
                    st.error(f"❌ {component.title()}: Failed")

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
    df = pd.DataFrame(list(config_data.items()), columns=["Setting", "Value"])
    st.dataframe(df, hide_index=True)

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
