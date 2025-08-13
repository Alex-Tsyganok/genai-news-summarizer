"""Shared utilities for Streamlit UI pages.

This module centralizes environment setup, session state initialization,
pipeline initialization, and the common sidebar with pipeline status.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import streamlit as st
from dotenv import load_dotenv


# --- Path and env bootstrap -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)           # .../src
ROOT_DIR = os.path.dirname(SRC_DIR)              # repo root

def _ensure_paths():
    for p in (ROOT_DIR, SRC_DIR):
        if p and p not in sys.path:
            sys.path.insert(0, p)

def setup_paths_and_env():
    """Ensure import paths and load .env early."""
    _ensure_paths()
    try:
        load_dotenv()
    except Exception:
        pass


# Perform bootstrap before local imports that depend on paths
setup_paths_and_env()

# Now safe to import project modules
from config.settings import Settings  # noqa: E402
from config import settings, logger   # noqa: E402
from src.pipeline import NewsPipeline  # noqa: E402


# --- Session state helpers --------------------------------------------------
DEFAULT_SESSION_STATE = {
    'pipeline': None,
    'search_results': [],
    'processing_results': None,
    'refreshing_stats': False,
    'last_stats_refreshed_at': None,
}

def initialize_session_state():
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- Pipeline orchestration -------------------------------------------------
def initialize_pipeline() -> Optional[NewsPipeline]:
    """Initialize the pipeline with health checks and UI feedback."""
    
    # First, reload environment variables to catch any recent .env changes
    try:
        load_dotenv(override=True)  # Force reload of .env file
    except Exception:
        pass
    
    try:
        # Validate required settings but don't expose secrets
        try:
            Settings.validate()
        except Exception:
            # Defer to UI error below if critical
            pass

        if not settings.OPENAI_API_KEY:
            st.error("🚨 **CRITICAL ERROR: OpenAI API Key Missing!**")
            st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables or add it to a .env file.")
            st.info("💡 **Quick Fix:** Uncomment the OPENAI_API_KEY line in your .env file")
            return None

        with st.spinner("Initializing AI News Pipeline..."):
            pipeline = NewsPipeline()
            health = pipeline.health_check()

            if not health.get('overall', False):
                st.error("❌ Pipeline health check failed:")
                for component, status in health.items():
                    if component != 'overall' and not status:
                        st.error(f"- {component}: Failed")
                return None

            st.success("✅ Pipeline initialized successfully!")
            return pipeline

    except Exception as e:
        st.error(f"❌ **Pipeline Initialization Failed:** {e}")
        
        # Provide specific guidance based on error type
        error_msg = str(e).lower()
        if "api key" in error_msg or "openai" in error_msg:
            st.error("🔑 **API Key Issue Detected**")
            st.info("💡 **API Key Setup Options:**")
            st.markdown("""
            - **Option 1**: Set environment variable: `OPENAI_API_KEY=your_key_here`
            - **Option 2**: Create a `.env` file in the project root with: `OPENAI_API_KEY=your_key_here`
            - **Option 3**: Make sure the OPENAI_API_KEY line in .env is not commented out (no # at the start)
            """)
        elif "connection" in error_msg or "network" in error_msg:
            st.error("🌐 **Network Issue Detected**")
            st.info("Check your internet connection and try again.")
        else:
            st.error("🔧 **General Setup Issue**")
            st.info("Check the error message above and ensure all dependencies are installed correctly.")
        
        return None


def render_sidebar() -> bool:
    """Render the common sidebar with title and pipeline status.

    Returns True if the pipeline is ready, otherwise False.
    """
    with st.sidebar:
        st.title("📰 AI News Summarizer")
        st.markdown("---")

        # Pipeline status
        st.subheader("🔧 Pipeline Status")

        if st.session_state.pipeline is None:
            if st.button("Initialize Pipeline"):
                # Clear any previous error messages
                with st.container():
                    pipeline_result = initialize_pipeline()
                    if pipeline_result is not None:
                        st.session_state.pipeline = pipeline_result
                        st.rerun()
                    else:
                        # Ensure error is visible by adding extra spacing
                        st.error("**❌ Pipeline initialization failed!** Please check the error messages above and fix the issues before trying again.")
            return False

        # Pipeline ready path
        st.success("✅ Pipeline Ready")

        try:
            is_refreshing = bool(st.session_state.get('refreshing_stats'))

            # Placeholder for metric above button
            metric_ph = st.empty()

            # Refresh button (below the metric)
            clicked = st.button(
                "🔄 Refresh Stats",
                key="btn_refresh_stats",
                type="primary",
                use_container_width=True,
                disabled=is_refreshing,
                help="Refresh collection stats",
            )

            if clicked:
                st.session_state.refreshing_stats = True
                is_refreshing = True

            # Fetch stats
            if is_refreshing:
                with st.spinner("Refreshing stats..."):
                    stats = st.session_state.pipeline.get_statistics()
                    st.session_state.last_stats_refreshed_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.refreshing_stats = False
                st.toast("Stats refreshed", icon="✅")
            else:
                stats = st.session_state.pipeline.get_statistics()

            with metric_ph.container():
                st.metric("Total Articles", stats.get('total_articles', 0))

            if st.session_state.last_stats_refreshed_at:
                st.caption(f"Last refreshed: {st.session_state.last_stats_refreshed_at}")

        except Exception as e:
            st.error(f"Failed to get stats: {e}")

    return st.session_state.pipeline is not None


def show_welcome_page():
    st.title("🤖 AI News Summarizer & Semantic Search")
    st.markdown(
        """
        ### Welcome to the AI-Powered News Processing Pipeline!

        This application provides:

        🔍 **Smart News Extraction**: Automatically extract and parse news articles from URLs

        🤖 **AI Summarization**: Generate concise summaries and identify key topics using OpenAI GPT

        🔍 **Semantic Search**: Natural language search over processed articles using vector embeddings

        📊 **Analytics**: Insights and trends from your article collection

        ### Get Started:
        1. Set your OpenAI API key in environment variables or create a .env file
        2. Click "Initialize Pipeline" in the sidebar
        3. Start adding articles and searching!
        """
    )
    st.info("💡 Click 'Initialize Pipeline' in the sidebar to get started!")


__all__ = [
    "setup_paths_and_env",
    "initialize_session_state",
    "initialize_pipeline",
    "render_sidebar",
    "show_welcome_page",
    "ROOT_DIR",
    "settings",
    "logger",
]
