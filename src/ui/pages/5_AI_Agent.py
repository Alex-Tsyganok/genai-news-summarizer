"""AI Agent page for interactive chat and semantic search."""
import asyncio
import os
import sys
import streamlit as st
import threading
import concurrent.futures
import queue
from datetime import datetime

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(ui_dir)
root_dir = os.path.dirname(src_dir)

for path in [root_dir, src_dir, ui_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from _shared import initialize_session_state, render_sidebar, settings


def initialize_agent_session_state():
    """Initialize agent-specific session state."""
    if 'agent_messages' not in st.session_state:
        st.session_state.agent_messages = []
    if 'agent_model' not in st.session_state:
        st.session_state.agent_model = 'gpt-3.5-turbo'
    if 'agent_max_results' not in st.session_state:
        st.session_state.agent_max_results = 5
    if 'agent_threshold' not in st.session_state:
        st.session_state.agent_threshold = float(settings.SIMILARITY_THRESHOLD)
    if 'agent_processing' not in st.session_state:
        st.session_state.agent_processing = False
    if 'agent_debug' not in st.session_state:
        st.session_state.agent_debug = True  # Enable debug by default temporarily


def run_agent_sync(query: str, model: str, max_results: int, threshold: float):
    """Synchronous wrapper for running the agent query."""
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
        
        # Try to run with asyncio.run in a simpler way
        import asyncio
        try:
            # Try direct asyncio.run first
            result = asyncio.run(agent_graph.ainvoke(input_data, config))
            return result
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                # If there's already a running loop, use threading approach
                import threading
                import queue
                
                result_queue = queue.Queue()
                exception_queue = queue.Queue()
                
                def run_in_thread():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(agent_graph.ainvoke(input_data, config))
                        result_queue.put(result)
                    except Exception as e:
                        exception_queue.put(e)
                    finally:
                        loop.close()
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join(timeout=60)  # 60 second timeout
                
                if not exception_queue.empty():
                    raise exception_queue.get()
                if not result_queue.empty():
                    return result_queue.get()
                else:
                    raise TimeoutError("Agent processing timed out")
            else:
                raise e
                
    except Exception as e:
        st.error(f"❌ Error in agent processing: {e}")
        import traceback
        with st.expander("🐛 Debug Details", expanded=False):
            st.code(traceback.format_exc())
        return None


def format_agent_response(result):
    """Format the agent response for display."""
    if not result:
        return None
    
    # Debug: Show the raw result structure
    if st.session_state.get('agent_debug', False):
        with st.expander("🐛 Raw Agent Result", expanded=False):
            st.json(result)
    
    response_data = {
        'timestamp': datetime.now(),
        'queries': [],
        'retrieved_docs': [],
        'ai_response': None
    }
    
    # Extract queries - try different possible keys
    for key in ['queries', 'query', 'search_queries']:
        if key in result and result[key]:
            response_data['queries'] = result[key] if isinstance(result[key], list) else [result[key]]
            break
    
    # Extract retrieved documents - try different possible keys
    for key in ['retrieved_docs', 'documents', 'docs', 'context']:
        if key in result and result[key]:
            response_data['retrieved_docs'] = result[key]
            break
    
    # Extract AI response from messages
    if "messages" in result and result["messages"]:
        for msg in reversed(result["messages"]):  # Start from the last message
            if hasattr(msg, 'type') and msg.type == "ai":
                response_data['ai_response'] = msg.content
                break
            elif hasattr(msg, 'content') and msg.content:
                # Fallback for different message formats
                response_data['ai_response'] = msg.content
                break
    
    # If no AI response found, try other keys
    if not response_data['ai_response']:
        for key in ['response', 'answer', 'content', 'output']:
            if key in result and result[key]:
                response_data['ai_response'] = result[key]
                break
    
    return response_data


def display_agent_message(message, is_user=True):
    """Display a chat message."""
    if is_user:
        with st.chat_message("user"):
            st.write(message['content'])
    else:
        with st.chat_message("assistant"):
            # Show AI response
            if message.get('ai_response'):
                st.write(message['ai_response'])
            
            # Show additional details in an expander
            with st.expander("📋 Search Details", expanded=False):
                # Generated queries
                if message.get('queries'):
                    st.write("**🔍 Search Queries Generated:**")
                    for i, query in enumerate(message['queries'], 1):
                        st.write(f"{i}. {query}")
                
                # Retrieved documents
                if message.get('retrieved_docs'):
                    st.write(f"\n**📚 Retrieved Documents ({len(message['retrieved_docs'])}):**")
                    for i, doc in enumerate(message['retrieved_docs'][:3], 1):  # Show first 3
                        title = doc.metadata.get('title', 'Untitled')
                        url = doc.metadata.get('source_url', '#')
                        content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                        
                        st.write(f"**{i}. [{title}]({url})**")
                        st.write(content_preview)
                        st.write("")


def display_chat_history():
    """Display the chat history."""
    for message in st.session_state.agent_messages:
        if message['role'] == 'user':
            display_agent_message(message, is_user=True)
        else:
            display_agent_message(message, is_user=False)


def main():
    st.set_page_config(page_title="🤖 AI Agent", page_icon="🤖", layout="wide")
    initialize_session_state()
    initialize_agent_session_state()
    render_sidebar()

    if st.session_state.pipeline is None:
        st.info("Initialize the pipeline from the sidebar to use this page.")
        return

    st.title("🤖 AI Agent Chat")
    st.markdown("Ask the AI agent questions about your news articles. It will search for relevant content and provide comprehensive answers.")

    # Configuration sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("🤖 Agent Configuration")
        
        st.session_state.agent_model = st.selectbox(
            "AI Model:",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0,
            help="Choose the AI model for responses"
        )
        
        st.session_state.agent_max_results = st.slider(
            "Max Results:",
            min_value=1,
            max_value=20,
            value=st.session_state.agent_max_results,
            help="Maximum number of articles to retrieve"
        )
        
        st.session_state.agent_threshold = st.slider(
            "Similarity Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.agent_threshold,
            step=0.05,
            help="Minimum similarity score for retrieved articles"
        )
        
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.agent_messages = []
            st.rerun()
        
        # Debug toggle
        st.session_state.agent_debug = st.toggle(
            "🐛 Debug Mode",
            value=st.session_state.agent_debug,
            help="Show processing status and debug information"
        )

    # Chat interface
    chat_container = st.container()
    
    # Debug information
    if st.session_state.agent_debug:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🐛 Debug Info")
        st.sidebar.write(f"Processing: {st.session_state.agent_processing}")
        st.sidebar.write(f"Messages: {len(st.session_state.agent_messages)}")
        st.sidebar.write(f"Model: {st.session_state.agent_model}")
        st.sidebar.write(f"Threshold: {st.session_state.agent_threshold}")
    
    with chat_container:
        # Display chat history
        display_chat_history()
        
        # Handle processing state for ongoing agent query
        if st.session_state.agent_processing:
            with st.chat_message("assistant"):
                with st.spinner("🤖 AI agent is thinking..."):
                    # Get the last user message to process
                    last_user_message = None
                    for msg in reversed(st.session_state.agent_messages):
                        if msg['role'] == 'user':
                            last_user_message = msg
                            break
                    
                    if last_user_message:
                        query = last_user_message['content']
                        
                        # Run the agent synchronously with proper async handling
                        try:
                            status_placeholder = st.empty()
                            status_placeholder.info("🔄 Running agent query...")
                            
                            result = run_agent_sync(
                                query,
                                st.session_state.agent_model,
                                st.session_state.agent_max_results,
                                st.session_state.agent_threshold
                            )
                            
                            status_placeholder.empty()  # Clear the status
                            
                            if result:
                                # Format and store response
                                response_data = format_agent_response(result)
                                if response_data:
                                    # Add to message history
                                    response_data['role'] = 'assistant'
                                    st.session_state.agent_messages.append(response_data)
                                    
                                    # Display the response
                                    if response_data.get('ai_response'):
                                        st.write(response_data['ai_response'])
                                    else:
                                        st.warning("No response content generated")
                                    
                                    # Show additional details in an expander
                                    with st.expander("📋 Search Details", expanded=False):
                                        # Generated queries
                                        if response_data.get('queries'):
                                            st.write("**🔍 Search Queries Generated:**")
                                            for i, q in enumerate(response_data['queries'], 1):
                                                st.write(f"{i}. {q}")
                                        
                                        # Retrieved documents
                                        if response_data.get('retrieved_docs'):
                                            st.write(f"\n**📚 Retrieved Documents ({len(response_data['retrieved_docs'])}):**")
                                            for i, doc in enumerate(response_data['retrieved_docs'][:3], 1):  # Show first 3
                                                title = doc.metadata.get('title', 'Untitled')
                                                url = doc.metadata.get('source_url', '#')
                                                content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                                                
                                                st.write(f"**{i}. [{title}]({url})**")
                                                st.write(content_preview)
                                                st.write("")
                                        else:
                                            st.write("No documents retrieved")
                                else:
                                    st.error("Failed to format agent response")
                                    st.json(result)  # Show raw result for debugging
                            else:
                                st.error("Agent returned no result")
                                
                        except Exception as e:
                            st.error(f"Error running agent: {e}")
                        
                        finally:
                            # Clear processing state
                            st.session_state.agent_processing = False
                            st.rerun()
        
        # Chat input (only show if not processing)
        if not st.session_state.agent_processing:
            if query := st.chat_input("Ask me anything about your news articles..."):
                # Add user message to history
                st.session_state.agent_messages.append({
                    'role': 'user',
                    'content': query,
                    'timestamp': datetime.now()
                })
                
                # Set processing state and trigger rerun
                st.session_state.agent_processing = True
                st.rerun()

    # Example queries section
    if not st.session_state.agent_messages and not st.session_state.agent_processing:
        st.markdown("---")
        st.subheader("💡 Example Queries")
        st.markdown("Try asking the agent questions like:")
        
        example_queries = [
            "What are the latest trends in artificial intelligence?",
            "Tell me about recent technology deals and acquisitions",
            "What news do you have about renewable energy?",
            "Summarize the most important tech news",
            "What are the emerging trends in the laptop market?"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            col = cols[i % 2]
            with col:
                if st.button(f"💬 {example}", key=f"example_{i}", use_container_width=True):
                    # Add the example query and trigger processing
                    st.session_state.agent_messages.append({
                        'role': 'user',
                        'content': example,
                        'timestamp': datetime.now()
                    })
                    st.session_state.agent_processing = True
                    st.rerun()

    # Statistics
    if st.session_state.agent_messages:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_messages = len([m for m in st.session_state.agent_messages if m['role'] == 'user'])
            st.metric("💬 User Messages", user_messages)
        
        with col2:
            agent_messages = len([m for m in st.session_state.agent_messages if m['role'] == 'assistant'])
            st.metric("🤖 Agent Responses", agent_messages)
        
        with col3:
            total_docs = sum(len(m.get('retrieved_docs', [])) for m in st.session_state.agent_messages if m['role'] == 'assistant')
            st.metric("📚 Documents Retrieved", total_docs)


if __name__ == "__main__":
    main()
