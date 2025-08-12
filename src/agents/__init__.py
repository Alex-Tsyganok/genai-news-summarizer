"""
AI Agent module for intelligent news search and analysis.

This module provides AI-powered agents for news processing, search, and analysis.
It includes LangGraph-based agents that can understand natural language queries,
perform semantic searches, and provide contextual responses based on retrieved documents.

Classes:
    AgentState: Main state object for agent conversations
    AgentInputState: Input state for external agent interactions
    RetrievalConfiguration: Configuration for agent behavior and models

Objects:
    agent_graph: Compiled LangGraph agent for news retrieval and chat

Functions:
    make_retriever: Create ChromaDB retriever with configuration

Example:
    Basic usage of the agent:
    
    >>> from src.agents import agent_graph, RetrievalConfiguration
    >>> from langchain_core.messages import HumanMessage
    >>> from langchain_core.runnables import RunnableConfig
    >>> 
    >>> # Create configuration
    >>> config = RunnableConfig(configurable={
    ...     "response_model": "gpt-3.5-turbo",
    ...     "max_results": 10
    ... })
    >>> 
    >>> # Run the agent
    >>> result = await agent_graph.ainvoke(
    ...     {"messages": [HumanMessage(content="What's new in AI?")]},
    ...     config
    ... )
    >>> print(result["messages"][-1].content)
"""

from .agent_state import AgentState, AgentInputState
from .configuration import RetrievalConfiguration
from .retrieval_agent import agent_graph
from .chroma_retrieval import make_retriever

__all__ = [
    "AgentState",
    "AgentInputState", 
    "RetrievalConfiguration",
    "agent_graph",
    "make_retriever",
]

__version__ = "0.1.0"