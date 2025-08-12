"""Manage ChromaDB retriever configuration for the news summarizer agent.

This module provides functionality to create and manage retrievers specifically
for ChromaDB vector store backend used in the news summarizer project.

The retriever integrates with the existing ChromaDB storage infrastructure
and supports filtering results by similarity threshold and result limits.
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from .configuration import RetrievalConfiguration


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a ChromaDB retriever for the agent, based on the current configuration."""
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from config.settings import settings
    
    configuration = RetrievalConfiguration.from_runnable_config(config)
    
    # Get ChromaDB configuration
    chromadb_config = settings.get_chromadb_config()
    
    # Create OpenAI embedding model to match the stored embeddings
    embedding_model = OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL
    )
    
    # Instead of creating a new client, use the existing collection approach
    # Get the client from the existing vector storage
    from src.storage.vector_storage import VectorStorage
    
    # Initialize vector storage which already handles the client
    vector_storage = VectorStorage()
    vector_storage._initialize_client()
    
    # Create Chroma vector store using the existing client and collection
    vstore = Chroma(
        client=vector_storage.client,
        collection_name=chromadb_config["collection_name"],
        embedding_function=embedding_model,
    )
    
    # Configure search parameters
    search_kwargs = {"k": configuration.max_results}
    search_type = "similarity"
    
    # Use similarity_score_threshold if threshold is specified
    if hasattr(configuration, 'similarity_threshold') and configuration.similarity_threshold:
        search_kwargs["score_threshold"] = configuration.similarity_threshold
        search_type = "similarity_score_threshold"
    
    # Create and yield the native LangChain retriever
    retriever = vstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    yield retriever
