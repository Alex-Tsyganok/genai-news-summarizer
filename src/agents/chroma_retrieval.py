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
    from config import logger
    
    configuration = RetrievalConfiguration.from_runnable_config(config)
    
    # Debug: Log configuration
    logger.info(f"🔍 Agent Retriever Configuration:")
    logger.info(f"   Max results: {configuration.max_results}")
    logger.info(f"   Similarity threshold: {configuration.similarity_threshold}")
    
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
        logger.info(f"   Using similarity_score_threshold search with threshold: {configuration.similarity_threshold}")
    else:
        logger.info(f"   Using similarity search without threshold")
    
    logger.info(f"   Search type: {search_type}")
    logger.info(f"   Search kwargs: {search_kwargs}")
    
    # Create and yield the native LangChain retriever
    retriever = vstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    # Wrap the retriever to add debug logging
    class DebugRetriever:
        def __init__(self, base_retriever):
            self.base_retriever = base_retriever
        
        def invoke(self, query, config=None):
            logger.info(f"🔍 Agent Vector Search Query: '{query}'")
            results = self.base_retriever.invoke(query, config)
            logger.info(f"📊 Retrieved {len(results)} documents:")
            
            for i, doc in enumerate(results):
                title = doc.metadata.get('title', 'No title')[:60]
                similarity = getattr(doc, 'similarity', 'N/A')
                logger.info(f"   {i+1}. {title}... (similarity: {similarity})")
                
                # Log snippet of content for relevance check
                content_snippet = doc.page_content[:100].replace('\n', ' ')
                logger.info(f"      Content: {content_snippet}...")
            
            return results
        
        def __getattr__(self, name):
            # Delegate all other attributes to the base retriever
            return getattr(self.base_retriever, name)
    
    yield DebugRetriever(retriever)
