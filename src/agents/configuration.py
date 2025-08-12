"""Define the configurable parameters for the retrieval agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from config import settings


@dataclass(kw_only=True)
class RetrievalConfiguration:
    """Simplified configuration class for the retrieval agent.
    
    This class provides minimal configuration needed for the retrieval agent
    while integrating with the existing project settings system.
    """

    # Response generation
    response_system_prompt: str = field(
        default="You are a helpful AI assistant that answers questions based on the provided news articles. Use the retrieved documents to provide accurate, informative responses.",
        metadata={"description": "The system prompt used for generating responses."},
    )

    response_model: str = field(
        default="gpt-3.5-turbo",
        metadata={"description": "The language model used for generating responses."},
    )

    # Query processing
    query_system_prompt: str = field(
        default="Generate a search query based on the conversation history. Focus on the main topics and keywords that would help find relevant news articles.",
        metadata={"description": "The system prompt used for processing and refining queries."},
    )

    query_model: str = field(
        default="gpt-3.5-turbo",
        metadata={"description": "The language model used for processing and refining queries."},
    )

    # Search parameters
    max_results: int = field(
        default=10,
        metadata={"description": "Maximum number of documents to retrieve."},
    )

    similarity_threshold: float = field(
        default_factory=lambda: settings.SIMILARITY_THRESHOLD,
        metadata={"description": "Minimum similarity score for retrieved documents."},
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create a RetrievalConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of RetrievalConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        
        # Get field names from the dataclass
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        
        # Filter configurable items to only include valid fields
        filtered_config = {k: v for k, v in configurable.items() if k in field_names}
        
        return cls(**filtered_config)

    @classmethod
    def from_project_settings(cls, settings) -> T:
        """Create configuration from existing project settings.
        
        Args:
            settings: The project settings object
            
        Returns:
            T: An instance of RetrievalConfiguration
        """
        return cls(
            response_model=getattr(settings, 'OPENAI_MODEL', 'gpt-3.5-turbo'),
            query_model=getattr(settings, 'OPENAI_MODEL', 'gpt-3.5-turbo'),
            max_results=getattr(settings, 'DEFAULT_SEARCH_LIMIT', 10),
            similarity_threshold=getattr(settings, 'SIMILARITY_THRESHOLD', 0.7),
        )


T = TypeVar("T", bound=RetrievalConfiguration)
