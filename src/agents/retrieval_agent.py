"""Main entrypoint for the conversational agent.

This module defines the core structure and functionality of the conversational
agent. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from .agent_state import AgentInputState, AgentState
from .configuration import RetrievalConfiguration
from . import chroma_retrieval

# Define the function that calls the model


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


async def generate_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query
    based on the conversation context.

    Args:
        state (AgentState): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query
          based on the conversation history and context.
    """
    from langchain_openai import ChatOpenAI
    
    messages = state.messages
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = str(messages[-1].content)  # Simplified message text extraction
        return {"queries": [human_input]}
    else:
        # Multi-turn conversation - use LLM to generate refined query
        configuration = RetrievalConfiguration.from_runnable_config(config)
        
        # Initialize the language model for query generation
        llm = ChatOpenAI(
            model=configuration.query_model,
            temperature=0.0,  # Deterministic query generation
            max_tokens=150,   # Concise query generation
        )
        
        # Create conversation context
        conversation_context = "\n".join([
            f"{'User' if i % 2 == 0 else 'Assistant'}: {msg.content}"
            for i, msg in enumerate(messages)
        ])
        
        # Create the query generation prompt
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", configuration.query_system_prompt + """

Based on the conversation history, generate a focused search query that will help find relevant news articles to answer the user's latest question. The query should:

1. Extract the main topics and keywords from the latest user message
2. Consider the conversation context for additional relevant terms
3. Be specific enough to find relevant articles but broad enough to get good results
4. Focus on newsworthiness - what would be in news articles about this topic

Return only the search query, no additional text or explanation."""),
            ("human", """Conversation History:
{conversation}

Generate a search query to find relevant news articles for the user's latest question.""")
        ])
        
        try:
            # Generate the refined query
            chain = query_prompt | llm
            response = await chain.ainvoke({
                "conversation": conversation_context
            })
            
            # Extract the query from the response
            refined_query = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            return {"queries": [refined_query]}
            
        except Exception as e:
            # Fallback to using the latest message directly
            human_input = str(messages[-1].content)
            return {"queries": [human_input]}


async def retrieve(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the ChromaDB retriever, and returns
    the retrieved documents.

    Args:
        state (AgentState): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    # Use the ChromaDB retriever to get relevant documents
    with chroma_retrieval.make_retriever(config) as retriever:
        # Get the latest query from the state
        latest_query = state.queries[-1] if state.queries else ""
        
        # Retrieve documents using the ChromaDB retriever
        response = await retriever.ainvoke(latest_query, config)
        
        return {"retrieved_docs": response}


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a comprehensive response using retrieved documents and conversation context.
    
    This function uses the language model to synthesize information from retrieved documents
    and generate a contextual, informative response to the user's query.
    
    Args:
        state (AgentState): The current state containing messages, queries, and retrieved documents.
        config (RunnableConfig): Configuration containing model settings and parameters.
        
    Returns:
        dict[str, list[BaseMessage]]: A dictionary with a 'messages' key containing the AI response.
    """
    from langchain_openai import ChatOpenAI
    
    configuration = RetrievalConfiguration.from_runnable_config(config)
    
    # Initialize the language model
    llm = ChatOpenAI(
        model=configuration.response_model,
        temperature=0.1,  # Lower temperature for more focused, factual responses
        max_tokens=1000,  # Reasonable limit for response length
    )
    
    # Prepare context from retrieved documents
    document_context = ""
    if state.retrieved_docs:
        document_context = "\n\n".join([
            f"**Source: {doc.metadata.get('title', 'Unknown')}**\n"
            f"URL: {doc.metadata.get('source_url', 'No URL')}\n"
            f"Summary: {doc.metadata.get('summary', doc.page_content[:200] + '...')}\n"
            f"Topics: {doc.metadata.get('topics', 'No topics')}"
            for doc in state.retrieved_docs
        ])
    else:
        document_context = "No relevant documents were found for this query."
    
    # Get the latest user message for context
    user_query = ""
    if state.messages:
        user_query = str(state.messages[-1].content)
    
    # Create the response prompt
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", configuration.response_system_prompt + """

Based on the retrieved news articles provided below, generate a comprehensive and informative response to the user's question. Your response should:

1. **Synthesize Information**: Combine insights from multiple sources when relevant
2. **Cite Sources**: Reference specific articles by title when mentioning facts
3. **Be Objective**: Present information factually without bias
4. **Structure Clearly**: Use clear sections and bullet points when helpful
5. **Acknowledge Limitations**: If the retrieved documents don't fully answer the question, say so

**Important Guidelines:**
- Only use information from the provided documents
- If no relevant documents are available, explain this clearly
- Provide specific details and examples from the articles
- Maintain a helpful, informative tone
- Include URLs when referencing specific articles
"""),
        ("human", """User Question: {query}

Retrieved Articles:
{context}

Please provide a comprehensive response based on the above information.""")
    ])
    
    # Generate the response
    try:
        # Create the chain
        chain = response_prompt | llm
        
        # Invoke the chain with the context
        response = await chain.ainvoke({
            "query": user_query,
            "context": document_context
        })
        
        # Extract the content from the response
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Create AI message with the generated response
        ai_response = AIMessage(content=response_content)
        
        return {"messages": [ai_response]}
        
    except Exception as e:
        # Fallback response if LLM call fails
        error_response = AIMessage(
            content=f"I apologize, but I encountered an error while generating a response: {str(e)}\n\n"
                   f"However, I found {len(state.retrieved_docs)} relevant articles for your query about '{user_query}'. "
                   f"You may want to try rephrasing your question or check the individual article sources."
        )
        return {"messages": [error_response]}


# Define a new graph (It's just a pipe)

builder = StateGraph(AgentState, input_schema=AgentInputState)

builder.add_node(generate_query)
builder.add_node(retrieve)
builder.add_node(respond)
builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "respond")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
agent_graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
agent_graph.name = "NewsAgent"