"""
Integration tests for the retrieval agent.

This module contains tests to verify the basic functionality of the agent graph,
including message processing, query generation, and response creation.
"""

import sys
import os
import uuid
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.retrieval_agent import agent_graph
from src.agents.configuration import RetrievalConfiguration


@pytest.mark.asyncio
async def test_agent_graph_basic_flow():
    """Test the basic flow of the agent graph with a simple query."""
    
    # Create a unique user identifier for isolation
    user_id = "test__" + uuid.uuid4().hex
    
    # Create configuration for the test
    config = RunnableConfig(
        configurable={
            "response_model": "gpt-3.5-turbo",
            "query_model": "gpt-3.5-turbo",
            "max_results": 5,
            "similarity_threshold": 0.7,
        }
    )
    
    # Test input message
    test_query = "What are the latest trends in artificial intelligence?"
    input_data = {
        "messages": [HumanMessage(content=test_query)]
    }
    
    # Execute the agent graph
    result = await agent_graph.ainvoke(input_data, config)
    
    # Verify the result structure
    assert "messages" in result
    assert len(result["messages"]) >= 1  # Should have at least the original message
    
    # Check that we have both user and AI messages
    messages = result["messages"]
    assert any(msg.type == "human" for msg in messages), "Should contain human message"
    assert any(msg.type == "ai" for msg in messages), "Should contain AI response"
    
    # Verify the AI response is not empty
    ai_messages = [msg for msg in messages if msg.type == "ai"]
    assert len(ai_messages) > 0, "Should have at least one AI response"
    
    last_ai_message = ai_messages[-1]
    assert last_ai_message.content, "AI response should not be empty"
    assert len(last_ai_message.content.strip()) > 0, "AI response should contain text"


@pytest.mark.asyncio
async def test_agent_graph_query_generation():
    """Test that the agent properly generates queries from user input."""
    
    config = RunnableConfig(
        configurable={
            "response_model": "gpt-3.5-turbo",
            "query_model": "gpt-3.5-turbo",
        }
    )
    
    # Test with a simple, direct query
    test_query = "Tell me about machine learning"
    input_data = {
        "messages": [HumanMessage(content=test_query)]
    }
    
    result = await agent_graph.ainvoke(input_data, config)
    
    # Verify that queries were generated
    assert "queries" in result
    assert len(result["queries"]) > 0, "Should generate at least one query"
    
    # For a single message, the query should be the user input
    assert test_query in result["queries"], "Should include the original user query"


@pytest.mark.asyncio
async def test_agent_graph_retrieval_placeholder():
    """Test that the retrieval step completes without errors (placeholder implementation)."""
    
    config = RunnableConfig(
        configurable={
            "max_results": 3,
            "similarity_threshold": 0.8,
        }
    )
    
    input_data = {
        "messages": [HumanMessage(content="Search for news about technology")]
    }
    
    result = await agent_graph.ainvoke(input_data, config)
    
    # Verify retrieved_docs field exists (even if empty due to placeholder)
    assert "retrieved_docs" in result
    assert isinstance(result["retrieved_docs"], list), "retrieved_docs should be a list"
    
    # In the placeholder implementation, this will be empty
    # When real retrieval is implemented, we can test for actual documents


@pytest.mark.asyncio
async def test_agent_graph_configuration():
    """Test that the agent respects different configuration options."""
    
    # Test with custom configuration
    custom_config = RunnableConfig(
        configurable={
            "response_model": "gpt-3.5-turbo",
            "query_model": "gpt-3.5-turbo",
            "max_results": 15,
            "similarity_threshold": 0.6,
            "response_system_prompt": "You are a helpful assistant that provides concise answers.",
            "query_system_prompt": "Generate search queries for news articles.",
        }
    )
    
    input_data = {
        "messages": [HumanMessage(content="What's happening in the world today?")]
    }
    
    # Should not raise any errors with custom configuration
    result = await agent_graph.ainvoke(input_data, custom_config)
    
    assert "messages" in result
    assert "queries" in result
    assert "retrieved_docs" in result


@pytest.mark.asyncio
async def test_agent_graph_multiple_messages():
    """Test the agent with a conversation history (multiple messages)."""
    
    config = RunnableConfig(
        configurable={
            "response_model": "gpt-3.5-turbo",
            "query_model": "gpt-3.5-turbo",
        }
    )
    
    # Simulate a conversation with multiple turns
    input_data = {
        "messages": [
            HumanMessage(content="Tell me about renewable energy"),
            # Normally would have AI response here, but testing just the flow
            HumanMessage(content="What about solar power specifically?")
        ]
    }
    
    result = await agent_graph.ainvoke(input_data, config)
    
    # Should handle multiple messages without errors
    assert "messages" in result
    assert len(result["messages"]) >= 2  # Original messages plus AI response
    
    # Should generate appropriate queries
    assert "queries" in result
    assert len(result["queries"]) > 0


def test_configuration_creation():
    """Test that RetrievalConfiguration can be created and used."""
    
    # Test default configuration
    config = RetrievalConfiguration()
    assert config.response_model == "gpt-3.5-turbo"
    assert config.query_model == "gpt-3.5-turbo"
    assert config.max_results == 10
    assert config.similarity_threshold == 0.7
    
    # Test custom configuration
    custom_config = RetrievalConfiguration(
        response_model="gpt-4",
        query_model="gpt-4",
        max_results=20,
        similarity_threshold=0.8,
    )
    assert custom_config.response_model == "gpt-4"
    assert custom_config.max_results == 20


def test_runnable_config_conversion():
    """Test that RetrievalConfiguration can be created from RunnableConfig."""
    
    runnable_config = RunnableConfig(
        configurable={
            "response_model": "gpt-4",
            "max_results": 15,
            "similarity_threshold": 0.85,
        }
    )
    
    config = RetrievalConfiguration.from_runnable_config(runnable_config)
    assert config.response_model == "gpt-4"
    assert config.max_results == 15
    assert config.similarity_threshold == 0.85
    
    # Default values should be used for missing fields
    assert config.query_model == "gpt-3.5-turbo"  # default value


if __name__ == "__main__":
    # Run tests directly for quick verification
    import asyncio
    
    async def run_tests():
        print("🧪 Running agent integration tests...")
        
        try:
            await test_agent_graph_basic_flow()
            print("✅ Basic flow test passed")
            
            await test_agent_graph_query_generation()
            print("✅ Query generation test passed")
            
            await test_agent_graph_retrieval_placeholder()
            print("✅ Retrieval placeholder test passed")
            
            await test_agent_graph_configuration()
            print("✅ Configuration test passed")
            
            await test_agent_graph_multiple_messages()
            print("✅ Multiple messages test passed")
            
            test_configuration_creation()
            print("✅ Configuration creation test passed")
            
            test_runnable_config_conversion()
            print("✅ RunnableConfig conversion test passed")
            
            print("\n🎉 All agent integration tests passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_tests())
