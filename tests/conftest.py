"""
Pytest configuration file for WoRMS Agent.
"""
import importlib
from importlib.resources.abc import Traversable
import pytest
from ichatbio.agent_response import ResponseChannel, ResponseContext, ResponseMessage
from src.agent import WoRMSReActAgent


class InMemoryResponseChannel(ResponseChannel):
    """
    In-memory channel for testing without network calls.
    Stores all agent responses in a message buffer.
    
    Example:
        messages = list()
        channel = InMemoryResponseChannel(messages)
        context = ResponseContext(channel, TEST_CONTEXT_ID)
        
        agent = WoRMSReActAgent()
        await agent.run(context, "Tell me about killer whales", "research_marine_species", params)
        
        # messages now contains all agent responses
        assert len(messages) > 0
    """
    def __init__(self, message_buffer: list):
        self.message_buffer = message_buffer
    
    async def submit(self, message: ResponseMessage, context_id: str):
        self.message_buffer.append(message)


TEST_CONTEXT_ID = "617727d1-4ce8-4902-884c-db786854b51c"


@pytest.fixture(scope="function")
def agent():
    """Create a fresh WoRMS agent instance for each test."""
    return WoRMSReActAgent()


@pytest.fixture(scope="function")
def messages() -> list[ResponseMessage]:
    """During unit tests, agent replies are stored in this list."""
    return list()


@pytest.fixture(scope="function")
def context(messages) -> ResponseContext:
    """
    Test context that captures agent response messages.
    Messages outside process blocks use context_id: 617727d1-4ce8-4902-884c-db786854b51c
    """
    return ResponseContext(InMemoryResponseChannel(messages), TEST_CONTEXT_ID)


def resource(*path, text=True) -> str | Traversable:
    """
    Load test resource files from the resources/ directory.
    
    Args:
        *path: Path components to the resource file
        text: If True, returns file content as string. If False, returns Traversable
    
    Example:
        # Load mock WoRMS API response
        mock_data = resource("worms_responses", "orcinus_orca.json")
    """
    file = importlib.resources.files("resources").joinpath(*path)
    if text:
        return file.read_text()
    return file