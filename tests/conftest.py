"""
Pytest configuration file for WoRMS Agent tests.
"""
import pytest
from ichatbio.agent_response import ResponseChannel, ResponseContext, ResponseMessage
from src.agent import WoRMSReActAgent


class InMemoryResponseChannel(ResponseChannel):
    """In-memory response channel for testing"""
    
    def __init__(self, message_buffer: list):
        self.message_buffer = message_buffer
    
    async def submit(self, message: ResponseMessage, context_id: str):
        """Store messages in buffer instead of sending"""
        self.message_buffer.append(message)


TEST_CONTEXT_ID = "617727d1-4ce8-4902-884c-db786854b51c"


@pytest.fixture(scope="function")
def agent():
    """Create a fresh WoRMSReActAgent instance for each test"""
    return WoRMSReActAgent()


@pytest.fixture(scope="function")
def messages() -> list[ResponseMessage]:
    """Create an empty message buffer for each test"""
    return []


@pytest.fixture(scope="function")
def context(messages) -> ResponseContext:
    """Create a ResponseContext with in-memory channel for each test"""
    return ResponseContext(InMemoryResponseChannel(messages), TEST_CONTEXT_ID)