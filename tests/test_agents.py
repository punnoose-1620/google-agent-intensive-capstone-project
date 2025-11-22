"""
Tests for Agent Framework - Coordinator and Agent base classes.

Tests agent registration, message routing, and communication.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent_framework import Agent, Coordinator
from typing import Dict, Any, Optional


class EchoAgent(Agent):
    """
    Simple test agent that echoes back the message payload.
    """
    
    def handle_message(self, message: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Echo back the message payload."""
        action = message.get("action", "unknown")
        payload = message.get("payload", {})
        request_id = message.get("request_id", "unknown")
        
        return {
            "status": "ok",
            "payload": {
                "echo": payload,
                "action": action,
                "original_request_id": request_id
            },
            "request_id": request_id,
            "meta": {
                "agent": self.name
            }
        }


def mock_gemini_response(json_string: str):
    """
    Mock helper that simulates Gemini client returning a JSON string.
    
    This can be used to mock google.generativeai.GenerativeModel for testing
    without making actual API calls.
    
    Args:
        json_string: JSON string to return as the response
        
    Returns:
        Mock response object with .text attribute containing the JSON string
        
    Example:
        >>> mock_response = mock_gemini_response('{"summary": "test", "steps": []}')
        >>> assert mock_response.text == '{"summary": "test", "steps": []}'
    """
    class MockResponse:
        def __init__(self, text: str):
            self.text = text
    
    return MockResponse(json_string)


def test_echo_agent_creation():
    """Test that EchoAgent can be created."""
    agent = EchoAgent(name="test_echo")
    assert agent.name == "test_echo"
    assert isinstance(agent, Agent)


def test_echo_agent_handle_message():
    """Test that EchoAgent correctly handles messages."""
    agent = EchoAgent(name="test_echo")
    
    message = {
        "action": "echo",
        "payload": {"test": "data"},
        "request_id": "test-001"
    }
    
    response = agent.handle_message(message)
    
    assert response["status"] == "ok"
    assert response["payload"]["echo"] == {"test": "data"}
    assert response["payload"]["action"] == "echo"
    assert response["request_id"] == "test-001"


def test_coordinator_creation():
    """Test that Coordinator can be created."""
    coordinator = Coordinator(timeout=10.0)
    assert coordinator.timeout == 10.0
    assert len(coordinator.list_agents()) == 0


def test_coordinator_register_agent():
    """Test that agents can be registered with Coordinator."""
    coordinator = Coordinator()
    agent = EchoAgent(name="echo_agent")
    
    coordinator.register(agent)
    
    assert "echo_agent" in coordinator.list_agents()
    assert len(coordinator.list_agents()) == 1


def test_coordinator_register_duplicate_agent():
    """Test that registering duplicate agent names raises ValueError."""
    coordinator = Coordinator()
    agent1 = EchoAgent(name="echo_agent")
    agent2 = EchoAgent(name="echo_agent")
    
    coordinator.register(agent1)
    
    try:
        coordinator.register(agent2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "already registered" in str(e)


def test_coordinator_send_message():
    """Test that Coordinator can send messages between agents."""
    coordinator = Coordinator()
    agent = EchoAgent(name="echo_agent")
    
    coordinator.register(agent)
    
    message = {
        "action": "echo",
        "payload": {"message": "hello"},
        "request_id": "test-002"
    }
    
    response = coordinator.send(
        from_agent="echo_agent",
        to_agent="echo_agent",
        message=message
    )
    
    assert response["status"] == "ok"
    assert response["payload"]["echo"]["message"] == "hello"
    assert response["payload"]["action"] == "echo"
    assert "request_id" in response


def test_coordinator_send_to_unregistered_agent():
    """Test that sending to unregistered agent raises KeyError."""
    coordinator = Coordinator()
    agent = EchoAgent(name="echo_agent")
    coordinator.register(agent)
    
    message = {
        "action": "echo",
        "payload": {},
        "request_id": "test-003"
    }
    
    try:
        coordinator.send(
            from_agent="echo_agent",
            to_agent="unknown_agent",
            message=message
        )
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "not registered" in str(e)


def test_coordinator_auto_generate_request_id():
    """Test that Coordinator auto-generates request_id if missing."""
    coordinator = Coordinator()
    agent = EchoAgent(name="echo_agent")
    coordinator.register(agent)
    
    message = {
        "action": "echo",
        "payload": {}
        # No request_id
    }
    
    response = coordinator.send(
        from_agent="echo_agent",
        to_agent="echo_agent",
        message=message
    )
    
    assert "request_id" in response
    assert response["request_id"] is not None
    assert len(response["request_id"]) > 0


def test_mock_gemini_response():
    """Test the mock Gemini response helper."""
    json_string = '{"summary": "test summary", "steps": ["step1", "step2"]}'
    mock_response = mock_gemini_response(json_string)
    
    assert mock_response.text == json_string
    assert isinstance(mock_response.text, str)


def test_mock_gemini_response_with_complex_json():
    """Test mock Gemini response with complex JSON structure."""
    complex_json = '''{
        "summary": "Complex test",
        "step_by_step": ["Step 1", "Step 2"],
        "examples": [{"title": "Example 1", "description": "Test"}],
        "difficulties": ["Difficulty 1"]
    }'''
    
    mock_response = mock_gemini_response(complex_json)
    assert mock_response.text == complex_json
    
    # Verify it's valid JSON
    import json
    parsed = json.loads(mock_response.text)
    assert parsed["summary"] == "Complex test"
    assert len(parsed["step_by_step"]) == 2


if __name__ == "__main__":
    # Run tests
    test_echo_agent_creation()
    print("✓ test_echo_agent_creation passed")
    
    test_echo_agent_handle_message()
    print("✓ test_echo_agent_handle_message passed")
    
    test_coordinator_creation()
    print("✓ test_coordinator_creation passed")
    
    test_coordinator_register_agent()
    print("✓ test_coordinator_register_agent passed")
    
    test_coordinator_register_duplicate_agent()
    print("✓ test_coordinator_register_duplicate_agent passed")
    
    test_coordinator_send_message()
    print("✓ test_coordinator_send_message passed")
    
    test_coordinator_send_to_unregistered_agent()
    print("✓ test_coordinator_send_to_unregistered_agent passed")
    
    test_coordinator_auto_generate_request_id()
    print("✓ test_coordinator_auto_generate_request_id passed")
    
    test_mock_gemini_response()
    print("✓ test_mock_gemini_response passed")
    
    test_mock_gemini_response_with_complex_json()
    print("✓ test_mock_gemini_response_with_complex_json passed")
    
    print("\n✅ All tests passed!")

