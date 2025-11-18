"""
Agent Framework - Base classes for multi-agent system coordination.

This module provides the foundational Agent class and Coordinator for managing
agent-to-agent communication in the Educational Tutor Agent system.
"""

import time
import uuid
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod


# Message Schema
# Expected message format:
# {
#     "action": str,           # Action to perform (e.g., "explain", "search", "evaluate")
#     "payload": dict,         # Action-specific data
#     "request_id": str,       # Unique identifier for this request
#     "meta": dict            # Additional metadata (timestamp, source, etc.)
# }


class Agent(ABC):
    """
    Base class for all agents in the system.
    
    Each agent has a name, a set of tools it can use, and implements
    a handle_message method to process incoming messages.
    """
    
    def __init__(self, name: str, tools: Optional[Dict[str, Callable]] = None):
        """
        Initialize an agent.
        
        Args:
            name: Unique name identifier for this agent
            tools: Dictionary of tool_name -> callable function that this agent can use
        """
        self.name = name
        self.tools = tools or {}
    
    @abstractmethod
    def handle_message(self, message: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        
        Args:
            message: Message dictionary with keys: action, payload, request_id, meta
            context: Optional context dictionary with additional information
            
        Returns:
            Response dictionary with keys: status, payload, request_id, meta
        """
        pass
    
    def use_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Execute a tool available to this agent.
        
        Args:
            tool_name: Name of the tool to use
            *args, **kwargs: Arguments to pass to the tool
            
        Returns:
            Result from the tool execution
            
        Raises:
            KeyError: If tool_name is not in self.tools
        """
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' not available to agent '{self.name}'")
        return self.tools[tool_name](*args, **kwargs)


class Coordinator:
    """
    Coordinates communication between agents.
    
    Manages agent registration, message routing, timeouts, and logging.
    """
    
    def __init__(self, timeout: float = 30.0, observability_hook: Optional[Callable] = None):
        """
        Initialize the coordinator.
        
        Args:
            timeout: Default timeout in seconds for agent message handling
            observability_hook: Optional function to call for logging events.
                               Should accept (agent_name, event_type, payload)
        """
        self.agents: Dict[str, Agent] = {}
        self.timeout = timeout
        self.observability_hook = observability_hook
    
    def register(self, agent: Agent) -> None:
        """
        Register an agent with the coordinator.
        
        Args:
            agent: Agent instance to register
            
        Raises:
            ValueError: If an agent with the same name is already registered
        """
        if agent.name in self.agents:
            raise ValueError(f"Agent '{agent.name}' is already registered")
        self.agents[agent.name] = agent
        self._log_event(agent.name, "registered", {"agent_name": agent.name})
    
    def send(
        self,
        from_agent: str,
        to_agent: str,
        message: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Send a message from one agent to another.
        
        Args:
            from_agent: Name of the sending agent
            to_agent: Name of the receiving agent
            message: Message dictionary with keys: action, payload, request_id, meta
            context: Optional context dictionary
            timeout: Optional timeout override (uses default if None)
            
        Returns:
            Response dictionary from the receiving agent
            
        Raises:
            KeyError: If from_agent or to_agent is not registered
            TimeoutError: If agent response exceeds timeout
            Exception: Any exception raised by the agent's handle_message
        """
        # Validate agents
        if from_agent not in self.agents:
            raise KeyError(f"Agent '{from_agent}' is not registered")
        if to_agent not in self.agents:
            raise KeyError(f"Agent '{to_agent}' is not registered")
        
        # Ensure message has required fields
        if "request_id" not in message:
            message["request_id"] = str(uuid.uuid4())
        if "meta" not in message:
            message["meta"] = {}
        
        # Add metadata
        message["meta"]["from_agent"] = from_agent
        message["meta"]["to_agent"] = to_agent
        message["meta"]["timestamp"] = time.time()
        
        # Use provided timeout or default
        timeout = timeout or self.timeout
        
        # Log message send
        self._log_event(
            from_agent,
            "message_sent",
            {
                "to_agent": to_agent,
                "action": message.get("action"),
                "request_id": message["request_id"]
            }
        )
        
        # Get target agent
        target_agent = self.agents[to_agent]
        
        # Execute with timeout
        start_time = time.time()
        try:
            # In a real implementation, you might use threading or asyncio for true timeout
            # For simplicity, we'll just track time and raise if it exceeds
            response = target_agent.handle_message(message, context)
            elapsed_time = time.time() - start_time
            
            if elapsed_time > timeout:
                raise TimeoutError(
                    f"Agent '{to_agent}' response took {elapsed_time:.2f}s, "
                    f"exceeding timeout of {timeout}s"
                )
            
            # Ensure response has required fields
            if "status" not in response:
                response["status"] = "ok"
            if "request_id" not in response:
                response["request_id"] = message["request_id"]
            if "meta" not in response:
                response["meta"] = {}
            
            response["meta"]["elapsed_time"] = elapsed_time
            response["meta"]["from_agent"] = to_agent
            
            # Log successful response
            self._log_event(
                to_agent,
                "message_handled",
                {
                    "from_agent": from_agent,
                    "action": message.get("action"),
                    "request_id": message["request_id"],
                    "elapsed_time": elapsed_time
                }
            )
            
            return response
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            # Log error
            self._log_event(
                to_agent,
                "message_error",
                {
                    "from_agent": from_agent,
                    "action": message.get("action"),
                    "request_id": message["request_id"],
                    "error": str(e),
                    "elapsed_time": elapsed_time
                }
            )
            raise
    
    def _log_event(self, agent_name: str, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Log an event using the observability hook if available.
        
        Args:
            agent_name: Name of the agent involved
            event_type: Type of event (e.g., "registered", "message_sent", "message_handled")
            payload: Additional event data
        """
        if self.observability_hook:
            try:
                self.observability_hook(agent_name, event_type, payload)
            except Exception as e:
                # Don't let logging errors break the coordinator
                print(f"Warning: Observability hook failed: {e}")
    
    def list_agents(self) -> list:
        """
        Get a list of all registered agent names.
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())


# Example Usage
if __name__ == "__main__":
    # Example: Create a simple echo agent
    class EchoAgent(Agent):
        """Simple agent that echoes back the message payload."""
        
        def handle_message(self, message: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            action = message.get("action", "unknown")
            payload = message.get("payload", {})
            
            return {
                "status": "ok",
                "payload": {
                    "echo": payload,
                    "action": action
                },
                "request_id": message.get("request_id"),
                "meta": {
                    "agent": self.name
                }
            }
    
    # Create coordinator
    coordinator = Coordinator(timeout=10.0)
    
    # Create and register agents
    echo_agent = EchoAgent(name="echo_agent")
    coordinator.register(echo_agent)
    
    # Send a message
    message = {
        "action": "echo",
        "payload": {"text": "Hello, world!"},
        "request_id": "test-123"
    }
    
    response = coordinator.send(
        from_agent="echo_agent",  # Self-send for demo
        to_agent="echo_agent",
        message=message
    )
    
    print("Message sent:", message)
    print("Response received:", response)
    print("Registered agents:", coordinator.list_agents())

