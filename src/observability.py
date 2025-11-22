"""
Observability Module - Logging, metrics, and tracing utilities.

This module provides observability features for monitoring agent interactions,
tracking metrics, and measuring performance.
"""

import json
import time
import functools
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from collections import Counter
import threading

# Try to import loguru, fallback to basic logging
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False


# Thread-safe metrics counter
_metrics_lock = threading.Lock()
_metrics = Counter()

# Log directory
_log_dir = Path("data/logs")
_log_dir.mkdir(parents=True, exist_ok=True)


def log_event(agent_name: str, event_type: str, payload: Dict[str, Any]) -> None:
    """
    Log an event with agent name, event type, and payload.
    
    Events are written as JSON-lines to data/logs/{date}.log for easy parsing
    and analysis. Also updates metrics counters.
    
    Args:
        agent_name: Name of the agent generating the event
        event_type: Type of event (e.g., "message_sent", "message_handled", "error")
        payload: Additional event data as dictionary
        
    Example:
        >>> log_event("tutor_agent", "message_handled", {
        ...     "action": "explain",
        ...     "request_id": "req-123",
        ...     "elapsed_time": 1.23
        ... })
    """
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent_name": agent_name,
        "event_type": event_type,
        "payload": payload
    }
    
    # Write to JSON-lines log file (one JSON object per line)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = _log_dir / f"{date_str}.log"
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')  # JSON-lines format
    except Exception as e:
        # Fallback to logger if file write fails
        logger.error(f"Failed to write log entry: {e}")
    
    # Update metrics (thread-safe)
    with _metrics_lock:
        _metrics[f"{agent_name}:{event_type}"] += 1
        _metrics[f"total:{event_type}"] += 1
    
    # Also log to logger if loguru is available
    if LOGURU_AVAILABLE:
        logger.info(f"[{agent_name}] {event_type}: {payload}")
    else:
        logger.info(f"[{agent_name}] {event_type}: {payload}")


def get_metrics() -> Dict[str, int]:
    """
    Get current metrics counters.
    
    Returns:
        Dictionary of metric names to counts
        
    Example:
        >>> metrics = get_metrics()
        >>> print(metrics)
        {'tutor_agent:message_handled': 5, 'total:message_handled': 10, ...}
    """
    with _metrics_lock:
        return dict(_metrics)


def reset_metrics() -> None:
    """Reset all metrics counters."""
    with _metrics_lock:
        _metrics.clear()


def trace_request(func: Optional[Callable] = None, *, agent_name: Optional[str] = None) -> Callable:
    """
    Decorator to measure latency on function calls and log the execution.
    
    Can be used as:
        @trace_request
        def my_function():
            ...
    
    Or with agent name:
        @trace_request(agent_name="tutor_agent")
        def my_function():
            ...
    
    Args:
        func: Function to decorate (if used as @trace_request)
        agent_name: Optional agent name for logging (defaults to function name)
        
    Returns:
        Decorated function that logs execution time
        
    Example:
        >>> @trace_request(agent_name="tutor_agent")
        >>> def handle_explain(payload):
        ...     # function implementation
        ...     return result
        >>> # Automatically logs execution time and metrics
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = agent_name or f.__name__
            
            # Log function start
            log_event(
                func_name,
                "function_start",
                {
                    "function": f.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
            )
            
            try:
                # Execute function
                result = f(*args, **kwargs)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Log successful completion
                log_event(
                    func_name,
                    "function_complete",
                    {
                        "function": f.__name__,
                        "elapsed_time": elapsed_time,
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                # Calculate elapsed time even on error
                elapsed_time = time.time() - start_time
                
                # Log error
                log_event(
                    func_name,
                    "function_error",
                    {
                        "function": f.__name__,
                        "elapsed_time": elapsed_time,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "success": False
                    }
                )
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    # Handle both @trace_request and @trace_request(agent_name="...")
    if func is None:
        return decorator
    else:
        return decorator(func)


# Example usage
if __name__ == "__main__":
    # Example 1: Basic logging
    log_event(
        "tutor_agent",
        "message_handled",
        {
            "action": "explain",
            "topic": "neural networks",
            "request_id": "req-001",
            "elapsed_time": 1.23
        }
    )
    
    # Example 2: Using trace_request decorator
    @trace_request(agent_name="tutor_agent")
    def example_function(x: int, y: int) -> int:
        """Example function to demonstrate tracing."""
        time.sleep(0.1)  # Simulate work
        return x + y
    
    # Call the traced function
    result = example_function(5, 3)
    print(f"Result: {result}")
    
    # Example 3: View metrics
    metrics = get_metrics()
    print("\nCurrent metrics:")
    for metric, count in sorted(metrics.items()):
        print(f"  {metric}: {count}")
    
    # Example 4: Log error event
    log_event(
        "quiz_agent",
        "error",
        {
            "error": "Failed to generate quiz",
            "request_id": "req-002",
            "error_type": "ValueError"
        }
    )
    
    # Show updated metrics
    print("\nUpdated metrics:")
    metrics = get_metrics()
    for metric, count in sorted(metrics.items()):
        print(f"  {metric}: {count}")
    
    print(f"\n✓ Log files written to: {_log_dir}")

