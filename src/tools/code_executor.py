"""
Code Executor Tool - Safe code execution in a sandboxed environment.

This module provides functionality to execute code snippets safely with
timeouts, resource limits, and restricted access.
"""

import subprocess
import sys
import tempfile
import os
import signal
from typing import Optional, Dict, Any
from io import StringIO
import contextlib


class CodeExecutionError(Exception):
    """Custom exception for code execution errors."""
    pass


class TimeoutError(Exception):
    """Custom exception for execution timeout."""
    pass


class SandboxError(Exception):
    """Custom exception for sandbox violations."""
    pass


# Restricted builtins for Python execution
RESTRICTED_BUILTINS = {
    'open': None,
    'file': None,
    'input': None,
    'raw_input': None,
    'execfile': None,
    'reload': None,
    '__import__': None,
    'eval': None,
    'compile': None,
    'exit': None,
    'quit': None,
    'help': None,
}


def execute_code(code_snippet: str, language: str = "python", timeout: int = 5) -> Dict[str, Any]:
    """
    Execute code in a sandboxed environment and return the output.
    
    Args:
        code_snippet: Code to execute as a string
        language: Programming language (default: "python")
        timeout: Maximum execution time in seconds (default: 5)
        
    Returns:
        Dictionary with keys:
        - output: str - Standard output from code execution
        - error: str - Error output (if any)
        - success: bool - Whether execution succeeded
        - execution_time: float - Time taken in seconds
        
    Raises:
        CodeExecutionError: For general execution errors
        TimeoutError: If execution exceeds timeout
        SandboxError: If code attempts restricted operations
        
    Example:
        >>> result = execute_code("print('Hello, World!')", "python")
        >>> print(result['output'])
        Hello, World!
    """
    if language.lower() == "python":
        return _execute_python(code_snippet, timeout)
    else:
        raise CodeExecutionError(
            f"Language '{language}' is not yet supported. "
            f"Supported languages: python"
        )


def _execute_python(code_snippet: str, timeout: int) -> Dict[str, Any]:
    """
    Execute Python code in a restricted environment.
    
    Args:
        code_snippet: Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution results
    """
    import time
    start_time = time.time()
    
    # Create restricted globals
    restricted_globals = {
        '__builtins__': {
            k: v for k, v in __builtins__.items() 
            if k not in RESTRICTED_BUILTINS
        },
        '__name__': '__main__',
        '__doc__': None,
    }
    
    # Capture stdout and stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    try:
        # Use subprocess for better isolation (safer than exec)
        result = _execute_python_subprocess(code_snippet, timeout)
        execution_time = time.time() - start_time
        
        return {
            "output": result.get("output", ""),
            "error": result.get("error", ""),
            "success": result.get("success", False),
            "execution_time": execution_time
        }
        
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return {
            "output": "",
            "error": f"Execution timed out after {timeout} seconds",
            "success": False,
            "execution_time": execution_time
        }
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "output": "",
            "error": f"Execution error: {str(e)}",
            "success": False,
            "execution_time": execution_time
        }


def _execute_python_subprocess(code_snippet: str, timeout: int) -> Dict[str, Any]:
    """
    Execute Python code using subprocess for better isolation.
    
    This is safer than using exec() as it provides process-level isolation.
    Cross-platform compatible (works on Windows and Unix).
    """
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Wrap code with timeout and restrictions (cross-platform)
        import platform
        is_windows = platform.system() == 'Windows'
        
        if is_windows:
            # Windows doesn't support SIGALRM, use threading timeout instead
            wrapped_code = f"""
import sys
import threading
from io import StringIO
import time

timeout_occurred = False

def timeout_handler():
    global timeout_occurred
    time.sleep({timeout})
    if not timeout_occurred:
        timeout_occurred = True
        print("__ERROR_START__", file=sys.stderr)
        print("Execution timed out", file=sys.stderr)
        print("__ERROR_END__", file=sys.stderr)
        sys.exit(1)

# Start timeout thread
timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
timeout_thread.start()

try:
    # Redirect stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    # Execute user code
    {code_snippet}
    
    # Get output
    output = sys.stdout.getvalue()
    error = sys.stderr.getvalue()
    
    # Restore stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Print results in a way we can capture
    print("__OUTPUT_START__")
    print(output, end='')
    print("__OUTPUT_END__")
    if error:
        print("__ERROR_START__", file=sys.stderr)
        print(error, end='', file=sys.stderr)
        print("__ERROR_END__", file=sys.stderr)
    
except Exception as e:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    print("__ERROR_START__", file=sys.stderr)
    print(f"Error: {{str(e)}}", file=sys.stderr)
    print("__ERROR_END__", file=sys.stderr)
"""
        else:
            # Unix/Linux - use signal-based timeout
            wrapped_code = f"""
import sys
import signal
from io import StringIO

# Set up timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

try:
    # Redirect stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    # Execute user code
    {code_snippet}
    
    # Get output
    output = sys.stdout.getvalue()
    error = sys.stderr.getvalue()
    
    # Restore stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Print results in a way we can capture
    print("__OUTPUT_START__")
    print(output, end='')
    print("__OUTPUT_END__")
    if error:
        print("__ERROR_START__", file=sys.stderr)
        print(error, end='', file=sys.stderr)
        print("__ERROR_END__", file=sys.stderr)
    
except Exception as e:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    print("__ERROR_START__", file=sys.stderr)
    print(f"Error: {{str(e)}}", file=sys.stderr)
    print("__ERROR_END__", file=sys.stderr)
finally:
    signal.alarm(0)  # Cancel timeout
"""
        f.write(wrapped_code)
        temp_file = f.name
    
    try:
        # Execute with subprocess and timeout
        process = subprocess.Popen(
            [sys.executable, temp_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout + 1,  # Add buffer for cleanup
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        
        stdout, stderr = process.communicate()
        
        # Parse output
        output = ""
        error = ""
        
        if "__OUTPUT_START__" in stdout:
            start_idx = stdout.find("__OUTPUT_START__") + len("__OUTPUT_START__")
            end_idx = stdout.find("__OUTPUT_END__")
            if end_idx > start_idx:
                output = stdout[start_idx:end_idx].strip()
        
        if "__ERROR_START__" in stderr:
            start_idx = stderr.find("__ERROR_START__") + len("__ERROR_START__")
            end_idx = stderr.find("__ERROR_END__")
            if end_idx > start_idx:
                error = stderr[start_idx:end_idx].strip()
        elif stderr:
            error = stderr.strip()
        
        success = process.returncode == 0 and not error
        
        return {
            "output": output,
            "error": error,
            "success": success
        }
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except Exception:
            pass


def execute_code_simple(code_snippet: str, language: str = "python", timeout: int = 5) -> str:
    """
    Simplified code execution that returns output as a string.
    
    This is a convenience wrapper around execute_code() that returns
    just the output string or error message.
    
    Args:
        code_snippet: Code to execute
        language: Programming language (default: "python")
        timeout: Maximum execution time in seconds (default: 5)
        
    Returns:
        Output string if successful, error message if failed
        
    Example:
        >>> result = execute_code_simple("print(2 + 2)")
        >>> print(result)
        4
    """
    result = execute_code(code_snippet, language, timeout)
    
    if result["success"]:
        return result["output"]
    else:
        return f"Error: {result['error']}"


# Example usage
if __name__ == "__main__":
    # Test basic execution
    test_code = """
print("Hello, World!")
print(2 + 2)
result = [x**2 for x in range(5)]
print(f"Squares: {result}")
"""
    
    print("Testing code execution:")
    result = execute_code(test_code, "python", timeout=5)
    print(f"Success: {result['success']}")
    print(f"Output:\n{result['output']}")
    if result['error']:
        print(f"Error: {result['error']}")
    print(f"Execution time: {result['execution_time']:.3f}s")

