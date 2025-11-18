"""
Tools package - Utility tools for agents.

This package contains tools that agents can use to interact with external resources.
"""

from .web_search import web_search, web_search_placeholder, WebSearchError
from .pdf_fetcher import (
    fetch_pdf,
    parse_pdf,
    fetch_and_parse_pdf,
    ParseError,
    FetchError
)
from .code_executor import (
    execute_code,
    execute_code_simple,
    CodeExecutionError,
    TimeoutError as CodeTimeoutError,
    SandboxError
)

__all__ = [
    'web_search',
    'web_search_placeholder',
    'WebSearchError',
    'fetch_pdf',
    'parse_pdf',
    'fetch_and_parse_pdf',
    'ParseError',
    'FetchError',
    'execute_code',
    'execute_code_simple',
    'CodeExecutionError',
    'CodeTimeoutError',
    'SandboxError',
]

