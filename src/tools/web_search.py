"""
Web Search Tool - Search functionality for finding educational resources.

This module provides web search capabilities. Currently implements a placeholder
that returns sample results. Replace with actual SerpAPI or Google Search API integration.
"""

from typing import List, Dict, Any


class WebSearchError(Exception):
    """Custom exception for web search errors."""
    pass


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web for educational resources.
    
    This is a placeholder implementation that returns sample results.
    Replace this with actual API integration (SerpAPI, Google Search API, etc.).
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        List of dictionaries, each containing:
        - title: str - Title of the search result
        - snippet: str - Brief description/snippet
        - url: str - URL of the resource
        - source: str - Source of the result (e.g., "serpapi", "google")
        
    Raises:
        NotImplementedError: Always raised with instructions to implement actual API
        WebSearchError: For other search-related errors
        
    Example:
        >>> results = web_search("linear regression tutorial")
        >>> print(results[0]['title'])
    """
    # Placeholder implementation - returns sample results
    # TODO: Replace with actual SerpAPI or Google Search API integration
    
    raise NotImplementedError(
        "web_search() is not yet implemented with a real API.\n"
        "To implement:\n"
        "1. Install SerpAPI: pip install google-search-results\n"
        "   OR use Google Custom Search API\n"
        "2. Get API key from https://serpapi.com/ or Google Cloud Console\n"
        "3. Replace this function with actual API calls\n"
        "4. Example SerpAPI usage:\n"
        "   from serpapi import GoogleSearch\n"
        "   params = {'q': query, 'api_key': os.getenv('SERPAPI_KEY')}\n"
        "   search = GoogleSearch(params)\n"
        "   results = search.get_dict()['organic_results']\n"
        "   return [{'title': r['title'], 'snippet': r['snippet'], "
        "'url': r['link'], 'source': 'serpapi'} for r in results[:max_results]]"
    )
    
    # Sample return structure (unreachable code, but shows expected format)
    sample_results = [
        {
            "title": f"Sample Result 1 for: {query}",
            "snippet": "This is a placeholder snippet. Replace with actual search results.",
            "url": "https://example.com/result1",
            "source": "placeholder"
        },
        {
            "title": f"Sample Result 2 for: {query}",
            "snippet": "Another placeholder result for demonstration purposes.",
            "url": "https://example.com/result2",
            "source": "placeholder"
        }
    ]
    
    return sample_results[:max_results]


def web_search_placeholder(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Placeholder web search that returns sample results without raising an error.
    
    Useful for testing and development when API keys are not available.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        List of sample search result dictionaries
    """
    sample_results = []
    for i in range(min(max_results, 5)):
        sample_results.append({
            "title": f"Sample Educational Resource {i+1}: {query}",
            "snippet": f"This is a placeholder search result for '{query}'. "
                      f"In production, this would be replaced with actual search results "
                      f"from SerpAPI or Google Search API.",
            "url": f"https://example.com/educational-resource-{i+1}",
            "source": "placeholder"
        })
    
    return sample_results

