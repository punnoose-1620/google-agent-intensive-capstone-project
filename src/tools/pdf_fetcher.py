"""
PDF Fetcher Tool - Fetch and parse PDF documents.

This module provides functionality to download PDF files from URLs and extract
text content using pdfplumber.
"""

import requests
import pdfplumber
from typing import Optional
from io import BytesIO


class ParseError(Exception):
    """Custom exception for PDF parsing errors."""
    pass


class FetchError(Exception):
    """Custom exception for PDF fetching errors."""
    pass


def fetch_pdf(url: str, timeout: int = 30) -> bytes:
    """
    Fetch a PDF file from a URL.
    
    Args:
        url: URL of the PDF file to fetch
        timeout: Request timeout in seconds (default: 30)
        
    Returns:
        PDF file content as bytes
        
    Raises:
        FetchError: If the PDF cannot be fetched (network error, invalid URL, etc.)
        
    Example:
        >>> pdf_bytes = fetch_pdf("https://example.com/document.pdf")
        >>> len(pdf_bytes)
        12345
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()
        
        # Check if content is actually a PDF
        content_type = response.headers.get('Content-Type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            # Still try to parse, but warn
            pass
        
        pdf_bytes = response.content
        
        # Basic validation - check PDF magic bytes
        if not pdf_bytes.startswith(b'%PDF'):
            raise FetchError(f"URL does not point to a valid PDF file: {url}")
        
        return pdf_bytes
        
    except requests.exceptions.Timeout:
        raise FetchError(f"Timeout while fetching PDF from {url} (exceeded {timeout}s)")
    except requests.exceptions.ConnectionError as e:
        raise FetchError(f"Connection error while fetching PDF from {url}: {e}")
    except requests.exceptions.HTTPError as e:
        raise FetchError(f"HTTP error while fetching PDF from {url}: {e}")
    except requests.exceptions.RequestException as e:
        raise FetchError(f"Error fetching PDF from {url}: {e}")
    except Exception as e:
        raise FetchError(f"Unexpected error while fetching PDF from {url}: {e}")


def parse_pdf(pdf_bytes: bytes) -> str:
    """
    Parse PDF bytes and extract text content.
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        Extracted text content as a string
        
    Raises:
        ParseError: If the PDF cannot be parsed or is corrupted
        
    Example:
        >>> pdf_bytes = fetch_pdf("https://example.com/document.pdf")
        >>> text = parse_pdf(pdf_bytes)
        >>> print(text[:100])
        "This is the extracted text from the PDF..."
    """
    if not pdf_bytes:
        raise ParseError("PDF bytes are empty")
    
    # Validate PDF magic bytes
    if not pdf_bytes.startswith(b'%PDF'):
        raise ParseError("Invalid PDF format: missing PDF header")
    
    try:
        text_content = []
        
        # Use pdfplumber to extract text
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                except Exception as e:
                    # Log but continue with other pages
                    print(f"Warning: Could not extract text from page {page_num}: {e}")
                    continue
        
        if not text_content:
            raise ParseError("No text content could be extracted from the PDF")
        
        # Join all pages with page breaks
        full_text = "\n\n--- Page Break ---\n\n".join(text_content)
        
        return full_text
        
    except pdfplumber.exceptions.PDFSyntaxError as e:
        raise ParseError(f"PDF syntax error: {e}")
    except Exception as e:
        raise ParseError(f"Error parsing PDF: {e}")


def fetch_and_parse_pdf(url: str, timeout: int = 30) -> str:
    """
    Convenience function to fetch and parse a PDF in one step.
    
    Args:
        url: URL of the PDF file to fetch and parse
        timeout: Request timeout in seconds (default: 30)
        
    Returns:
        Extracted text content as a string
        
    Raises:
        FetchError: If the PDF cannot be fetched
        ParseError: If the PDF cannot be parsed
        
    Example:
        >>> text = fetch_and_parse_pdf("https://arxiv.org/pdf/1234.5678.pdf")
        >>> print(f"Extracted {len(text)} characters")
    """
    pdf_bytes = fetch_pdf(url, timeout=timeout)
    return parse_pdf(pdf_bytes)

