#!/usr/bin/env python3
"""
Shared utilities for SEC data processing and HTTP requests.
Eliminates code duplication across the pipeline scripts.
"""

import os
import json
import requests
import time
from typing import Dict, List, Optional, Any

# Constants
SEC_BASE_URL = "https://data.sec.gov"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "LookThroughProfits Pipeline/1.0 (contact@example.com)"

def sec_headers() -> Dict[str, str]:
    """Standard headers for SEC API requests."""
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }

def make_request(url: str, timeout: int = 30, retries: int = 2) -> Optional[requests.Response]:
    """
    Makes HTTP request with standard headers and retry logic.
    Returns Response object or None if all attempts fail.
    """
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=sec_headers(), timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == retries:
                print(f"Request failed after {retries + 1} attempts: {e}")
                return None
            time.sleep(1)  # Wait before retry
    return None

def fetch_sec_ticker_map(cache_path: str = "staging/sec_ticker_cache.json") -> Dict[str, str]:
    """
    Downloads SEC ticker-to-CIK mapping. Uses cache if available.
    Returns dict mapping ticker -> CIK (zero-padded).
    """
    # Try cache first
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    
    # Download fresh data
    response = make_request(SEC_COMPANY_TICKERS_URL)
    if not response:
        return {}
    
    data = response.json()
    ticker_map = {}
    
    for entry in data.values():
        ticker = entry.get('ticker', '').upper()
        cik = entry.get('cik_str') or str(entry.get('cik', ''))
        if ticker and cik:
            ticker_map[ticker] = f"{int(cik):010d}"
    
    # Cache the result
    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
    try:
        with open(cache_path, 'w') as f:
            json.dump(ticker_map, f, indent=2)
    except Exception:
        pass
    
    return ticker_map

def fetch_sec_submissions(cik: str) -> Optional[Dict]:
    """
    Fetches SEC submissions data for a CIK.
    Returns submissions dict or None if failed.
    """
    cik_padded = str(cik).zfill(10)
    url = f"{SEC_BASE_URL}/submissions/CIK{cik_padded}.json"
    response = make_request(url)
    return response.json() if response else None

def load_json_file(filepath: str) -> Optional[Dict]:
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def save_json_file(data: Any, filepath: str, create_dirs: bool = True) -> bool:
    """Save data to JSON file with error handling."""
    try:
        if create_dirs:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        return False

def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol for consistent lookup.
    Handles common variations and formats.
    """
    if not ticker:
        return ""
    
    ticker = ticker.strip().upper()
    
    # Remove common non-ticker words
    if ticker in ['INC', 'CORP', 'CO', 'LTD', 'LLC']:
        return ""
    
    # Keep only alphanumeric, dots, and hyphens
    import re
    ticker = re.sub(r'[^A-Z0-9.-]', '', ticker)
    
    # Basic validation
    if re.match(r'^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$', ticker):
        return ticker
    
    return ""

def generate_ticker_variants(ticker: str) -> List[str]:
    """Generate common ticker variants for fuzzy matching."""
    if not ticker:
        return []
    
    base = ticker.strip().upper()
    variants = {
        base,
        base.replace('.', ''),
        base.replace('.', '-'),
        base.replace('-', ''),
        base.replace('-', '.')
    }
    
    return [v for v in variants if v]

def enrich_companies_with_cik(companies: List[Dict], ticker_map: Dict[str, str]) -> None:
    """
    Add CIK numbers to company records using ticker lookup.
    Modifies companies list in-place.
    """
    for company in companies:
        ticker = company.get('primary_ticker', '')
        if ticker and ticker in ticker_map:
            company['cik'] = ticker_map[ticker]
        else:
            company['cik'] = ""

def setup_openai():
    """Setup OpenAI client if available and configured."""
    try:
        import openai
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None
        
        if hasattr(openai, 'OpenAI'):
            return openai.OpenAI(api_key=api_key)
        else:
            openai.api_key = api_key
            return openai
    except ImportError:
        return None

def query_openai(prompt: str, max_tokens: int = 150) -> str:
    """
    Query OpenAI with error handling.
    Returns response text or error message.
    """
    client = setup_openai()
    if not client:
        return "OpenAI not configured or available"
    
    try:
        if hasattr(client, 'chat'):
            # New client
            response = client.chat.completions.create(
                model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        else:
            # Legacy client
            response = client.ChatCompletion.create(
                model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"OpenAI query failed: {e}"
