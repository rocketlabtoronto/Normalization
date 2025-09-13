#!/usr/bin/env python3
"""
Get economic weights for dual class companies by analyzing their SEC filings for share class information.
This script takes the dual_class_output.json and enriches it with economic weight data from SEC filings.
"""

import os
import json
import re
import requests
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys

# Attempt to load .env if python-dotenv is installed
try:  # optional
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Manual fallback .env loader if OPENAI_API_KEY still unset
if not os.getenv("OPENAI_API_KEY") and os.path.exists(".env"):
    try:
        with open(".env", "r", encoding="utf-8") as _f:
            for _ln in _f:
                if not _ln.strip() or _ln.strip().startswith("#"):
                    continue
                if "=" in _ln:
                    k, v = _ln.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# Helper: deterministic HTML fallback parsing for cover page trading symbols & classes
def _fallback_parse_cover_page(content: str) -> List[Dict[str, Any]]:
    """Parse the filing HTML deterministically to extract class names and tickers
    from the 'Securities registered pursuant to Section 12(b) of the Act' / 'Trading Symbol(s)' table.
    Returns a list of dicts with class_name, ticker (may be ''), ticker_source.
    """
    results: List[Dict[str, Any]] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return results

    try:
        soup = BeautifulSoup(content[:400000], 'lxml')  # limit parse scope
    except Exception:
        try:
            soup = BeautifulSoup(content[:400000], 'html.parser')
        except Exception:
            return results

    # Normalize text to search anchor nodes
    full_text_lower = soup.get_text(" \n", strip=True).lower()
    if 'trading symbol' not in full_text_lower and 'securities registered pursuant to section 12' not in full_text_lower:
        return results

    # Candidate tables: must contain 'Trading Symbol' and some 'Class' or 'Common Stock'
    tables = soup.find_all('table')
    for tbl in tables:
        try:
            txt = tbl.get_text(" ", strip=True)
            low = txt.lower()
            if ('trading symbol' in low or 'trading symbols' in low) and ('class' in low or 'common stock' in low):
                # Process rows
                rows = tbl.find_all('tr')
                for r in rows:
                    cells = r.find_all(['td', 'th'])
                    if len(cells) < 2:
                        continue
                    row_text = ' '.join(c.get_text(" ", strip=True) for c in cells)
                    rlow = row_text.lower()
                    if 'trading symbol' in rlow and len(cells) <= 3:
                        # Header row
                        continue
                    # Heuristic: class description usually longer, ticker cell short with A-Z chars
                    # Build candidate pairs by scanning cells
                    class_part = None
                    ticker_part = None
                    for c in cells:
                        ctext = c.get_text(" ", strip=True)
                        cstrip = ctext.strip()
                        # Potential tickers separated by commas/spaces
                        if 1 <= len(cstrip) <= 8 and re.fullmatch(r'[A-Za-z\.\-]{1,8}', cstrip):
                            ticker_part = cstrip.upper()
                        elif re.search(r'class\s+[a-z0-9]', cstrip.lower()) or 'common stock' in cstrip.lower():
                            class_part = cstrip
                    if not class_part:
                        # Try first cell as class if it contains stock keywords
                        first = cells[0].get_text(" ", strip=True)
                        if 'stock' in first.lower():
                            class_part = first
                    if class_part:
                        results.append({
                            'class_name': class_part,
                            'ticker': ticker_part if ticker_part else '',
                            'ticker_source': 'cover page fallback HTML'
                        })
        except Exception:
            continue

    # Deduplicate by normalized class key keeping first with ticker if any
    dedup: Dict[str, Dict[str, Any]] = {}
    def _norm(name: str) -> str:
        n = name.lower().strip()
        n = re.split(r'[:;(]', n)[0]
        n = re.sub(r'[^a-z0-9\s]', ' ', n)
        n = re.sub(r'\s+', ' ', n).strip()
        m = re.search(r'class\s*([a-z0-9]+)', n)
        if m:
            tok = m.group(1)
            if tok.isdigit():
                return f'class_num_{tok}'
            return f'class_{tok[0].upper()}'
        if 'common stock' in n and 'class' not in n:
            return 'common'
        return n[:60]

    for entry in results:
        key = _norm(entry['class_name'])
        if key not in dedup:
            dedup[key] = entry
        else:
            # Prefer one with ticker
            if not dedup[key].get('ticker') and entry.get('ticker'):
                dedup[key] = entry
    return list(dedup.values())

@dataclass
class EconomicWeight:
    """Data structure for economic weight information"""
    class_name: str
    ticker: str = ""
    shares_outstanding: Optional[float] = None
    economic_weight: Optional[float] = None  # Proportion of total economic value (0.0 to 1.0)
    conversion_ratio: Optional[float] = None  # How many units of primary class this represents
    votes_per_share: Optional[float] = None  # Voting power per share
    source: str = ""  # Source of the data (10-K, 10-Q, etc.)
    ticker_source: Optional[str] = None  # Where the ticker came from (e.g., Trading Symbol(s) cover page)

def fetch_sec_cik_map() -> Dict[str, str]:
    """Fetch SEC ticker to CIK mapping."""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {'User-Agent': 'Economic Weight Analysis Tool (contact@example.com)'}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        ticker_map = {}
        for entry in data.values():
            ticker = entry.get('ticker', '').upper()
            cik = entry.get('cik_str') or str(entry.get('cik', ''))
            if ticker and cik:
                # Format CIK with leading zeros
                ticker_map[ticker] = f"{int(cik):010d}"
        
        return ticker_map
    except Exception as e:
        print(f"Warning: Could not fetch SEC data: {e}")
        return {}

def investigate_no_cik_with_ai(company_name: str, primary_ticker: str) -> Dict[str, Any]:
    """Use AI to investigate why a company has no CIK - likely delisted, acquired, or name change"""
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('LLM_MODEL', 'gpt-4o')
    
    if not api_key:
        return {"status": "unknown", "reason": "No API key"}
    
    prompt = f"""Investigate why {company_name} (ticker: {primary_ticker}) cannot be found in SEC CIK database.

This usually happens when companies are:
1. DELISTED from stock exchanges
2. ACQUIRED by other companies
3. MERGED with other entities
4. WENT BANKRUPT/LIQUIDATED
5. CHANGED COMPANY NAME significantly

Please research and determine what happened to this company. Return ONLY JSON:

{{
  "status": "delisted" | "acquired" | "merged" | "bankrupt" | "name_changed" | "unknown",
  "reason": "Brief explanation of what happened",
  "acquirer": "Name of acquiring company if applicable",
  "date": "Approximate date if known",
  "still_active": true/false
}}

Company: {company_name} ({primary_ticker})"""

    try:
        try:
            import openai  # type: ignore
        except ImportError:
            return {"status": "unknown", "reason": "OpenAI not available"}

        if hasattr(openai, 'OpenAI'):
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial research analyst. Return ONLY JSON as specified."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            result = response.choices[0].message.content if response.choices else ''
        else:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial research analyst. Return ONLY JSON as specified."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            result = response['choices'][0]['message']['content']

        # Parse AI response
        if result:
            # Clean JSON
            cleaned = result.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Extract JSON
            if '{' in cleaned:
                start = cleaned.find('{')
                brace = 0
                end = -1
                for i, ch in enumerate(cleaned[start:], start=start):
                    if ch == '{':
                        brace += 1
                    elif ch == '}':
                        brace -= 1
                        if brace == 0:
                            end = i + 1
                            break
                if end != -1:
                    candidate = cleaned[start:end]
                else:
                    candidate = cleaned[start:]
                
                import json
                data = json.loads(candidate)
                return data
        
        return {"status": "unknown", "reason": "Could not parse AI response"}
        
    except Exception as e:
        print(f"    ❌ AI investigation failed: {e}")
        return {"status": "unknown", "reason": f"Error: {e}"}
    """Lookup CIK using company name or ticker with SEC API"""
    # First try the ticker map
    if primary_ticker and primary_ticker.upper() in ticker_map:
        cik = ticker_map[primary_ticker.upper()]
        print(f"    ✅ Found CIK via ticker {primary_ticker}: {cik}")
        return cik
    
    # For well-known companies, try common ticker variants
    if company_name and 'berkshire' in company_name.lower():
        for variant in ['BRK.A', 'BRK.B', 'BRK-A', 'BRK-B']:
            if variant in ticker_map:
                cik = ticker_map[variant]
                print(f"    ✅ Found Berkshire CIK via {variant}: {cik}")
                return cik
    
    # Try SEC company search API
    headers = {'User-Agent': 'Economic Weight Analysis Tool (contact@example.com)'}
    
    # Search by company name
    if company_name:
        try:
            search_url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(search_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Search through company titles
            for entry in data.values():
                title = entry.get('title', '').lower()
                if company_name.lower() in title or title in company_name.lower():
                    cik = entry.get('cik_str') or str(entry.get('cik', ''))
                    if cik:
                        formatted_cik = f"{int(cik):010d}"
                        print(f"    ✅ Found CIK via company name search: {formatted_cik}")
                        return formatted_cik
        except Exception as e:
            print(f"    → Company name search failed: {e}")
    
    print(f"    ❌ Could not find CIK for {company_name} ({primary_ticker})")
    return None


def get_latest_filing(cik: str, user_agent: str = "Economic Weight Analysis Tool (contact@example.com)") -> Optional[Dict]:
    """Get the latest 10-K or 10-Q filing for a given CIK"""
    cik_padded = str(cik).zfill(10)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    
    try:
        response = requests.get(submissions_url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Find the latest 10-K or 10-Q filing
        recent_filings = data.get("filings", {}).get("recent", {})
        
        if not recent_filings:
            return None
        
        forms = recent_filings.get("form", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        filing_dates = recent_filings.get("filingDate", [])
        primary_documents = recent_filings.get("primaryDocument", [])
        
        # Find the most recent 10-K or 10-Q
        for i, form in enumerate(forms):
            if form in ["10-K", "10-Q"]:
                primary_doc = primary_documents[i] if i < len(primary_documents) else None
                return {
                    "cik": cik_padded,
                    "form_type": form,
                    "accession_number": accession_numbers[i],
                    "filing_date": filing_dates[i],
                    "primary_document": primary_doc
                }
        
        return None
        
    except requests.RequestException as e:
        print(f"Error fetching filings for CIK {cik}: {e}")
        return None

def get_filing_content(filing_info: Dict, user_agent: str = "Economic Weight Analysis Tool (contact@example.com)") -> Optional[str]:
    """Download the filing content"""
    cik = int(filing_info["cik"])
    accession = filing_info["accession_number"]
    accession_nodash = accession.replace('-', '')
    primary_document = filing_info.get("primary_document")
    
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive"
    }
    
    # If we have the primary document name, use it first
    if primary_document:
        primary_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_document}"
        try:
            print(f"    → Trying primary document: {primary_url}")
            response = requests.get(primary_url, headers=headers, timeout=60)
            if response.status_code == 200 and len(response.text.strip()) > 1000:
                print(f"    → Successfully downloaded filing content ({len(response.text)} characters)")
                return response.text
            else:
                print(f"    → Primary document failed with status {response.status_code}")
        except Exception as e:
            print(f"    → Primary document error: {e}")
    
    # Try the standard SEC EDGAR URL pattern
    standard_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession_nodash}.htm"
    try:
        print(f"    → Trying standard URL: {standard_url}")
        response = requests.get(standard_url, headers=headers, timeout=60)
        if response.status_code == 200 and len(response.text.strip()) > 1000:
            print(f"    → Successfully downloaded filing content ({len(response.text)} characters)")
            return response.text
        else:
            print(f"    → Standard URL failed with status {response.status_code}")
    except Exception as e:
        print(f"    → Standard URL error: {e}")
    
    # Try alternative patterns
    alternative_urls = [
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession}.htm",
        f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{cik}/{accession_nodash}/{primary_document}" if primary_document else None,
    ]
    
    for alt_url in alternative_urls:
        if alt_url is None:
            continue
        try:
            print(f"    → Trying alternative: {alt_url}")
            response = requests.get(alt_url, headers=headers, timeout=60)
            if response.status_code == 200 and len(response.text.strip()) > 1000:
                print(f"    → Alternative URL succeeded ({len(response.text)} characters)")
                return response.text
        except Exception as e:
            print(f"    → Alternative URL error: {e}")
            continue
    
    print(f"    → All download attempts failed")
    return None

def extract_economic_weights_with_ai(content: str, company_name: str, ticker: str) -> List[EconomicWeight]:
    """Use OpenAI to extract economic weight information from SEC filing content
    Implements:
      - Content snippet reduction
      - Retry (3 attempts)
      - Raw response logging & empty guard
      - Regex / brace-scan JSON extraction
      - Deterministic HTML fallback if AI fails
    """
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('LLM_MODEL', 'gpt-4o')

    if not api_key:
        print(f"    ❌ OPENAI_API_KEY not set, skipping AI analysis")
        return []

    # Extract a focused snippet containing cover page + first occurrences of share keywords
    def _extract_relevant_snippet(html: str) -> str:
        upper_limit = 250000  # hard safety cap
        snippet = html[:upper_limit]
        
        # Priority 1: Find the cover page table with trading symbols
        cover_patterns = [
            r'Securities registered pursuant to Section 12',
            r'trading symbol',
            r'title of each class',
            r'name of each exchange'
        ]
        
        cover_sections = []
        for pattern in cover_patterns:
            matches = list(re.finditer(pattern, snippet, re.IGNORECASE))
            for match in matches:
                start = max(0, match.start() - 2000)
                end = min(len(snippet), match.start() + 20000)
                cover_sections.append(snippet[start:end])
        
        # Priority 2: Look for voting rights and capitalization sections
        voting_patterns = [
            r'voting rights|vote|voting power',
            r'conversion ratio|convertible into|each class',
            r'capitalization|capital structure|voting structure|class.*voting|dual.*class'
        ]
        
        voting_sections = []
        for pattern in voting_patterns:
            matches = list(re.finditer(pattern, html[:upper_limit], re.IGNORECASE))
            for match in matches[:2]:  # Limit to first 2 matches per pattern
                start = max(0, match.start() - 3000)
                end = min(len(html), match.start() + 10000)
                voting_sections.append(html[start:end])
        
        # Combine all sections, prioritizing cover page content
        combined = '\n\n=== COVER PAGE TABLE ===\n\n'.join(cover_sections[:3])  # Max 3 cover sections
        if voting_sections:
            combined += '\n\n=== VOTING AND STRUCTURE ===\n\n' + '\n\n'.join(voting_sections[:3])  # Max 3 voting sections
        
        return combined[:120000] if combined else snippet[:120000]  # final cap

    snippet = _extract_relevant_snippet(content)
    print(f"    🧪 Prompt snippet length: {len(snippet)} chars (original {len(content)} chars)")

    print(f"    🤖 Preparing AI prompt for OpenAI {model}...")
    prompt = f"""You are analyzing a SEC filing for {company_name} ({ticker}). This is a dual-class company.

Follow instructions precisely and return ONLY JSON (no markdown, no commentary). If data absent, use empty arrays.

REQUIRED JSON FORMAT:
{{
  "classes": [
    {{
      "class_name": "Class A" or "Class B" or "Common Stock" etc (STANDARDIZED),
      "ticker": "EXACT_TICKER" or "" if not publicly traded,
      "ticker_source": "cover page table" or "not traded" or "",
      "shares_outstanding": number (ONLY outstanding shares, not authorized),
      "conversion_ratio": number (economic conversion ratio - NEVER null),
      "votes_per_share": number or null (votes per share),
      "is_publicly_traded": true or false
    }}
  ]
}}

CRITICAL INSTRUCTIONS:
1. DETECT ALL SHARE CLASSES - Many companies have subtle dual-class structures:
   - Look for "Class A Common Stock" AND "Common Stock" as separate classes
   - Check for different par values ($5 vs $1) indicating separate classes
   - Search for different voting rights (1 vote vs 0.1 vote vs 10 votes)
   - Some companies call classes "Class A" and "Common Stock" (not "Class B")
   - Example: A.O. Smith has "Class A Common Stock" (1 vote) and "Common Stock" (0.1 vote)

2. STANDARDIZE class_name appropriately:
   - "Class A Common Stock" → "Class A" 
   - "Common Stock" (when different from Class A) → "Common Stock"
   - "Class B Common Stock" → "Class B"
   - Keep original naming if it's the standard identifier
   - Remove descriptive text but preserve meaningful distinctions

3. TICKER SYMBOLS - EXTRACT FROM COVER PAGE TABLE:
   - Find the table "Securities registered pursuant to Section 12(b) of the Act"
   - This table shows "Title of each class" and "Trading Symbol(s)" columns
   - Extract the EXACT ticker for each class from this table
   - Examples: 
     * "Class A Common Stock" with symbol "BRK.A" → ticker: "BRK.A", ticker_source: "cover page table"
     * "Class B Common Stock" with symbol "BRK.B" → ticker: "BRK.B", ticker_source: "cover page table"
     * If class shows "None" or empty trading symbol → ticker: "", ticker_source: "not traded"
   - CRITICAL: Use the exact ticker from the SEC table, not the company's primary ticker
   - If a class is not listed in the trading table, set ticker: "", ticker_source: "not traded"

4. PUBLICLY TRADED STATUS:
   - is_publicly_traded: true if the class has a ticker symbol in the cover table
   - is_publicly_traded: false if no ticker or shows "None" in trading symbol

5. SHARES OUTSTANDING: Extract ONLY outstanding shares (not authorized, not issued)
   - Look for "shares outstanding", "outstanding common shares"
   - Ignore "authorized shares" or "shares authorized"
   - CRITICAL: Each class must have different share counts
   - If you cannot find distinct share counts, search harder in the filing
   - Look for the LATEST quarterly data available in the filing

6. VOTING RIGHTS: Extract votes per share for each class
   - Look for "votes per share", "voting power", "voting rights"
   - Check for fractional voting (like 0.1 votes per share, 0.0001 votes per share)
   - Research exact voting ratios from your knowledge base and SEC filings
   - Common patterns: 1 vote, 10 votes, 0.1 votes, 0.0001 votes (1/10,000)
   - Examples: "Class A: 10,000 votes per share", "Class B: 1 vote per share", "Common Stock: 0.1 vote per share"
   - CRITICAL: This is essential data - extract even if not explicitly stated
   - NEVER leave votes_per_share as null for any class

7. CONVERSION RATIO - ECONOMIC EQUIVALENCE (NEVER NULL):
   - This represents economic conversion rights between classes
   - If shares have equal economic rights (dividends, liquidation): set to 1 for both classes
   - If Class A converts to X Class B shares: Class A = X, Class B = 1
   - Examples: 
     * Equal economic rights → Class A: 1, Class B: 1
     * Berkshire style → Class A: 1500, Class B: 1 (1 Class A = 1500 Class B)
   - NEVER use null - always provide a number

8. CLASS DIFFERENTIATION - AVOID DUPLICATES:
   - If filing mentions "Class A" and "Class B", create separate entries
   - If only "Common Stock" mentioned, look for voting differences
   - Each class MUST have unique characteristics (shares, votes, or ticker)
   - NEVER create identical duplicate entries

9. COMPREHENSIVE SEARCH: Look beyond obvious "Class A/B" patterns:
   - Search filing for "voting", "par value", "common stock"
   - Check if total outstanding shares match across all classes
   - Verify the company actually has multiple share classes

FILING CONTENT SNIPPET (HTML/text, sanitized):
{snippet[:100000]}"""

    try:
        try:
            import openai  # type: ignore
        except ImportError:
            print(f"    ❌ openai package not installed, skipping AI analysis")
            # Fallback directly
            fallback_entries = _fallback_parse_cover_page(content)
            if fallback_entries:
                print(f"    🔧 Fallback HTML extracted {len(fallback_entries)} classes")
                return [EconomicWeight(class_name=e['class_name'], ticker=e['ticker'], source='HTML fallback', ticker_source=e['ticker_source']) for e in fallback_entries]
            return []

        result: Optional[str] = None
        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                print(f"    📡 Calling OpenAI API attempt {attempt}/3 ({model})...")
                if hasattr(openai, 'OpenAI'):
                    client = openai.OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert financial analyst. Return ONLY strict JSON as specified."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    result = response.choices[0].message.content if response.choices else ''
                else:
                    openai.api_key = api_key  # type: ignore
                    response = openai.ChatCompletion.create(  # type: ignore
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert financial analyst. Return ONLY strict JSON as specified."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    result = response['choices'][0]['message']['content'] if response.get('choices') else ''  # type: ignore
                if not result or not str(result).strip():
                    raise ValueError('Empty LLM response')
                result = result.strip()
                print(f"    ✅ Received response ({len(result)} chars) | preview: {result[:120].replace('\n',' ')}...")
                # Attempt JSON parse
                parsed_weights = _parse_ai_json_to_weights(result, company_name, ticker)
                if parsed_weights is not None:
                    return parsed_weights
                else:
                    raise ValueError('JSON parse failed after extraction logic')
            except Exception as e:  # capture failure for retry
                last_error = e
                print(f"    ⚠️ Attempt {attempt} failed: {e}")
                if attempt < 3:
                    sleep_for = 1.5 * attempt
                    print(f"    ⏳ Backoff {sleep_for:.1f}s before retry...")
                    time.sleep(sleep_for)
                continue
        print(f"    ❌ All AI attempts failed: {last_error}")
        # Fallback deterministic parse
        fallback_entries = _fallback_parse_cover_page(content)
        if fallback_entries:
            print(f"    🔧 HTML fallback succeeded with {len(fallback_entries)} classes")
            return [EconomicWeight(class_name=e['class_name'], ticker=e['ticker'], source='HTML fallback', ticker_source=e['ticker_source']) for e in fallback_entries]
        return []
    except Exception as outer:
        print(f"Warning: AI analysis failed for {company_name}: {outer}")
        fallback_entries = _fallback_parse_cover_page(content)
        if fallback_entries:
            print(f"    🔧 HTML fallback succeeded with {len(fallback_entries)} classes")
            return [EconomicWeight(class_name=e['class_name'], ticker=e['ticker'], source='HTML fallback', ticker_source=e['ticker_source']) for e in fallback_entries]
        return []

# Helper: robust JSON extraction

def _standardize_class_name(raw_name: str) -> str:
    """Standardize class names while preserving important distinctions like 'Common Stock'"""
    if not raw_name:
        return "Class A"  # Default fallback
    
    import re
    
    # Preserve exact "Common Stock" when it's the full designation (not "Class A Common Stock")
    if raw_name.strip().lower() == "common stock":
        return "Common Stock"
    
    # Handle "Class A Common Stock" pattern - extract just "Class A"
    match = re.search(r'Class\s+([A-Z])\s+Common\s+Stock', raw_name, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        return f"Class {letter}"
    
    # Extract the class letter using regex for standard patterns
    # Look for patterns like "Class A", "Class B:", "Class A Common", etc.
    match = re.search(r'Class\s+([A-Z])', raw_name, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        return f"Class {letter}"
    
    # Handle preferred stock
    if 'preferred' in raw_name.lower():
        return "Preferred Stock"
    
    # Fallback: look for single letters that might indicate class
    match = re.search(r'\b([A-Z])\b', raw_name)
    if match:
        letter = match.group(1).upper()
        return f"Class {letter}"
    
    # Final fallback based on common terms
    if 'common' in raw_name.lower():
        return "Common Stock"
    elif 'b' in raw_name.lower():
        return "Class B"
    else:
        return "Class A"


def _assign_tickers_with_ai(company: Dict) -> Dict:
    """Use AI to assign correct tickers for well-known dual-class companies"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print(f"    ❌ No OpenAI API key, skipping AI ticker assignment")
        return company
    
    company_name = company.get('company_name', '')
    classes = company.get('classes', [])
    
    if not classes:
        return company
    
    # Create AI prompt to determine correct tickers
    class_info = []
    for cls in classes:
        class_info.append(f"- {cls.get('class_name', '')}: votes_per_share={cls.get('votes_per_share', 'unknown')}")
    
    class_list = '\n'.join(class_info)
    
    prompt = f"""You are a financial data expert. For the company "{company_name}", determine the correct trading ticker symbols for each share class.

Company: {company_name}
Share Classes:
{class_list}

Return ONLY JSON in this format:
{{
  "tickers": [
    {{"class_name": "Class A", "ticker": "CORRECT_TICKER_A"}},
    {{"class_name": "Class B", "ticker": "CORRECT_TICKER_B"}}
  ]
}}

Important:
- Use the standardized class names "Class A", "Class B", etc.
- Provide the exact ticker symbols traded on exchanges
- If a class is not publicly traded, use an empty string ""
- For Berkshire Hathaway: Class A = "BRK.A", Class B = "BRK.B"
"""

    try:
        import openai
        if hasattr(openai, 'OpenAI'):
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv('LLM_MODEL', 'gpt-4o'),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            raw_response = response.choices[0].message.content.strip()
        else:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=os.getenv('LLM_MODEL', 'gpt-4o'),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            raw_response = response.choices[0].message.content.strip()
        
        print(f"    ✅ AI ticker assignment response: {raw_response[:100]}...")
        
        # Parse the response
        import json
        import re
        
        # Clean the response
        cleaned = raw_response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # Extract JSON object
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            ticker_mappings = data.get('tickers', [])
            
            # Apply the ticker assignments
            for cls in company.get('classes', []):
                class_name = cls.get('class_name', '')
                for mapping in ticker_mappings:
                    if mapping.get('class_name') == class_name:
                        old_ticker = cls.get('ticker', '')
                        new_ticker = mapping.get('ticker', '')
                        cls['ticker'] = new_ticker
                        print(f"    🎯 Updated {class_name}: {old_ticker} → {new_ticker}")
                        break
            
            return company
        else:
            print(f"    ❌ Could not parse AI response as JSON")
            return company
            
    except Exception as e:
        print(f"    ❌ AI ticker assignment failed: {e}")
        return company


def _clean_ticker_by_trading_status(ticker: str, is_publicly_traded: bool) -> str:
    """Return empty string for non-traded securities, clean ticker otherwise"""
    if not is_publicly_traded:
        return ""
    return ticker.strip() if ticker else ""


def _parse_ai_json_to_weights(raw: str, company_name: str, primary_ticker: str) -> Optional[List[EconomicWeight]]:
    try:
        cleaned = raw.strip()
        # Strip markdown fences
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        # If multiple objects / commentary, isolate first JSON object using brace scan
        if '{' in cleaned:
            start = cleaned.find('{')
            brace = 0
            end = -1
            for i, ch in enumerate(cleaned[start:], start=start):
                if ch == '{':
                    brace += 1
                elif ch == '}':
                    brace -= 1
                    if brace == 0:
                        end = i + 1
                        break
            if end != -1:
                candidate = cleaned[start:end]
            else:
                candidate = cleaned[start:]
        else:
            # Try regex fallback
            m = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if not m:
                raise ValueError('No JSON braces found')
            candidate = m.group(0)
        data = json.loads(candidate)
        classes = data.get('classes', []) if isinstance(data, dict) else []
        weights: List[EconomicWeight] = []
        for c in classes:
            if not isinstance(c, dict):
                continue
            weights.append(EconomicWeight(
                class_name=_standardize_class_name(c.get('class_name', '')),
                ticker=_clean_ticker_by_trading_status(c.get('ticker', ''), c.get('is_publicly_traded', True)),
                shares_outstanding=c.get('shares_outstanding'),
                economic_weight=None,  # Removed - not needed
                conversion_ratio=c.get('conversion_ratio'),
                votes_per_share=c.get('votes_per_share'),
                source='AI analysis of SEC filing',
                ticker_source=c.get('ticker_source')
            ))
        print(f"    🧩 Parsed {len(weights)} classes from AI JSON")
        
        # Deduplicate identical class entries
        weights = _deduplicate_classes(weights)
        if len(weights) != len(data.get('classes', [])):
            print(f"    🔧 After deduplication: {len(weights)} unique classes")
        
        return weights
    except Exception as e:
        print(f"    ❌ JSON parse helper failed: {e}")
        print(f"       Raw snippet (first 200 chars): {raw[:200].replace('\n',' ')}...")
        return None

def calculate_economic_weights(weights: List[EconomicWeight]) -> List[EconomicWeight]:
    """Calculate relative economic weights if not already provided"""
    if not weights:
        return weights
    
    # If we have shares outstanding data, calculate weights
    total_economic_value = 0.0
    weights_to_calculate = []
    
    for weight in weights:
        if weight.shares_outstanding:
            # Use conversion ratio if available, otherwise assume 1.0
            conversion_ratio = weight.conversion_ratio if weight.conversion_ratio else 1.0
            
            # For economic calculation, we need to understand the economic equivalence
            # Class A shares typically have higher economic value per share
            # If conversion_ratio < 1.0, it means this class converts to fewer of the other class
            # If conversion_ratio > 1.0, it means this class converts to more of the other class
            
            # Calculate economic value based on shares outstanding
            # For now, use shares_outstanding directly - conversion ratios will be handled
            # when we calculate relative weights between classes
            economic_value = weight.shares_outstanding * conversion_ratio
            total_economic_value += economic_value
            weights_to_calculate.append((weight, economic_value))
    
    # Calculate proportions
    if total_economic_value > 0:
        for weight, economic_value in weights_to_calculate:
            weight.economic_weight = economic_value / total_economic_value
            
            # Debug output
            print(f"    → {weight.class_name}: {weight.shares_outstanding:,} shares × {weight.conversion_ratio} = {economic_value:,.0f} economic units ({weight.economic_weight:.1%})")
    
    return weights

def _deduplicate_classes(weights: List[EconomicWeight]) -> List[EconomicWeight]:
    """Remove duplicate class entries with identical data"""
    if not weights:
        return weights
    
    seen = {}
    deduplicated = []
    
    for weight in weights:
        # Create a key based on class name and key characteristics
        key = (
            weight.class_name,
            weight.ticker,
            weight.shares_outstanding,
            weight.votes_per_share
        )
        
        if key not in seen:
            seen[key] = weight
            deduplicated.append(weight)
        else:
            print(f"    ⚠️  Removing duplicate class entry: {weight.class_name}")
    
    return deduplicated

def analyze_company_economic_weights(company: Dict[str, Any], ticker_map: Dict[str, str]) -> Dict[str, Any]:
    """Analyze economic weights for a single company"""
    company_name = company.get('company_name', 'Unknown')
    primary_ticker = company.get('primary_ticker', 'N/A')
    
    print(f"📊 Starting analysis: {company_name} ({primary_ticker})", flush=True)
    
    # Add rate limiting - be respectful to SEC servers
    time.sleep(0.5)
    
    # Get CIK
    cik = company.get('cik')
    if not cik:
        print(f"  🔍 No CIK in data, attempting lookup by name/ticker...", flush=True)
        cik = lookup_cik_by_name_or_ticker(company_name, primary_ticker, ticker_map)
        if cik:
            company['cik'] = cik
            print(f"  ✅ Found CIK via lookup: {cik}", flush=True)
        else:
            print(f"  ❌ Could not find CIK for {company_name} ({primary_ticker})", flush=True)
            print(f"  🤖 Investigating with AI why no CIK found...", flush=True)
            
            # Use AI to investigate why no CIK - likely delisted/acquired
            investigation = investigate_no_cik_with_ai(company_name, primary_ticker)
            print(f"  📋 AI Investigation: {investigation.get('status')} - {investigation.get('reason')}", flush=True)
            
            # Return special marker for no-CIK companies
            return {
                **company,
                'investigation': investigation,
                'exclude_from_main_output': True
            }
    else:
        print(f"  ✅ Using existing CIK: {cik}", flush=True)
    
    # Get latest filing
    print(f"  📄 Searching for recent SEC filings (10-K/10-Q)...", flush=True)
    filing_info = get_latest_filing(cik)
    
    if not filing_info:
        print(f"  ❌ No recent 10-K/10-Q filings found for {company_name}", flush=True)
        return company
    
    print(f"  ✅ Found {filing_info['form_type']} filing from {filing_info['filing_date']}", flush=True)
    
    # Get filing content
    print(f"  ⬇️  Downloading filing content...", flush=True)
    content = get_filing_content(filing_info)
    
    if not content:
        print(f"  ❌ Could not download filing content for {company_name}", flush=True)
        return company
    
    print(f"  ✅ Downloaded {len(content):,} characters of filing data", flush=True)
    
    # Extract economic weights using AI
    print(f"  🤖 Running AI analysis to extract share class data...", flush=True)
    weights = extract_economic_weights_with_ai(
        content, 
        company_name, 
        primary_ticker
    )
    
    # AI extracted weights are used directly without recalculation
    if weights:
        print(f"  📊 Using AI-extracted share class data directly...")
    else:
        print(f"  ⚠️  No economic data extracted by AI for {company_name}")
    
    # Merge economic data into existing classes array
    updated_classes = []
    existing_classes = company.get('classes', [])
    
    print(f"  🔗 Merging AI results with existing voting data...")
    print(f"     • AI found: {len(weights)} share classes with economic data")
    print(f"     • Existing: {len(existing_classes)} voting classes from input data")

    # Helper: load SEC ticker cache
    def _load_sec_cache():
        cache_path = 'sec_company_tickers_cache.json'
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as cf:
                return json.load(cf)
        except Exception:
            return None

    sec_cache = _load_sec_cache()

    # Helper: load Step 2 ticker mappings (test file preferred)
    def _load_step2_mappings():
        candidates = ['step2_ticker_mappings_test.json', 'step2_ticker_mappings.json']
        for c in candidates:
            if os.path.exists(c):
                try:
                    with open(c, 'r', encoding='utf-8') as sf:
                        data = json.load(sf)
                        results = data.get('results') or data.get('mappings') or []
                        mapping = {}
                        for entry in results:
                            cik_key = entry.get('cik')
                            if not cik_key:
                                continue
                            cik_z = str(cik_key).zfill(10)
                            cm = mapping.setdefault(cik_z, {})
                            for tm in entry.get('ticker_mappings', []):
                                title = tm.get('title_of_class') or tm.get('class_name') or ''
                                tkr = tm.get('ticker')
                                if not title or not tkr:
                                    continue
                                k = _norm_class_name(title)
                                # store multiple variants for robustness
                                variants = set([k, k.lower(), k.upper()])
                                # if class_ prefix present, add letter-only variants
                                if k.startswith('class_'):
                                    tok = k[len('class_'):]
                                    variants.add(tok)
                                    variants.add(tok.lower())
                                    variants.add(tok.upper())
                                    if tok.startswith('num_'):
                                        variants.add('class_' + tok)
                                for v in variants:
                                    if v not in cm:
                                        cm[v] = tkr
                        return mapping
                except Exception:
                    continue
        return None

    step2_map = _load_step2_mappings()
    print(f"    🔎 Step2 mappings loaded for {len(step2_map) if step2_map else 0} CIKs")

    def _pick_ticker(class_key: str, ai_tkr: Optional[str], ai_src: Optional[str], ai_ticker_src: Optional[str], input_tkr: Optional[str], cik_val: Optional[str], primary: Optional[str]) -> str:
        # Debug: show lookup inputs
        print(f"      ▶ Picking ticker for class_key={class_key}, ai_tkr={ai_tkr}, ai_ticker_src={ai_ticker_src}, cik={cik_val}, primary={primary}")
        # 0) If AI explicitly found ticker from Trading Symbol(s) / cover page, prefer it
        try:
            if ai_tkr and ai_ticker_src and any(x in ai_ticker_src.lower() for x in ['cover page table', 'cover page', 'trading symbol']):
                print(f"      ✅ Using AI ticker from cover page: {ai_tkr}")
                return ai_tkr
        except Exception:
            pass
        # 1) Step 2 mapping for this CIK and class (authoritative class-specific tickers)
        try:
            if cik_val and step2_map:
                cik_z = str(cik_val).zfill(10)
                if cik_z in step2_map:
                    print(f"      🔁 Step2 mappings for CIK {cik_z}: {list(step2_map[cik_z].keys())}")
                if cik_z in step2_map and class_key in step2_map[cik_z]:
                    print(f"      ✅ Using STEP2 ticker {step2_map[cik_z][class_key]} for class {class_key}")
                    return step2_map[cik_z][class_key]
        except Exception:
            pass
        # 2) No authoritative ticker found -> treat as non-tradable
        print(f"      ⚪ No class-specific ticker found; marking as non-tradable (ticker='')")
        return ""

    # Helper: normalize class name to a stable key
    def _norm_class_name(name: str) -> str:
        if not name:
            return "__unknown__"
        raw_lower = name.lower().strip()
        # Special cases first (issuer-agnostic)
        if 'preferred' in raw_lower and 'stock' in raw_lower:
            return 'preferred'
        if 'common stock' in raw_lower and 'class' not in raw_lower:
            return 'common'
        # Trim everything after obvious separators like ':' or ';' or '('
        short = name.split(':', 1)[0].split(';', 1)[0].split('(', 1)[0]
        s = short.lower().strip()
        # Normalize non/ voting
        s = re.sub(r"\b(non[- ]?voting|nonvoting)\b", 'non_voting', s)
        # Remove punctuation and extra spaces (keep words for class detection)
        s = re.sub(r"[^a-z0-9\s]", ' ', s)
        s = re.sub(r"\s+", ' ', s).strip()

        # Prefer explicit 'Class X' patterns (letter or numeric)
        m = re.search(r"\bclass\s*([a-z0-9]+)\b", name.lower())
        if m:
            token = m.group(1)
            if token.isdigit():
                return f"class_num_{token}"
            letter = re.search(r"[a-z]", token)
            if letter:
                return f"class_{letter.group(0).upper()}"
            return f"class_{token}"

        # Recognize plain 'common stock'
        if 'common stock' in s:
            return 'common'

        # If short phrase begins with a single-letter token like 'a' or 'b', treat as Class A/B
        m2 = re.match(r"^([a-z])\b", s)
        if m2:
            return f"class_{m2.group(1).upper()}"
    
    # Update existing classes with economic data
    for weight in weights:
        class_key = _norm_class_name(weight.class_name)
        ai_ticker = weight.ticker
        ai_source = weight.source
        ai_ticker_source = weight.ticker_source
        input_ticker = None
        cik_value = None
        
        # Look up existing class data
        existing_class = next((c for c in existing_classes if _norm_class_name(c.get('class_name', '')) == class_key), None)
        
        if existing_class:
            print(f"    • Found existing class data for {existing_class.get('class_name')}")
            input_ticker = existing_class.get('ticker')
            cik_value = existing_class.get('cik')
            
            # Update with AI-extracted data, prioritizing AI ticker over existing
            existing_class['class_name'] = _standardize_class_name(existing_class.get('class_name', ''))
            existing_class['ticker'] = _pick_ticker(class_key, ai_ticker, ai_source, ai_ticker_source, input_ticker, cik_value, primary_ticker)
            existing_class['shares_outstanding'] = weight.shares_outstanding
            existing_class['conversion_ratio'] = weight.conversion_ratio
            existing_class['votes_per_share'] = weight.votes_per_share
            existing_class['source'] = weight.source
            existing_class['ticker_source'] = weight.ticker_source
            
            updated_classes.append(existing_class)
        else:
            print(f"    • No existing data for {weight.class_name}, adding new class")
            # New class entry
            new_class = {
                'class_name': weight.class_name,
                'ticker': _pick_ticker(class_key, ai_ticker, ai_source, ai_ticker_source, input_ticker, cik_value, primary_ticker),
                'cik': cik,
                'shares_outstanding': weight.shares_outstanding,
                'conversion_ratio': weight.conversion_ratio,
                'votes_per_share': weight.votes_per_share,
                'source': weight.source,
                'ticker_source': weight.ticker_source
            }
            
            updated_classes.append(new_class)
    
    # Preserve class-specific tickers; do not override with primary_ticker
    
    # Update company data
    company['classes'] = updated_classes
    
    return company

def lookup_cik_by_name_or_ticker(company_name: str, primary_ticker: str, ticker_map: Dict[str, str]) -> Optional[str]:
    """Lookup CIK using company name or ticker with SEC API"""
    # First try the ticker map
    if primary_ticker and primary_ticker.upper() in ticker_map:
        cik = ticker_map[primary_ticker.upper()]
        print(f"    ✅ Found CIK via ticker {primary_ticker}: {cik}")
        return cik
    
    # For well-known companies, try common ticker variants
    if company_name and 'berkshire' in company_name.lower():
        for variant in ['BRK.A', 'BRK.B', 'BRK-A', 'BRK-B']:
            if variant in ticker_map:
                cik = ticker_map[variant]
                print(f"    ✅ Found Berkshire CIK via {variant}: {cik}")
                return cik
    
    # Try SEC company search API
    headers = {'User-Agent': 'Economic Weight Analysis Tool (contact@example.com)'}
    
    # Search by company name
    if company_name:
        try:
            search_url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(search_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Search through company titles
            for entry in data.values():
                title = entry.get('title', '').lower()
                if company_name.lower() in title or title in company_name.lower():
                    cik = entry.get('cik_str') or str(entry.get('cik', ''))
                    if cik:
                        formatted_cik = f"{int(cik):010d}"
                        print(f"    ✅ Found CIK via company name search: {formatted_cik}")
                        return formatted_cik
        except Exception as e:
            print(f"    → Company name search failed: {e}")
    
    print(f"    ❌ Could not find CIK for {company_name} ({primary_ticker})")
    return None


# Main processing logic
if __name__ == "__main__":
    print(f"=== Economic Weight Analysis Pipeline ===")
    
    # Load input data
    input_file = 'dual_class_output.json'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run 1_dual_class_ingest.py first.")
        # Write minimal output to satisfy pipeline expectation
        output_data = {
            'economic_analysis_date': '2025-09-06',
            'companies_analyzed': 0,
            'successful_analyses': 0,
            'test_mode': False,
            'companies': []
        }
        # Use the test filename as a safe default placeholder
        output_file = 'dual_class_economic_weights_test.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Wrote empty output to {output_file} due to missing input.")
        sys.exit(1)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Standardize class names in input data before processing
    companies = input_data.get('companies', [])
    for company in companies:
        classes = company.get('classes', [])
        for cls in classes:
            if 'class_name' in cls:
                cls['class_name'] = _standardize_class_name(cls['class_name'])
    
    test_mode = input_data.get('test_mode', False)
    
    print(f"Loaded {len(companies)} companies from {input_file}")
    
    # Fetch latest CIK mappings from SEC
    ticker_map = fetch_sec_cik_map()
    
    # Analyze each company
    results = []
    no_cik_companies = []
    
    for company in companies:
        result = analyze_company_economic_weights(company, ticker_map)
        
        # Separate companies without CIKs (likely delisted/acquired)
        if result.get('exclude_from_main_output'):
            no_cik_companies.append(result)
        else:
            results.append(result)
    
    # Prepare main output data (only companies with CIKs)
    output_data = {
        'economic_analysis_date': '2025-09-06',
        'companies_analyzed': len(results),
        'successful_analyses': sum(1 for r in results if r.get('cik')),
        'test_mode': test_mode,
        'companies': results
    }
    
    # Prepare no-CIK output data (delisted/acquired companies)
    no_cik_data = {
        'analysis_date': '2025-09-06',
        'companies_without_cik': len(no_cik_companies),
        'reason': 'Companies likely delisted, acquired, merged, or renamed',
        'companies': no_cik_companies
    }
    
    # Determine output files
    output_file = 'dual_class_economic_weights_test.json' if test_mode else 'dual_class_economic_weights.json'
    no_cik_file = 'no_cik_found_test.json' if test_mode else 'no_cik_found.json'
    
    # Write main output data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Write no-CIK output data if any
    if no_cik_companies:
        with open(no_cik_file, 'w', encoding='utf-8') as f:
            json.dump(no_cik_data, f, indent=2, ensure_ascii=False)
        print(f"📊 {len(no_cik_companies)} companies without CIKs written to {no_cik_file}")
    
    print(f"✅ Analysis complete. {len(results)} companies with CIKs written to {output_file}")
