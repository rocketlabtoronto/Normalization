#!/usr/bin/env python3
"""
Simplified economic weight analyzer using shared utilities.
Eliminates code duplication with other scripts.
"""

import os
import json
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys

from shared_utils import (
    make_request,
    fetch_sec_ticker_map,
    load_json_file,
    save_json_file,
    setup_openai,
    query_openai,
    SEC_COMPANY_TICKERS_URL
)

# Attempt to load .env if python-dotenv is installed
try:  # optional
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

@dataclass
class EconomicWeight:
    """Data structure for economic weight information"""
    class_name: str
    ticker: str = ""
    shares_outstanding: Optional[float] = None
    economic_weight: Optional[float] = None  # Proportion of total economic value (0.0 to 1.0)
    conversion_ratio: Optional[float] = None  # How many units of primary class this represents
    votes_per_share: Optional[float] = None  # Voting power per share
    source: str = ""  # Source of the data (AI query, etc.)
    ticker_source: Optional[str] = None  # Where the ticker came from

def investigate_no_cik_with_ai(company_name: str, primary_ticker: str) -> Dict[str, Any]:
    """
    Uses AI to research why a company cannot be found in SEC databases.
    When a company has no CIK number, it usually means they were acquired, went bankrupt,
    delisted, or changed their name. This function asks AI to investigate and determine
    what happened to explain why the company is missing from current SEC records.
    """
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
        result = query_openai(prompt, max_tokens=300)
        
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
                
                data = json.loads(candidate)
                return data
        
        return {"status": "unknown", "reason": "Could not parse AI response"}
        
    except Exception as e:
        print(f"    ❌ AI investigation failed: {e}")
        return {"status": "unknown", "reason": f"Error: {e}"}


def lookup_cik_by_name_or_ticker(company_name: str, primary_ticker: str, ticker_map: Dict[str, str]) -> Optional[str]:
    """
    Attempts to find a company's SEC Central Index Key (CIK) number using various search methods.
    First tries direct ticker lookup, then tries company name matching, and handles special cases
    like Berkshire Hathaway. The CIK is essential for accessing SEC filings and official data.
    Returns None if the company cannot be found in SEC databases.
    """
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
    
    # Try SEC company search API using shared utilities
    if company_name:
        try:
            response = make_request(SEC_COMPANY_TICKERS_URL)
            if response:
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

def query_ai_for_economic_weights(company_name: str, ticker: str, cik: str, existing_classes: List[Dict]) -> List[EconomicWeight]:
    """Query OpenAI for economic weight information based on existing class structure
    This approach enriches known classes rather than trying to discover them from scratch.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('LLM_MODEL', 'gpt-4o')

    if not api_key:
        print(f"    ❌ OPENAI_API_KEY not set, skipping AI analysis")
        return []

    print(f"    🤖 Querying AI for specific share class information...")
    
    # Build class context from existing data
    class_context = []
    for cls in existing_classes:
        class_name = cls.get('class_name', '')
        votes = cls.get('votes_per_share', 'unknown')
        class_context.append(f"- {class_name}: {votes} votes per share")
    
    class_info = "\n".join(class_context)
    
    # Detect time-phased voting patterns and other single-class structures
    is_time_phased = any(
        'time-phased' in cls.get('class_name', '').lower() or 
        'until stock held for' in cls.get('class_name', '').lower() or
        'years, then' in cls.get('class_name', '').lower()
        for cls in existing_classes
    )
    
    # Detect formula-based voting patterns
    has_formula_voting = any(
        'formula (' in cls.get('class_name', '').lower() or
        'currently about' in cls.get('class_name', '').lower() or
        'currently approximately' in cls.get('class_name', '').lower()
        for cls in existing_classes
    )
    
    # Detect single class with descriptive labels that aren't actual separate classes
    has_descriptive_labels = any(
        cls.get('votes_per_share') is None or
        cls.get('class_name', '').lower().startswith(('time-phased', 'voting:', 'non-voting:', 'unequal voting'))
        for cls in existing_classes
    )
    
    prompt = f"""You are analyzing the share structure for {company_name} (ticker: {ticker}, CIK: {cik}).

I have these class descriptions from input data:
{class_info}

CRITICAL ANALYSIS PATTERNS:

1. TIME-PHASED VOTING DETECTION:
   - If voting changes based on holding period ("1 vote until 4 years, then 10 votes"), this is ONE class of stock, not multiple classes
   - Time-phased voting means voting power varies by how long shares are held, but there's only one type of share
   - Example: Aflac has ONE common stock class with time-phased voting (1 vote initially, 10 votes after 4 years)
   - Do NOT create separate classes for different voting phases

2. SINGLE CLASS WITH DESCRIPTIVE LABELS:
   - If labels are "Time-phased voting:" and "1 vote per share until...", this describes ONE class
   - If labels are "Voting: 1 vote" and "Non-voting: 0 votes" but same ticker, check if actually separate classes
   - Descriptive headers (null votes_per_share) are NOT separate classes

3. TRUE DUAL-CLASS STRUCTURES:
   - Class A vs Class B with different tickers or economic rights
   - Preferred vs Common stock with different liquidation preferences
   - Separate share classes with distinct economic or voting characteristics

Research this company's MOST RECENT quarterly earnings report and provide the ACTUAL share class structure.

Return ONLY JSON in this exact format (no markdown, no commentary):

{{
  "classes": [
    {{
      "class_name": "ACTUAL_CLASS_NAME",
      "ticker": "EXACT_TICKER" or "" if not publicly traded,
      "ticker_source": "known publicly traded" or "not traded",
      "shares_outstanding": number (exact shares from latest 10-Q/10-K),
      "conversion_ratio": number (economic conversion ratio - NEVER null),
      "votes_per_share": number (base voting power - use weighted average for time-phased),
      "is_publicly_traded": true or false
    }}
  ]
}}

RESEARCH INSTRUCTIONS:
1. SEARCH FOR LATEST QUARTERLY DATA (March 2025, December 2024, etc.):
   - Look for the most recent 10-Q filing data in your knowledge base
   - Find "shares issued and outstanding" or "outstanding shares" 
   - Use EXACT numbers from the latest quarterly report

2. FOR TIME-PHASED OR SINGLE-CLASS COMPANIES:
   - Return ONLY ONE class entry for the actual share class
   - Use the primary ticker for publicly traded shares
   - For time-phased voting, use the base voting rate (e.g., 1 vote for Aflac)
   - Do NOT create multiple entries for voting descriptions

3. FOR FORMULA-BASED VOTING:
   - If voting is described as "Formula (currently about X votes per share)", use the EXACT number X
   - Do NOT simplify large formula-based voting numbers (e.g., 3,414,443 should stay 3414443.0)
   - Formula-based voting typically indicates complex capital structures with extreme voting ratios
   - Preserve the actual voting power, even if it's millions of votes per share

4. STANDARDIZE class_name to actual share class:
   - "Common Stock" for single class companies
   - "Class A" and "Class B" only if truly separate classes
   - Remove descriptive text like "Time-phased voting:"

4. TICKER SYMBOLS - Use your knowledge of actual trading symbols:
   - For publicly traded classes, provide the exact ticker symbol
   - If a class is not publicly traded, set ticker: ""
   - Many companies have only one publicly traded class

5. CONVERSION RATIO - ECONOMIC EQUIVALENCE (NEVER NULL):
   - If shares have equal economic rights: set to 1
   - If Class A converts to X Class B shares: Class A = X, Class B = 1
   - For single class companies: always 1
   - NEVER use null - always provide a number

6. VOTING RIGHTS: Provide base voting power per share
   - For time-phased voting, use the initial voting rate
   - Research and verify EXACT voting structure from your knowledge base
   - Use your comprehensive financial knowledge for accurate voting ratios
   - Never leave votes_per_share as null

Company: {company_name} (Primary ticker: {ticker})
Detected pattern: {"Time-phased voting" if is_time_phased else "Formula-based voting" if has_formula_voting else "Standard dual-class" if not has_descriptive_labels else "Single class with descriptive labels"}"""

    try:
        try:
            import openai  # type: ignore
        except ImportError:
            print(f"    ❌ openai package not installed, skipping AI analysis")
            return []

        result: Optional[str] = None
        last_error: Optional[Exception] = None
        
        for attempt in range(1, 4):  # 3 attempts with retry
            try:
                print(f"    📡 Calling OpenAI API attempt {attempt}/3 ({model})...")
                if hasattr(openai, 'OpenAI'):
                    client = openai.OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert financial analyst with knowledge of dual-class share structures. Return ONLY strict JSON as specified."},
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
                            {"role": "system", "content": "You are an expert financial analyst with knowledge of dual-class share structures. Return ONLY strict JSON as specified."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    result = response['choices'][0]['message']['content'] if response.get('choices') else ''  # type: ignore
                
                if not result or not str(result).strip():
                    raise ValueError('Empty LLM response')
                
                result = result.strip()
                preview = result[:120].replace('\n', ' ')
                print(f"    ✅ Received response ({len(result)} chars) | preview: {preview}...")
                
                # Parse the JSON response
                parsed_weights = _parse_ai_json_to_weights(result, company_name, ticker)
                if parsed_weights is not None:
                    return parsed_weights
                else:
                    raise ValueError('JSON parse failed after extraction logic')
                    
            except Exception as e:
                last_error = e
                print(f"    ⚠️ Attempt {attempt} failed: {e}")
                if attempt < 3:
                    sleep_for = 1.5 * attempt
                    print(f"    ⏳ Backoff {sleep_for:.1f}s before retry...")
                    time.sleep(sleep_for)
                continue
        
        print(f"    ❌ All AI attempts failed: {last_error}")
        return []
        
    except Exception as outer:
        print(f"Warning: AI analysis failed for {company_name}: {outer}")
        return []

def _standardize_class_name(raw_name: str) -> str:
    """Standardize class names while preserving important distinctions like 'Class A Common' vs 'Common'
    Also handle time-phased voting and other descriptive patterns that aren't actual class names."""
    if not raw_name:
        return "Class A"  # Default fallback
    
    import re
    
    # Handle time-phased voting and descriptive labels that aren't real class names
    lower_name = raw_name.lower().strip()
    
    # Time-phased voting patterns - these are descriptive, not class names
    if any(pattern in lower_name for pattern in [
        'time-phased voting',
        'until stock held for',
        'years, then',
        'vote per share until',
        'phased voting'
    ]):
        return "Common Stock"  # Time-phased voting applies to common stock
    
    # Formula-based voting patterns - extract the actual class name
    if any(pattern in lower_name for pattern in [
        'formula (',
        'currently about',
        'currently approximately',
        'formula-based'
    ]):
        # Extract class name before the formula description
        if ':' in raw_name:
            before_colon = raw_name.split(':', 1)[0].strip()
            # Clean up the class name part
            if 'class' in before_colon.lower():
                match = re.search(r'Class\s+([A-Z])', before_colon, re.IGNORECASE)
                if match:
                    letter = match.group(1).upper()
                    return f"Class {letter}"
            return before_colon
        return "Class B"  # Formula-based voting is typically Class B
    
    # Descriptive voting labels that aren't class names
    if any(pattern in lower_name for pattern in [
        'voting:',
        'non-voting:',
        'unequal voting structure',
        'votes per share',
        'voting structure'
    ]) and ':' in raw_name:
        # Extract the actual class info if present
        if 'class a' in lower_name:
            return "Class A"
        elif 'class b' in lower_name:
            return "Class B"
        elif 'common' in lower_name:
            return "Common Stock"
        else:
            return "Common Stock"  # Default for unclear descriptive labels
    
    # Handle original input format with voting descriptions
    if ':' in raw_name:
        base_name = raw_name.split(':', 1)[0].strip()
    else:
        base_name = raw_name.strip()
    
    # Preserve A.O. Smith style naming
    if 'class a common' in base_name.lower():
        return "Class A Common"
    if base_name.lower() in ['common', 'common stock'] or base_name.lower().startswith('common'):
        return "Common Stock"
    
    # Handle "Class A Common Stock" pattern - extract just "Class A"
    match = re.search(r'Class\s+([A-Z])\s+Common\s+Stock', base_name, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        return f"Class {letter}"
    
    # Extract the class letter using regex for standard patterns
    # Look for patterns like "Class A", "Class B:", "Class A Common", etc.
    match = re.search(r'Class\s+([A-Z])', base_name, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        return f"Class {letter}"
    
    # Handle preferred stock
    if 'preferred' in base_name.lower():
        return "Preferred Stock"
    
    # Fallback: look for single letters that might indicate class
    match = re.search(r'\b([A-Z])\b', base_name)
    if match:
        letter = match.group(1).upper()
        return f"Class {letter}"
    
    # Final fallback based on common terms
    if 'common' in base_name.lower():
        return "Common Stock"
    elif 'b' in base_name.lower():
        return "Class B"
    else:
        return "Class A"

def _clean_ticker_by_trading_status(ticker: str, is_publicly_traded: bool) -> str:
    """Return empty string for non-traded securities, clean ticker otherwise"""
    if not is_publicly_traded:
        return ""
    return ticker.strip() if ticker else ""

def _parse_ai_json_to_weights(raw: str, company_name: str, primary_ticker: str) -> Optional[List[EconomicWeight]]:
    """Parse AI JSON response into EconomicWeight objects"""
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
        
        # Extract JSON object using brace scanning
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
                source='AI direct query',
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
        print(f"       Raw snippet (first 200 chars): {raw[:200].replace(chr(10),' ')}...")
        return None

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
    """
    Analyzes a single company to determine precise economic weights and voting structures for each share class.
    Uses AI to query current financial data and determine exact voting rights, shares outstanding,
    and economic relationships between different classes of stock. This is the main function that
    converts basic company information into detailed financial analysis with accurate voting structures.
    """
    company_name = company.get('company_name', 'Unknown')
    primary_ticker = company.get('primary_ticker', 'N/A')
    
    print(f"[ANALYSIS] Starting analysis: {company_name} ({primary_ticker})", flush=True)
    
    # Add rate limiting - be respectful to API servers
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
    
    # Query AI directly for economic weights (no SEC filing download)
    print(f"  🤖 Querying AI directly for share class data...", flush=True)
    existing_classes = company.get('classes', [])
    weights = query_ai_for_economic_weights(company_name, primary_ticker, cik, existing_classes)
    
    if not weights:
        print(f"  ⚠️  No economic data extracted by AI for {company_name}")
        return company
    
    # AI results take highest precedence - completely replace CSV data
    updated_classes = []
    existing_classes = company.get('classes', [])
    
    print(f"  🔗 Using AI results as authoritative source (highest precedence)...")
    print(f"     • AI found: {len(weights)} share classes with economic data")
    print(f"     • CSV had: {len(existing_classes)} classes (will be replaced if AI has data)")

    # Special handling for time-phased voting companies
    time_phased_patterns = [
        'time-phased voting',
        'until stock held for',
        'years, then',
        'vote per share until'
    ]
    
    # Special handling for formula-based voting companies  
    formula_patterns = [
        'formula (',
        'currently about',
        'currently approximately', 
        'formula-based'
    ]
    
    is_time_phased = any(
        any(pattern in cls.get('class_name', '').lower() for pattern in time_phased_patterns)
        for cls in existing_classes
    )
    
    has_formula_voting = any(
        any(pattern in cls.get('class_name', '').lower() for pattern in formula_patterns)
        for cls in existing_classes
    )
    
    if is_time_phased:
        print(f"    [DETECTION] Detected time-phased voting structure - consolidating to single class")
        # For time-phased voting, we should have only one class in the output
        # Find a class with actual voting data (not null) - prioritize non-descriptive headers
        primary_class = None
        
        # First, try to find a class that doesn't start with descriptive headers but has voting data
        for cls in existing_classes:
            class_name = cls.get('class_name', '').lower()
            # Skip pure descriptive headers
            if class_name.startswith('time-phased voting:'):
                continue
            # Take any class with voting data for time-phased scenarios
            if cls.get('votes_per_share') is not None:
                primary_class = cls
                break
        
        # If no primary class found, take any class with votes
        if not primary_class:
            for cls in existing_classes:
                if cls.get('votes_per_share') is not None:
                    primary_class = cls
                    break
        
        if primary_class and weights:
            # Use AI data but ensure only one class
            weight = weights[0]  # Take first AI result
            consolidated_class = {
                'class_name': "Common Stock",  # Standardize to Common Stock for time-phased
                'ticker': primary_class.get('ticker', primary_ticker),
                'shares_outstanding': weight.shares_outstanding,
                'conversion_ratio': weight.conversion_ratio or 1,
                'votes_per_share': primary_class.get('votes_per_share', weight.votes_per_share),
                'source': weight.source,
                'ticker_source': weight.ticker_source
            }
            updated_classes.append(consolidated_class)
        elif primary_class:
            # Fallback - use primary class data without AI enhancement
            consolidated_class = {
                'class_name': "Common Stock",
                'ticker': primary_class.get('ticker', primary_ticker),
                'votes_per_share': primary_class.get('votes_per_share'),
                'conversion_ratio': 1
            }
            updated_classes.append(consolidated_class)
        else:
            # Final fallback if no good primary class found
            updated_classes = existing_classes
        
        # Update company data and return early
        company['classes'] = updated_classes
        return company

    # If AI provided results, use them as the authoritative source
    if weights:
        print(f"    ✅ Using AI-generated class structure (highest precedence)")
        
        # Convert AI weights directly to class entries
        for weight in weights:
            ai_class = {
                'class_name': weight.class_name,
                'ticker': weight.ticker,
                'shares_outstanding': weight.shares_outstanding,
                'conversion_ratio': weight.conversion_ratio or 1,
                'votes_per_share': weight.votes_per_share,
                'source': weight.source,
                'ticker_source': weight.ticker_source
            }
            updated_classes.append(ai_class)
            print(f"    • AI class: {weight.class_name} - {weight.votes_per_share} votes per share")
    
    else:
        print(f"    ⚠️  No AI data available, falling back to CSV data")
        # Fallback to existing CSV data if AI failed
        updated_classes = existing_classes
    
    # Update company data
    company['classes'] = updated_classes
    
    return company


# Main processing logic
if __name__ == "__main__":
    """
    Main entry point for the economic weight analysis pipeline.
    Takes dual-class company data and enriches it with precise voting rights, shares outstanding,
    and economic weight information using AI queries. Creates the final dataset with accurate
    financial data for dual-class share analysis, replacing basic CSV data with current information.
    """
    print(f"=== Economic Weight Analysis Pipeline (OpenQuestion Mode) ===")
    
    # Load input data
    input_file = 'staging/1_dual_class_output.json'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run 1_dual_class_csv_to_json_converter.py first.")
        # Write minimal output to satisfy pipeline expectation
        output_data = {
            'economic_analysis_date': '2025-09-06',
            'companies_analyzed': 0,
            'successful_analyses': 0,
            'test_mode': False,
            'companies': []
        }
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        # Use the test filename as a safe default placeholder
        output_file = 'results/3_dual_class_economic_weights_test.json'
        save_json_file(output_data, output_file)
        print(f"Wrote empty output to {output_file} due to missing input.")
        sys.exit(1)
    
    input_data = load_json_file(input_file)
    if not input_data:
        sys.exit(1)
    
    # Standardize class names in input data before processing
    companies = input_data.get('companies', [])
    for company in companies:
        classes = company.get('classes', [])
        for cls in classes:
            if 'class_name' in cls:
                cls['class_name'] = _standardize_class_name(cls['class_name'])
    
    test_mode = input_data.get('test_mode', False)
    
    print(f"Loaded {len(companies)} companies from {input_file}")
    print(f"Mode: OpenQuestion (Direct AI queries - no SEC filing downloads)")
    
    # Fetch latest CIK mappings from SEC using shared utility
    ticker_map = fetch_sec_ticker_map()
    
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
    os.makedirs('results', exist_ok=True)
    output_file = 'results/3_dual_class_economic_weights_test.json' if test_mode else 'results/3_dual_class_economic_weights.json'
    no_cik_file = 'results/1.75_no_cik_found_test.json' if test_mode else 'results/1.75_no_cik_found.json'
    
    # Write main output data
    save_json_file(output_data, output_file)
    
    # Write no-CIK output data if any
    if no_cik_companies:
        save_json_file(no_cik_data, no_cik_file)
        print(f"📊 {len(no_cik_companies)} companies without CIKs written to {no_cik_file}")
    
    print(f"✅ Analysis complete. {len(results)} companies with CIKs written to {output_file}")
