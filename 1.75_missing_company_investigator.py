"""
Simplified investigation tool for missing companies.
Focuses on companies without CIK numbers and attempts to find them or investigate why they're missing.

Usage:
python 1.75_missing_company_investigator.py [--ai-check]

Note: For individual company SEC filing analysis, use 1.5_ExtractionSEC.py instead.

Output: writes "staging/1.75_dual_class_output_nocik.json" in the staging directory.
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Optional

from shared_utils import (
    fetch_sec_ticker_map,
    load_json_file,
    save_json_file,
    generate_ticker_variants,
    setup_openai,
    query_openai
)


def openai_investigate_company(symbol: str, company_name: str = None) -> str:
    """
    Uses OpenAI to research what happened to a company that can't be found in SEC databases.
    Asks AI to investigate if the company was acquired, went bankrupt, changed names, or delisted.
    This provides a quick way to understand why a company might be missing from current records.
    """
    company_ref = f"{company_name} ({symbol})" if company_name else symbol
    
    prompt = f"""What happened to the company {company_ref}? Specifically, was it:
- Delisted from a stock exchange? If so, when and why?
- Acquired or merged with another company? If so, by whom and when?
- Filed for bankruptcy? If so, when?
- Suspended from trading? If so, when and why?
- Still actively trading but under a different ticker?

Please provide a brief, factual response with specific dates if available."""
    
    try:
        setup_openai()
        response = query_openai(prompt, max_tokens=150)
        return response.strip() if response else "Unable to get AI response"
    except Exception as e:
        return f"Error querying AI: {str(e)}"


def investigate_company_with_ai(company: Dict, ticker_map: Dict) -> Dict:
    """
    Investigates a company without a CIK using AI to find out what happened to it.
    This tries to determine if the company was delisted, acquired, or went out of business,
    which would explain why it doesn't have a current SEC CIK number.
    """
    symbol = company.get("primary_ticker", "").strip()
    name = company.get("company_name", "").strip()
    
    if not symbol and not name:
        return {**company, "ai_investigation": "No symbol or name available for investigation"}
    
    # Try to find CIK using ticker variants
    variants = generate_ticker_variants(symbol) if symbol else []
    found_cik = None
    
    for variant in variants:
        if variant in ticker_map:
            found_cik = ticker_map[variant]
            break
    
    if found_cik:
        return {
            **company, 
            "cik": found_cik,
            "ai_investigation": f"Found CIK {found_cik} using ticker variant"
        }
    
    # Use AI to investigate what happened
    ai_response = openai_investigate_company(symbol, name)
    
    return {
        **company,
        "ai_investigation": ai_response,
        "investigation_note": "Company not found in current SEC databases"
    }


def process_no_cik_file(input_path: str = "staging/1.75_dual_class_output_nocik.json",
                       output_path: str = "staging/1.75_dual_class_output_investigated.json",
                       ai_check: bool = False) -> Dict:
    """
    Processes companies that don't have CIK numbers and investigates what happened to them.
    Can optionally use AI to research why companies are missing from SEC databases.
    """
    print(f"Loading input file: {input_path}")
    try:
        data = load_json_file(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return {"error": "Could not load input file"}
    
    if not data or "companies" not in data:
        print("No companies found in input file")
        return {"error": "No companies found in input file"}
    
    companies = [c for c in data["companies"] if not c.get("cik")]
    print(f"Found {len(companies)} companies without CIKs")
    
    if not companies:
        return {"total": 0, "summary": "No companies without CIKs found"}
    
    # Get SEC ticker mapping for CIK lookup attempts
    print("Fetching SEC ticker to CIK mapping...")
    ticker_map = fetch_sec_ticker_map()
    print(f"Loaded {len(ticker_map)} ticker mappings")
    
    investigated = []
    summary = {
        "found_cik": 0,
        "ai_investigated": 0,
        "no_investigation": 0
    }
    
    for i, company in enumerate(companies):
        print(f"Processing {i+1}/{len(companies)}: {company.get('company_name', 'Unknown')}")
        
        if ai_check:
            result = investigate_company_with_ai(company, ticker_map)
            if result.get("cik"):
                summary["found_cik"] += 1
            elif result.get("ai_investigation"):
                summary["ai_investigated"] += 1
        else:
            # Simple CIK lookup without AI
            symbol = company.get("primary_ticker", "").strip()
            variants = generate_ticker_variants(symbol) if symbol else []
            found_cik = None
            
            for variant in variants:
                if variant in ticker_map:
                    found_cik = ticker_map[variant]
                    break
            
            if found_cik:
                result = {**company, "cik": found_cik, "found_via": "ticker_variant"}
                summary["found_cik"] += 1
            else:
                result = {**company, "investigation_note": "No CIK found, AI investigation not enabled"}
                summary["no_investigation"] += 1
        
        investigated.append(result)
    
    # Save results
    output_data = {
        "input_file": input_path,
        "processed_at": data.get("as_of", "unknown"),
        "total_processed": len(investigated),
        "summary": summary,
        "companies": investigated
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json_file(output_data, output_path)
    print(f"Results saved to {output_path}")
    
    return {"total": len(companies), "summary": summary}


if __name__ == "__main__":
    import sys
    # Batch processing mode for companies missing CIK numbers
    ai_check = "--ai-check" in sys.argv or "--openai" in sys.argv
    result = process_no_cik_file(ai_check=ai_check)
    print(json.dumps(result, indent=2))
