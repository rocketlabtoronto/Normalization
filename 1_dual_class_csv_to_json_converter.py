#!/usr/bin/env python3
"""
Simplified dual class list ingestion: Extract CIK, ticker, and company name only.
Downstream processing will handle voting structures and other complex data.
"""

import os
import json
import csv
import re
import glob
import requests
from typing import List, Dict, Any

def find_input_file() -> str:
    """
    Searches for the DualClassList CSV file in the input directory.
    Looks for 'DualClassList.csv' first, then tries files with date patterns like 'DualClassList_19Aug2025'.
    Returns the path to the first matching file found.
    """
    # First try the main file without date in input folder
    if os.path.exists('input/DualClassList.csv'):
        return 'input/DualClassList.csv'
    
    # Then try files with date pattern in input folder
    patterns = ['input/DualClassList_19Aug2025.*', 'input/DualClassList*19Aug2025*']
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    
    # Fallback to current directory for backward compatibility
    if os.path.exists('DualClassList.csv'):
        return 'DualClassList.csv'
    
    patterns = ['DualClassList_19Aug2025.*', 'DualClassList*19Aug2025*']
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    raise FileNotFoundError("Could not find DualClassList CSV file in input/ or current directory")

def fetch_sec_cik_map() -> Dict[str, str]:
    """
    Downloads the official SEC company ticker-to-CIK mapping from SEC.gov.
    Creates a dictionary mapping stock tickers (like 'AAPL') to their Central Index Key numbers (like '0000320193').
    CIK numbers are the unique identifiers SEC uses to track companies in their filing system.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {'User-Agent': 'Dual Class Ingest Tool (contact@example.com)'}
    
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

def normalize_ticker(ticker: str) -> str:
    """
    Cleans up stock ticker symbols by removing extra characters and standardizing format.
    Converts things like 'BRK/A' to 'BRK.A' and removes parentheses, quotes, and whitespace.
    This ensures tickers match the format used in SEC databases.
    """
    if not ticker:
        return ""
    
    ticker = ticker.strip().upper()
    # Remove common non-ticker words
    if ticker in ['INC', 'CORP', 'CO', 'LTD']:
        return ""
    
    # Keep only alphanumeric and dots/hyphens
    ticker = re.sub(r'[^A-Z0-9.-]', '', ticker)
    
    # Basic ticker validation (1-5 chars, optional dot extension)
    if re.match(r'^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$', ticker):
        return ticker
    
    return ""

def read_dual_class_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Reads the DualClassList CSV file and extracts basic company information.
    Returns only company name, primary ticker, and CIK for downstream processing.
    """
    companies = []
    
    # Try different encodings
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                reader = csv.reader(f)
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not read file with any supported encoding")
    
    if not rows:
        return []
    
    # Skip the first row (header) and process data
    for i, row in enumerate(rows[1:], 1):  # Skip header
        if not any(cell.strip() for cell in row):  # Skip empty rows
            continue
        
        company_name = row[0].strip() if len(row) > 0 else ""
        ticker = row[1].strip() if len(row) > 1 else ""
        
        # Only process rows with company names
        if company_name:
            normalized_ticker = normalize_ticker(ticker) if ticker else ""
            
            companies.append({
                'company_name': company_name,
                'primary_ticker': normalized_ticker,
                'cik': ""  # Will be filled by enrich_with_cik
            })
    
    return companies

def enrich_with_cik(companies: List[Dict[str, Any]], ticker_map: Dict[str, str]) -> None:
    """
    Adds SEC Central Index Key (CIK) numbers to companies by looking up their ticker symbols.
    Uses the SEC ticker-to-CIK mapping to find the unique identifier for each company.
    This CIK is essential for downloading SEC filings and accessing official financial data.
    """
    for company in companies:
        primary_ticker = company.get('primary_ticker', "")
        
        # Try to find CIK using the primary ticker
        if primary_ticker and primary_ticker in ticker_map:
            company['cik'] = ticker_map[primary_ticker]
        else:
            company['cik'] = ""

def main():
    """
    Simplified dual-class data ingestion process.
    Extracts company name, primary ticker, and CIK only.
    All complex voting structure analysis is left for downstream processing.
    """
    print("Starting simplified dual class list ingestion...")
    
    # Find input file
    try:
        input_file = find_input_file()
        print(f"Found input file: {input_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Fetch SEC ticker to CIK mapping
    print("Fetching SEC ticker to CIK mapping...")
    ticker_map = fetch_sec_cik_map()
    print(f"Loaded {len(ticker_map)} ticker mappings")
    
    # Parse the dual class file
    print("Parsing dual class file...")
    companies = read_dual_class_file(input_file)
    print(f"Found {len(companies)} companies")
    
    # Enrich with CIK data
    print("Enriching with CIK data...")
    enrich_with_cik(companies, ticker_map)
    
    # Count companies with CIK
    with_cik = sum(1 for c in companies if c.get('cik'))
    print(f"Successfully mapped {with_cik} companies to CIKs")
    
    # Prepare simplified output
    output_data = {
        'as_of': '2025-08-19',
        'total_companies': len(companies),
        'companies_with_cik': with_cik,
        'companies': companies
    }
    
    # Write to staging folder for intermediary results
    os.makedirs('staging', exist_ok=True)
    output_file = 'staging/1_dual_class_output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Output written to {output_file}")
    
    # Print sample output
    print(f"\nSample output (first 3 companies):")
    for i, company in enumerate(companies[:3]):
        print(f"  {i+1}. {company['company_name']}")
        print(f"     Ticker: {company['primary_ticker']}")
        print(f"     CIK: {company['cik']}")
        print()

if __name__ == '__main__':
    main()
