#!/usr/bin/env python3
"""
Ingest DualClassList_19Aug2025 and output JSON with CIK, company name, primary ticker, and voting rights by class.
"""

import os
import json
import csv
import re
import glob
import requests
from typing import List, Dict, Any, Optional

def find_input_file() -> str:
    """Find DualClassList_19Aug2025 file in current directory."""
    patterns = ['DualClassList_19Aug2025.*', 'DualClassList*19Aug2025*']
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    raise FileNotFoundError("Could not find DualClassList_19Aug2025 file")

def fetch_sec_cik_map() -> Dict[str, str]:
    """Fetch SEC ticker to CIK mapping."""
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
    """Clean and normalize ticker symbol."""
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

def parse_voting_rights(text: str) -> Optional[float]:
    """Parse voting rights from text."""
    if not text:
        return None
    
    text = str(text).lower().strip()
    
    # Handle non-voting
    if any(term in text for term in ['non-voting', 'no voting', 'no vote']):
        return 0.0
    
    # Extract numeric value
    numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
    if numbers:
        return float(numbers[0])
    
    # Handle text numbers
    text_numbers = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    for word, num in text_numbers.items():
        if word in text:
            return float(num)
    
    return None

def read_dual_class_file(filepath: str) -> List[Dict[str, Any]]:
    """Read and parse the dual class listing file."""
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
    current_company = None
    current_ticker = None
    current_classes = []
    
    for i, row in enumerate(rows[1:], 1):  # Skip header
        if not any(cell.strip() for cell in row):  # Skip empty rows
            continue
        
        company_name = row[0].strip() if len(row) > 0 else ""
        ticker = row[1].strip() if len(row) > 1 else ""
        voting_structure = row[2].strip() if len(row) > 2 else ""
        
        # If we have a company name, this starts a new company
        if company_name:
            # Save previous company if it exists
            if current_company and current_classes:
                companies.append({
                    'company_name': current_company,
                    'primary_ticker': current_ticker,
                    'classes': current_classes
                })
            
            # Start new company
            current_company = company_name
            current_ticker = normalize_ticker(ticker) if ticker else ""
            current_classes = []
            
            # Parse the voting structure for this company
            if voting_structure:
                classes = parse_voting_structure(voting_structure, current_ticker)
                current_classes.extend(classes)
        
        # If no company name but we have voting info, add to current company
        elif voting_structure and current_company:
            classes = parse_voting_structure(voting_structure, current_ticker)
            current_classes.extend(classes)
    
    # Don't forget the last company
    if current_company and current_classes:
        companies.append({
            'company_name': current_company,
            'primary_ticker': current_ticker,
            'classes': current_classes
        })
    
    return companies

def parse_voting_structure(structure: str, primary_ticker: str = "") -> List[Dict[str, Any]]:
    """Parse voting structure text into class information."""
    classes = []
    
    # Handle multi-line structures (split by newlines)
    lines = structure.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to extract class information
        class_match = re.search(r'(class\s+[A-Z0-9]+(?:\s+[A-Za-z]+)?|common|non-voting|voting)[:=]\s*(.+)', line, re.IGNORECASE)
        
        if class_match:
            class_name = class_match.group(1).strip()
            voting_info = class_match.group(2).strip()
            
            # Parse voting rights
            votes = parse_voting_rights(voting_info)
            
            # Try to extract ticker from the voting info or use primary ticker
            ticker = ""
            if primary_ticker:
                ticker = primary_ticker
            
            classes.append({
                'class_name': f"{class_name}: {voting_info}",
                'ticker': ticker,
                'votes_per_share': votes
            })
        else:
            # Fallback: treat the whole line as a class
            votes = parse_voting_rights(line)
            classes.append({
                'class_name': line,
                'ticker': primary_ticker,
                'votes_per_share': votes
            })
    
    return classes

def determine_primary_ticker(classes: List[Dict[str, Any]], provided_ticker: str = "") -> str:
    """Determine the primary ticker from class information."""
    # If we have a provided ticker, use it
    if provided_ticker:
        return provided_ticker
    
    # Priority 1: Class with 1 vote per share
    for cls in classes:
        if cls.get('votes_per_share') == 1.0 and cls.get('ticker'):
            return cls['ticker']
    
    # Priority 2: Any class with voting rights and ticker
    for cls in classes:
        votes = cls.get('votes_per_share')
        if votes and votes > 0 and cls.get('ticker'):
            return cls['ticker']
    
    # Priority 3: First ticker found
    for cls in classes:
        if cls.get('ticker'):
            return cls['ticker']
    
    return ""

def enrich_with_cik(companies: List[Dict[str, Any]], ticker_map: Dict[str, str]) -> None:
    """Add CIK information to companies using ticker lookup."""
    for company in companies:
        cik = ""
        primary_ticker = company.get('primary_ticker', "")
        
        # If no primary ticker set, determine it
        if not primary_ticker:
            primary_ticker = determine_primary_ticker(company['classes'])
            company['primary_ticker'] = primary_ticker
        
        # Try to find CIK using the primary ticker first
        if primary_ticker and primary_ticker in ticker_map:
            cik = ticker_map[primary_ticker]
        
        # If no CIK found with primary ticker, try any ticker from the classes
        if not cik:
            for cls in company['classes']:
                ticker = cls.get('ticker')
                if ticker and ticker in ticker_map:
                    cik = ticker_map[ticker]
                    break
        
        company['cik'] = cik

def main():
    print("Starting dual class list ingestion...")
    
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
    
    # Prepare output
    output_data = {
        'as_of': '2025-08-19',
        'total_companies': len(companies),
        'companies_with_cik': with_cik,
        'companies': companies
    }
    
    # Write to file
    output_file = 'dual_class_output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Output written to {output_file}")
    
    # Also print to console
    print("\nJSON Output:")
    print(json.dumps(output_data, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
