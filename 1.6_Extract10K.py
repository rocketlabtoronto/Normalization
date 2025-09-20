"""
10-K Equity Class Extraction Tool
Extracts equity class details from 10-K filings and structures them into JSON format.

This script focuses on extracting from standardized 10-K sections:
1. Cover page - Registered securities and outstanding shares
2. Part II, Item 8 - Stockholders' Equity notes 
3. Exhibit 4 - Description of Registrant's Securities
4. Exhibits 3.1/3.2 - Charter & Bylaws
5. Item 5 - Market for Common Equity

Usage:
python 1.6_Extract10K.py --cik 0000123456 [--symbol TICK --exchange NASDAQ]

Input:
- Uses 10-K files downloaded by 1.5_Download10K.py from sec_filings/10K/ folder

Output: 
- Structured equity data saved to staging/cik_{cik}_equity_extraction.json
"""
from __future__ import annotations

import json
import re
import datetime
import os
import sys
from typing import Dict, List, Optional, Union, Any
from bs4 import BeautifulSoup
from shared_utils import (
    save_json_file, 
    load_json_file
)


def find_10k_filing(cik: str) -> Optional[str]:
    """
    Finds the downloaded 10-K filing for a given CIK from the sec_filings/10K folder.
    Returns the file path if found, None otherwise.
    """
    cik_padded = str(cik).zfill(10)
    filing_dir = "sec_filings/10K"
    
    if not os.path.exists(filing_dir):
        return None
    
    # Look for files starting with the CIK
    for filename in os.listdir(filing_dir):
        if filename.startswith(cik_padded) and filename.endswith('.htm'):
            return os.path.join(filing_dir, filename)
    
    return None


def extract_cover_page_data(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extracts equity information from the 10-K cover page.
    Looks for registered securities under Exchange Act §12(b) and outstanding share counts.
    """
    cover_data = {
        "note_sections": [],
        "registered_securities": [],
        "outstanding_shares": {},
        "trading_symbols": []
    }
    
    # Get the first 30,000 characters which should cover the cover page and initial sections
    text = soup.get_text()[:30000]
    html_content = str(soup)[:30000]
    
    # Remove XBRL metadata noise patterns
    xbrl_patterns = [
        r'flws-\d+[A-Za-z0-9]*',
        r'us-gaap:[A-Za-z0-9]+Member',
        r'iso4217:[A-Z]+',
        r'xbrli:[a-z]+',
        r'srt:[A-Za-z0-9]+Member',
        r'http://[^\s]+',
        r'\d{10,}',  # Long numeric sequences
        r'Member\d{4}-\d{2}-\d{2}',
        r'[A-Za-z]+Member[A-Za-z]*Member'
    ]
    
    clean_text = text
    for pattern in xbrl_patterns:
        clean_text = re.sub(pattern, ' ', clean_text)
    
    # Clean up excessive whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    # Look for meaningful cover page sections
    cover_sections = []
    
    # 1. SEC Form Header
    sec_header_match = re.search(
        r'(UNITED STATES SECURITIES AND EXCHANGE COMMISSION.*?FORM 10-K.*?fiscal year ended [^.]+)',
        clean_text, re.IGNORECASE | re.DOTALL
    )
    if sec_header_match:
        header_text = sec_header_match.group(1)[:500]
        cover_sections.append({
            "heading": "SEC Form 10-K Header",
            "content_preview": header_text
        })
    
    # 2. Company Identification
    company_match = re.search(
        r'(Commission File No\..*?Employer Identification No\..*?principal executive offices.*?\(\d{3}\)\s*\d{3}-\d{4})',
        clean_text, re.IGNORECASE | re.DOTALL
    )
    if company_match:
        company_text = company_match.group(1)[:400]
        cover_sections.append({
            "heading": "Company Identification",
            "content_preview": company_text
        })
    
    # 3. Securities Registration Table - look for the actual table content
    securities_table_match = re.search(
        r'(Securities registered pursuant to Section 12\(b\).*?Title of each class.*?Trading symbol.*?Name of each exchange.*?Class A common stock.*?FLWS.*?Nasdaq)',
        clean_text, re.IGNORECASE | re.DOTALL
    )
    if securities_table_match:
        table_text = securities_table_match.group(1)[:400]
        cover_sections.append({
            "heading": "Securities Registration Table",
            "content_preview": table_text
        })
    
    # 4. Outstanding Shares Information - look for specific share count mentions
    shares_match = re.search(
        r'(\d{1,3}(?:,\d{3})*\s*\([Nn]umber of shares of class A common stock outstanding.*?\)\s*\d{1,3}(?:,\d{3})*\s*\([Nn]umber of shares of class B common stock outstanding)',
        clean_text, re.IGNORECASE | re.DOTALL
    )
    if shares_match:
        shares_text = shares_match.group(1)[:300]
        cover_sections.append({
            "heading": "Outstanding Shares Summary",
            "content_preview": shares_text
        })
    
    # 5. Business Description Section
    business_match = re.search(
        r'(Item 1\.\s*BUSINESS.*?The Company.*?(?:is a leading|provides|operates).*?)',
        clean_text, re.IGNORECASE | re.DOTALL
    )
    if business_match:
        business_text = business_match.group(1)[:500]
        cover_sections.append({
            "heading": "Business Description",
            "content_preview": business_text
        })
    
    cover_data["note_sections"] = cover_sections
    
    # Extract registered securities with cleaner patterns
    securities_patterns = [
        r'Class [AB] common stock',
        r'common stock[,\s]*[Cc]lass [AB]',
        r'preferred stock',
        r'warrants'
    ]
    
    for pattern in securities_patterns:
        matches = re.findall(pattern, clean_text, re.IGNORECASE)
        for match in matches:
            clean_security = match.strip()
            if clean_security and clean_security not in cover_data["registered_securities"]:
                cover_data["registered_securities"].append(clean_security)
    
    # Extract ticker symbols using both HTML and text patterns
    ticker_patterns = [
        r"<ix:nonNumeric[^>]*name=\"dei:TradingSymbol\"[^>]*>([A-Z]{1,6})</ix:nonNumeric>",
        r"ticker symbol[:\s]+([A-Z]{2,6})",
        r"symbol[:\s]+([A-Z]{2,6})"
    ]
    
    for ticker_pattern in ticker_patterns:
        ticker_matches = re.findall(ticker_pattern, html_content, re.IGNORECASE)
        for ticker in ticker_matches:
            if ticker and len(ticker) >= 2 and ticker not in cover_data["trading_symbols"]:
                cover_data["trading_symbols"].append(ticker)
    
    # Also check clean text for FLWS specifically
    flws_match = re.search(r'\bFLWS\b', clean_text)
    if flws_match and "FLWS" not in cover_data["trading_symbols"]:
        cover_data["trading_symbols"].append("FLWS")
    
    # Extract outstanding shares with specific Class A/B patterns
    class_a_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*\([Nn]umber of shares of class A common stock outstanding', clean_text, re.IGNORECASE)
    if class_a_match:
        cover_data["outstanding_shares"]["Class A"] = class_a_match.group(1)
    
    class_b_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*\([Nn]umber of shares of class B common stock outstanding', clean_text, re.IGNORECASE)
    if class_b_match:
        cover_data["outstanding_shares"]["Class B"] = class_b_match.group(1)
    
    return cover_data


def extract_stockholders_equity_notes(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extracts equity information from Part II, Item 8 - Notes to Consolidated Financial Statements.
    Looks for Stockholders' Equity, Capital Stock, Share Capital notes.
    """
    equity_notes = {
        "note_sections": [],
        "authorized_shares": {},
        "issued_shares": {},
        "outstanding_shares": {},
        "par_values": {},
        "rights_preferences": {},
        "preferred_stock": {}
    }
    
    # Look for relevant note headings
    note_headings = [
        "stockholders'?\s+equity",
        "capital\s+stock",
        "share\s+capital", 
        "earnings\s+per\s+share",
        "shareowners'?\s+equity",
        "shareholders'?\s+equity"
    ]
    
    text = soup.get_text()
    
    for heading_pattern in note_headings:
        pattern = rf"(?:note\s+\d+\.?\s*[-–—]?\s*)?{heading_pattern}"
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            # Extract content following the heading (next 3000 characters)
            start_pos = match.start()
            section_text = text[start_pos:start_pos + 3000]
            
            equity_notes["note_sections"].append({
                "heading": match.group(),
                "content_preview": section_text[:500]
            })
            
            # Extract authorized shares
            auth_patterns = [
                r"authorized\s*:?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million\s+)?shares?\s+of\s+([^,\n]+)",
                r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million\s+)?shares?\s+authorized\s+([^,\n]+)"
            ]
            
            for auth_pattern in auth_patterns:
                auth_matches = re.findall(auth_pattern, section_text, re.IGNORECASE)
                for shares, class_type in auth_matches:
                    equity_notes["authorized_shares"][class_type.strip()] = shares.replace(",", "")
            
            # Extract par values
            par_patterns = [
                r"\$(\d+\.?\d*)\s+par\s+value",
                r"par\s+value\s+\$(\d+\.?\d*)",
                r"(\$\d+\.?\d*)\s+par"
            ]
            
            for par_pattern in par_patterns:
                par_matches = re.findall(par_pattern, section_text, re.IGNORECASE)
                for par_value in par_matches:
                    equity_notes["par_values"]["common_stock"] = par_value
            
            # Look for conversion terms
            conversion_patterns = [
                r"convert(?:ible)?\s+(?:into|to)\s+([^.]+)",
                r"conversion\s+ratio\s*:?\s*([^.]+)",
                r"each\s+share\s+(?:of\s+)?([^.]+)\s+(?:is\s+)?convertible"
            ]
            
            for conv_pattern in conversion_patterns:
                conv_matches = re.findall(conv_pattern, section_text, re.IGNORECASE)
                for conversion_info in conv_matches:
                    if "conversion_terms" not in equity_notes["rights_preferences"]:
                        equity_notes["rights_preferences"]["conversion_terms"] = []
                    equity_notes["rights_preferences"]["conversion_terms"].append(conversion_info.strip())
    
    return equity_notes


def extract_exhibit_4_description(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extracts information from Exhibit 4 - Description of Registrant's Securities.
    This is often the most comprehensive plain-English summary.
    """
    exhibit_4 = {
        "found": False,
        "securities_descriptions": [],
        "voting_rights": {},
        "conversion_rights": {},
        "dividend_rights": {},
        "liquidation_rights": {}
    }
    
    text = soup.get_text()
    
    # Look for Exhibit 4 section
    exhibit_patterns = [
        r"exhibit\s+4[.\s]*[-–—]?\s*description\s+of\s+(?:registrant'?s?\s+)?securities",
        r"description\s+of\s+(?:registrant'?s?\s+)?securities",
        r"exhibit\s+4\.1"
    ]
    
    for pattern in exhibit_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            exhibit_4["found"] = True
            start_pos = match.start()
            # Extract substantial content (next 5000 characters)
            section_text = text[start_pos:start_pos + 5000]
            
            exhibit_4["securities_descriptions"].append({
                "heading": match.group(),
                "content": section_text
            })
            
            # Extract voting rights
            voting_patterns = [
                r"entitled\s+to\s+(\d+)\s+vote[s]?\s+per\s+share",
                r"(\d+)\s+vote[s]?\s+per\s+share",
                r"voting\s+rights?\s*:?\s*([^.]+)",
                r"each\s+share\s+entitles?\s+(?:the\s+)?holder\s+to\s+([^.]+)"
            ]
            
            for voting_pattern in voting_patterns:
                voting_matches = re.findall(voting_pattern, section_text, re.IGNORECASE)
                for voting_info in voting_matches:
                    exhibit_4["voting_rights"]["common_stock"] = voting_info.strip()
            
            # Extract dividend rights
            dividend_patterns = [
                r"dividend\s+rights?\s*:?\s*([^.]+)",
                r"entitled\s+to\s+receive\s+dividends?\s+([^.]+)",
                r"participate\s+(?:equally\s+)?in\s+dividends?\s+([^.]+)"
            ]
            
            for div_pattern in dividend_patterns:
                div_matches = re.findall(div_pattern, section_text, re.IGNORECASE)
                for div_info in div_matches:
                    exhibit_4["dividend_rights"]["common_stock"] = div_info.strip()
    
    return exhibit_4


def extract_charter_bylaws_info(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extracts information from Exhibits 3.1/3.2 - Charter & Bylaws.
    These contain the definitive legal terms.
    """
    charter_info = {
        "found": False,
        "documents": [],
        "conversion_ratios": {},
        "class_powers": {},
        "preferred_designations": []
    }
    
    text = soup.get_text()
    
    # Look for charter and bylaws exhibits
    charter_patterns = [
        r"exhibit\s+3\.1\s*[-–—]?\s*(?:amended\s+and\s+restated\s+)?(?:certificate\s+of\s+)?incorporation",
        r"exhibit\s+3\.2\s*[-–—]?\s*(?:amended\s+and\s+restated\s+)?bylaws",
        r"certificate\s+of\s+designation",
        r"articles\s+of\s+incorporation"
    ]
    
    for pattern in charter_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            charter_info["found"] = True
            start_pos = match.start()
            section_text = text[start_pos:start_pos + 3000]
            
            charter_info["documents"].append({
                "document_type": match.group(),
                "content_preview": section_text[:500]
            })
            
            # Look for specific conversion ratios
            conversion_patterns = [
                r"conversion\s+ratio\s*:?\s*(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)",
                r"each\s+share\s+(?:of\s+)?([^.]+)\s+shall\s+be\s+convertible\s+into\s+(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s+(?:shares?\s+of\s+)?common\s+stock\s+for\s+each\s+(?:share\s+of\s+)?([^.]+)"
            ]
            
            for conv_pattern in conversion_patterns:
                conv_matches = re.findall(conv_pattern, section_text, re.IGNORECASE)
                for match_data in conv_matches:
                    if len(match_data) == 2:
                        charter_info["conversion_ratios"][f"ratio_{len(charter_info['conversion_ratios'])}"] = {
                            "from": match_data[1] if match_data[1].replace('.','').isdigit() else match_data[0],
                            "to": match_data[0] if match_data[1].replace('.','').isdigit() else match_data[1]
                        }
    
    return charter_info


def extract_market_equity_info(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extracts information from Item 5 - Market for Registrant's Common Equity.
    Provides trading markets and holder information.
    """
    market_info = {
        "found": False,
        "trading_markets": [],
        "holder_count": {},
        "recent_sales": []
    }
    
    text = soup.get_text()
    
    # Look for Item 5 section
    item5_patterns = [
        r"item\s+5\s*[-–—]?\s*market\s+for\s+(?:registrant'?s?\s+)?common\s+equity",
        r"market\s+for\s+(?:the\s+)?(?:registrant'?s?\s+)?common\s+stock"
    ]
    
    for pattern in item5_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            market_info["found"] = True
            start_pos = match.start()
            section_text = text[start_pos:start_pos + 2000]
            
            # Extract trading venues
            market_patterns = [
                r"(?:traded\s+on|listed\s+on|quoted\s+on)\s+(?:the\s+)?([^.]+)",
                r"nasdaq\s+(?:global\s+)?(?:select\s+)?market",
                r"new\s+york\s+stock\s+exchange",
                r"nyse"
            ]
            
            for market_pattern in market_patterns:
                market_matches = re.findall(market_pattern, section_text, re.IGNORECASE)
                market_info["trading_markets"].extend(market_matches)
            
            # Extract holder counts
            holder_patterns = [
                r"(?:approximately\s+)?(\d{1,3}(?:,\d{3})*)\s+holders?\s+of\s+record",
                r"(\d{1,3}(?:,\d{3})*)\s+(?:record\s+)?(?:share)?holders?"
            ]
            
            for holder_pattern in holder_patterns:
                holder_matches = re.findall(holder_pattern, section_text, re.IGNORECASE)
                for holder_count in holder_matches:
                    market_info["holder_count"]["record_holders"] = holder_count.replace(",", "")
    
    return market_info


def extract_equity_data(cik: str, filing_path: str) -> Dict[str, Any]:
    """
    Main extraction function that coordinates all extraction methods.
    """
    print(f"  🔍 Extracting equity data from 10-K filing...")
    
    try:
        with open(filing_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {"error": f"Failed to read filing: {e}"}
    
    soup = BeautifulSoup(content, 'html.parser')
    
    extraction_result = {
        "cik": str(cik),
        "filing_path": filing_path,
        "extracted_at": datetime.datetime.utcnow().isoformat() + "Z",
        "cover_page": extract_cover_page_data(soup),
        "stockholders_equity_notes": extract_stockholders_equity_notes(soup),
        "exhibit_4_securities": extract_exhibit_4_description(soup),
        "charter_bylaws": extract_charter_bylaws_info(soup),
        "market_equity": extract_market_equity_info(soup)
    }
    
    # Add summary statistics
    extraction_result["extraction_summary"] = {
        "cover_page_found": len(extraction_result["cover_page"]["note_sections"]) > 0,
        "equity_notes_found": len(extraction_result["stockholders_equity_notes"]["note_sections"]) > 0,
        "exhibit_4_found": extraction_result["exhibit_4_securities"]["found"],
        "charter_found": extraction_result["charter_bylaws"]["found"],
        "market_info_found": extraction_result["market_equity"]["found"]
    }
    
    return extraction_result


def analyze_cik_equity(cik: str, symbol: Optional[str] = None, exchange: Optional[str] = None) -> Dict:
    """
    Main function to extract equity class details from a company's 10-K filing.
    """
    print(f"🔍 Extracting equity class details for CIK: {cik}")
    
    # Find the 10-K filing
    filing_path = find_10k_filing(cik)
    
    if not filing_path:
        return {
            "error": f"No 10-K filing found for CIK {cik}. Run 1.5_Download10K.py first.",
            "cik": str(cik)
        }
    
    print(f"  📄 Found 10-K filing: {os.path.basename(filing_path)}")
    
    # Extract equity data
    equity_data = extract_equity_data(cik, filing_path)
    
    if "error" in equity_data:
        return equity_data
    
    # Add metadata
    equity_data["input_info"] = {
        "symbol": symbol,
        "exchange": exchange,
        "filing_size_kb": round(os.path.getsize(filing_path) / 1024, 1)
    }
    
    # Save results
    os.makedirs('staging', exist_ok=True)
    outname = f"staging/cik_{str(cik).zfill(10)}_equity_extraction.json"
    save_json_file(equity_data, outname)
    
    return equity_data


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 1.6_Extract10K.py --cik 0000123456 [--symbol TICK --exchange NASDAQ]")
        sys.exit(1)
    
    if sys.argv[1] != "--cik":
        print("First argument must be --cik")
        sys.exit(1)
    
    cik = sys.argv[2]
    symbol = None
    exchange = None
    
    # Parse optional symbol and exchange
    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--symbol" and i + 1 < len(sys.argv):
            symbol = sys.argv[i + 1]
        elif sys.argv[i] == "--exchange" and i + 1 < len(sys.argv):
            exchange = sys.argv[i + 1]
    
    result = analyze_cik_equity(cik, symbol, exchange)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("📋 Extraction Results:")
        
        summary = result.get("extraction_summary", {})
        print(f"  📊 Cover Page Data: {'✅' if summary.get('cover_page_found') else '❌'}")
        print(f"  📊 Equity Notes: {'✅' if summary.get('equity_notes_found') else '❌'}")
        print(f"  📊 Exhibit 4: {'✅' if summary.get('exhibit_4_found') else '❌'}")
        print(f"  📊 Charter/Bylaws: {'✅' if summary.get('charter_found') else '❌'}")
        print(f"  📊 Market Info: {'✅' if summary.get('market_info_found') else '❌'}")
        
        # Show some key extracted data
        if result.get("cover_page", {}).get("trading_symbols"):
            print(f"  🎫 Trading Symbols: {', '.join(result['cover_page']['trading_symbols'])}")
        
        if result.get("stockholders_equity_notes", {}).get("authorized_shares"):
            auth_shares = result["stockholders_equity_notes"]["authorized_shares"]
            print(f"  📈 Authorized Shares: {len(auth_shares)} class(es) found")
        
        print(f"\n💾 Detailed extraction saved to: staging/cik_{str(cik).zfill(10)}_equity_extraction.json")
