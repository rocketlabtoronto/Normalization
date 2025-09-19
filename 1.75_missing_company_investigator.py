"""
Simplified investigation tool for missing companies and SEC filing analysis.
Uses shared utilities to eliminate code duplication.

Usage:
python 1.75_missing_company_investigator.py --cik 0000123456 [--symbol TICK --exchange NASDAQ]
python 1.75_missing_company_investigator.py [--ai-check]

Output: writes "staging/cik_{cik}_events.json" in the staging directory.
"""
from __future__ import annotations

import json
import time
import re
import datetime
import os
import urllib.parse
from typing import List, Dict, Optional

from shared_utils import (
    make_request, 
    fetch_sec_submissions,
    fetch_sec_ticker_map,
    load_json_file,
    save_json_file,
    generate_ticker_variants,
    setup_openai,
    query_openai
)


def extract_recent_filings(submissions: Dict, months: int = 18) -> List[Dict]:
    """
    Filters a company's filing history to find important events in the last X months.
    Looks specifically for 8-K forms (major events), Form 25 (delistings), and Form 15 (going private).
    These forms typically contain announcements about mergers, acquisitions, or companies going out of business.
    """
    """Return a list of recent filings (dicts with form, accessionNumber, filingDate, primaryDocument).
    The SEC submissions JSON contains filings -> recent arrays.
    """
    recent = []
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=months * 30)).date()
    filings = submissions.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    acc_nums = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])
    primary_docs = filings.get("primaryDocument", [])
    for form, acc, fdate, pdoc in zip(forms, acc_nums, dates, primary_docs):
        try:
            d = datetime.datetime.strptime(fdate, "%Y-%m-%d").date()
        except Exception:
            continue
        if d >= cutoff:
            recent.append({"form": form, "accessionNumber": acc, "filingDate": fdate, "primaryDocument": pdoc})
    # also include filings from older sections if present (not necessary for minimal)
    return recent


def build_filing_archives_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Construct a likely Archives URL for a filing document."""
    # accession_number may look like '0000123456-21-000123' -> remove dashes
    acc_clean = accession_number.replace('-', '')
    cik_int = str(int(str(cik)))  # remove leading zeros
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{primary_document}"


def fetch_filing_text(cik: str, accession_number: str, primary_document: str, form_type: str = "unknown") -> Optional[str]:
    """
    Downloads the actual text content of a specific SEC filing document and saves it to organized folders by form type.
    Creates permanent file storage that can be reused across pipeline steps (Step 1.75 → Step 2).
    
    Args:
        cik: Company CIK number
        accession_number: SEC accession number 
        primary_document: Primary document filename
        form_type: Type of form (8-K, 10-K, 10-Q, etc.) for folder organization
    """
    # Normalize CIK for consistent naming
    cik_padded = str(cik).zfill(10)
    acc_clean = accession_number.replace('-', '')
    
    # Create organized folder structure by form type
    form_folder = form_type.replace('-', '').replace('/', '_')  # Clean folder name (8K, 10K, 10Q, etc.)
    storage_dir = f"sec_filings/{form_folder}"
    os.makedirs(storage_dir, exist_ok=True)
    
    # Filename: CIK_accession_document (e.g., 0000123456_000012345621000123_document.htm)
    filename = f"{cik_padded}_{acc_clean}_{primary_document}"
    file_path = os.path.join(storage_dir, filename)
    
    # Check if file already exists in permanent storage
    if os.path.exists(file_path):
        print(f"  � Using stored {form_type}: {filename}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"  ⚠️ File read failed: {e}, re-downloading...")
    
    # Download from SEC Archives
    cik_int = str(int(str(cik)))  # remove leading zeros for URL
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{primary_document}"
    
    print(f"  📡 Downloading {form_type}: {filename}")
    response = make_request(url)
    content = None
    
    if response and response.text:
        content = response.text
        time.sleep(0.2)
    else:
        # fallback: IX viewer
        ix = f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{cik_int}/{acc_clean}/{primary_document}"
        response = make_request(ix)
        if response and response.text:
            content = response.text
            time.sleep(0.2)
    
    # Save the downloaded filing to permanent storage
    if content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  💾 Saved {form_type}: {filename}")
        except Exception as e:
            print(f"  ⚠️ File save failed: {e}")
    
    return content


def save_8k_items(cik: str, accession_number: str, items: Dict[str, str], filing_date: str, form_type: str) -> None:
    """
    Saves extracted 8-K items to JSON format in organized folders by company.
    Creates permanent storage for 8-K event extractions that can be referenced later.
    
    Args:
        cik: Company CIK number
        accession_number: SEC accession number
        items: Extracted 8-K items dictionary
        filing_date: Date of the filing
        form_type: Form type (8-K, etc.)
    """
    if not items:  # Don't save empty items
        return
        
    # Normalize CIK for consistent naming
    cik_padded = str(cik).zfill(10)
    acc_clean = accession_number.replace('-', '')
    
    # Create folder structure: 8k_items/CIK0000123456/
    items_dir = f"sec_filings/8k_items/CIK{cik_padded}"
    os.makedirs(items_dir, exist_ok=True)
    
    # JSON filename: CIK_accession_items.json
    json_filename = f"{cik_padded}_{acc_clean}_items.json"
    json_path = os.path.join(items_dir, json_filename)
    
    # Create structured data
    items_data = {
        "cik": cik,
        "accession_number": accession_number,
        "filing_date": filing_date,
        "form_type": form_type,
        "extracted_items": items,
        "extraction_date": datetime.now().isoformat()
    }
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(items_data, f, indent=2, ensure_ascii=False)
        print(f"  📋 Saved 8-K items: 8k_items/CIK{cik_padded}/{json_filename}")
    except Exception as e:
        print(f"  ⚠️ Failed to save 8-K items: {e}")


def parse_8k_items(text: str) -> Dict[str, str]:
    """
    Extracts specific event types from 8-K filing text by looking for standard SEC item numbers.
    Item 1.03 = bankruptcy, Item 2.01 = acquisitions/mergers, Item 3.01 = material agreements.
    These item codes tell you what kind of major corporate event happened to the company.
    """
    """Search for Items 1.03, 2.01, 3.01 in 8-K text and return short excerpts if found."""
    items = {}
    if not text:
        return items
    # Normalize whitespace
    t = re.sub(r"\s+", " ", text)
    # Look for common patterns for Item headings
    for item in ("1.03", "2.01", "3.01"):
        pat = re.compile(rf"(Item\s+{re.escape(item)}\b.*?)(?=Item\s+\d+\.\d+|$)", re.I)
        m = pat.search(t)
        if m:
            excerpt = m.group(1).strip()
            # keep first 600 chars
            items[item] = excerpt[:600]
    return items


def analyze_cik(cik: str, months: int = 18, symbol: Optional[str] = None, exchange: Optional[str] = None) -> Dict:
    """
    Investigates what happened to a company by analyzing their recent SEC filings.
    Downloads their filing history, finds important events (mergers, bankruptcies, delistings),
    and creates a timeline of what occurred. This helps explain why a company might not
    have a CIK or why their stock stopped trading.
    """
    report: Dict = {
        "cik": str(cik), 
        "requested_at": datetime.datetime.utcnow().isoformat() + "Z", 
        "filings": [], 
        "narrative": [], 
        "exchange_checks": {}
    }
    
    try:
        subs = fetch_sec_submissions(cik)
    except Exception as e:
        report["error"] = f"Could not fetch submissions JSON: {e}"
        return report

    if not subs:
        report["error"] = "No submissions data returned"
        return report

    company_name = subs.get("name") or subs.get("companyName") or None
    if company_name:
        report["company_name"] = company_name

    recent = extract_recent_filings(subs, months=months)
    # filter forms of interest
    forms_of_interest = set(["8-K", "8-K/A", "25", "25-NSE", "15", "15-12G", "15-12B", "8-K/A"])
    relevant = [f for f in recent if f.get("form") in forms_of_interest or f.get("form", "").upper().startswith("8-")]

    # fetch and parse relevant filings
    for f in sorted(relevant, key=lambda x: x.get("filingDate")):
        form_type = f.get("form", "unknown")
        entry = {
            "form": form_type, 
            "filingDate": f.get("filingDate"), 
            "accessionNumber": f.get("accessionNumber"), 
            "primaryDocument": f.get("primaryDocument"), 
            "link": build_filing_archives_url(cik, f.get("accessionNumber"), f.get("primaryDocument"))
        }
        txt = fetch_filing_text(cik, f.get("accessionNumber"), f.get("primaryDocument"), form_type)
        if form_type.upper().startswith("8-"):
            items = parse_8k_items(txt or "")
            entry["8k_items"] = items
            # Save 8-K items to separate JSON files organized by CIK
            save_8k_items(cik, f.get("accessionNumber"), items, f.get("filingDate"), form_type)
        # detect Form 15 presence
        if form_type.startswith("15") or form_type.upper() == "FORM 15":
            entry["deregistration"] = True
        report["filings"].append(entry)

    # Build a simple narrative ordered by filing date
    for e in sorted(report["filings"], key=lambda x: x.get("filingDate")):
        date = e.get("filingDate")
        form = e.get("form")
        summary = f"{date}: {form}"
        if e.get("8k_items"):
            keys = ",".join(sorted(e.get("8k_items").keys()))
            summary += f" (8-K items: {keys})"
        if e.get("deregistration"):
            summary += " (Form 15 / deregistration present)"
        report["narrative"].append(summary)

    # Exchange-specific minimal checks (best-effort, require symbol)
    if symbol and exchange:
        exchange = exchange.upper()
        if exchange == "NASDAQ":
            report["exchange_checks"]["nasdaq"] = discover_nasdaq_alerts(symbol, months)
        elif exchange == "NYSE":
            report["exchange_checks"]["nyse"] = discover_nyse_notices(symbol, months)
        else:
            report["exchange_checks"]["note"] = "Unsupported exchange for automated checks"
    else:
        report["exchange_checks"]["note"] = "symbol or exchange not provided; skipping exchange checks"

    # OTC fallback: placeholder
    report["otc_check"] = "To check FINRA Daily List, provide approximate event dates; this script can be extended to query FINRA endpoints."

    # deregistration summary
    report["deregistration_flag"] = any(e.get("deregistration") for e in report["filings"])

    # save
    os.makedirs('staging', exist_ok=True)
    outname = f"staging/cik_{str(cik).zfill(10)}_events.json"
    save_json_file(report, outname)

    return report


# --- Exchange check placeholder implementations ---

def discover_nasdaq_alerts(symbol: str, months: int = 18) -> Dict:
    """Best-effort placeholder: attempt to find Nasdaq Trader corporate actions/alerts for the symbol.
    NasdaTrader does not provide a stable public JSON for this; this function returns a note and attempted URL(s).
    """
    # Example target URLs (not guaranteed):
    urls = [f"https://www.nasdaqtrader.com/Search?q={symbol}", f"https://www.nasdaqtrader.com/TraderNews.aspx?symbol={symbol}"]
    return {"note": "manual_check_recommended", "attempted_urls": urls}


def discover_nyse_notices(symbol: str, months: int = 18) -> Dict:
    """Best-effort placeholder: NYSE market notices and delisting pages vary. Return URLs to check manually."""
    urls = [f"https://www.nyse.com/quote/{symbol}", f"https://www.nyse.com/press-releases?symbol={symbol}"]
    return {"note": "manual_check_recommended", "attempted_urls": urls}


# --- OpenAI investigation helpers ---

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

Please provide a brief, factual 1-2 sentence summary focusing on corporate actions that would explain why it might not appear in current SEC filings or have missing CIK data. If the company is still active and trading normally, just say "Company appears to be active and trading normally."

Answer:"""

    return query_openai(prompt, max_tokens=150)


def investigate_company_with_ai(symbol: str, company_name: str = None) -> str:
    """Run OpenAI investigation and return result with progress feedback."""
    try:
        print(f"  → Querying OpenAI about {company_name or symbol}...", flush=True)
        result = openai_investigate_company(symbol, company_name)
        print(f"  → AI result: {result[:100]}{'...' if len(result) > 100 else ''}", flush=True)
        return result
    except Exception as e:
        print(f"  → AI query failed for {symbol}: {e}", flush=True)
        return f"AI investigation failed: {e}"


def process_no_cik_file(input_path: str = "staging/1.75_dual_class_output_nocik.json",
                        output_path: Optional[str] = None,
                        cache_path: str = "staging/sec_company_tickers_cache.json",
                        force_download: bool = False,
                        months: int = 18,
                        run_filings_on_found: bool = True,
                        ai_check: bool = True,
                        ai_delay_seconds: float = 0.5) -> Dict:
    """
    Processes a file containing companies that couldn't be matched to SEC CIK numbers.
    For each missing company, it uses multiple investigation methods (SEC filings, AI research,
    Google News) to determine what happened - whether they were acquired, went bankrupt,
    changed names, or were delisted. Creates detailed investigation reports for each company.
    """
    if output_path is None:
        output_path = input_path

    print(f"Loading input file: {input_path}")
    data = load_json_file(input_path)
    if not data:
        return {"error": "Could not load input file"}

    print(f"Loading SEC company ticker mappings (cache: {cache_path})...")
    # Use shared utility instead of duplicated function
    ticker_map = fetch_sec_ticker_map(cache_path)
    print(f"Loaded {len(ticker_map)} ticker mappings")
    
    if ai_check:
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️  AI investigation requested but OPENAI_API_KEY not set in environment")
            ai_check = False
        else:
            print("ℹ️  AI investigation enabled - will query OpenAI about companies not found in SEC cache")
    else:
        print("ℹ️  AI investigation disabled (use --ai-check to enable)")

    companies = data.get("companies", [])
    total_companies = len(companies)
    print(f"Processing {total_companies} companies for CIK resolution and Google News checks...")
    
    for i, comp in enumerate(companies, 1):
        try:
            comp_name = (comp.get("company_name") or "").strip()
            primary_ticker = (comp.get("primary_ticker") or "").strip()
            
            print(f"\n[{i}/{total_companies}] Processing: {comp_name} ({primary_ticker or 'no ticker'})")

            # default AI_REASON empty
            if ai_check:
                comp["AI_REASON"] = None

            # skip if CIK already present
            if comp.get("cik"):
                comp["reason"] = "already_has_cik"
                comp["reason_detail"] = {"cik": comp.get("cik")}
                print(f"  → Already has CIK: {comp.get('cik')}")
                # still optionally run AI check for context
                if ai_check and primary_ticker:
                    try:
                        comp["AI_REASON"] = investigate_company_with_ai(primary_ticker, comp_name)
                        time.sleep(ai_delay_seconds)
                    except Exception as e:
                        comp["AI_REASON"] = f"ai_check_error: {e}"
                continue

            # 1) header detection
            if comp_name.lower() in ("company name", "company") or (not primary_ticker and comp.get("classes") and len(comp.get("classes"))==1 and comp.get("classes")[0].get("class_name","").lower() in ("unequal","unequal voting","n/a")):
                comp["reason"] = "header_row_or_placeholder"
                comp["reason_detail"] = {"note": "appears to be header/placeholder row"}
                print(f"  → Detected as header/placeholder row")
                if ai_check and primary_ticker:
                    try:
                        comp["AI_REASON"] = investigate_company_with_ai(primary_ticker, comp_name)
                        time.sleep(ai_delay_seconds)
                    except Exception as e:
                        comp["AI_REASON"] = f"ai_check_error: {e}"
                continue

            # 2) try ticker variants
            found = None
            if primary_ticker:
                print(f"  → Searching for CIK by ticker variants of {primary_ticker}...")
                for v in generate_ticker_variants(primary_ticker):
                    cik = ticker_map.get(v.upper())
                    if cik:
                        found = (v.upper(), cik)
                        break
                        
            if found:
                key, cik = found
                comp["cik"] = cik
                comp["reason"] = "found_by_ticker_variant"
                comp["reason_detail"] = {"matched_variant": key, "cik": cik}
                print(f"  → Found CIK {cik} by ticker variant {key}")
                
                # optionally fetch recent filings
                if run_filings_on_found:
                    try:
                        print(f"  → Fetching recent filings for CIK {cik}...")
                        report = analyze_cik(cik, months=months, symbol=primary_ticker, exchange=None)
                        comp["filings_summary"] = report.get("narrative", [])[:6]
                        comp["deregistration_flag"] = report.get("deregistration_flag", False)
                        print(f"  → Found {len(comp['filings_summary'])} recent filings")
                    except Exception as e:
                        comp["filings_summary"] = []
                        comp["filings_error"] = str(e)
                        print(f"  → Error fetching filings: {e}")
                        
                # AI check
                if ai_check and primary_ticker:
                    try:
                        comp["AI_REASON"] = investigate_company_with_ai(primary_ticker, comp_name)
                        time.sleep(ai_delay_seconds)
                    except Exception as e:
                        comp["AI_REASON"] = f"ai_check_error: {e}"
                continue

            # 3) unable to find in SEC cache - mark for manual/exchange checks
            comp["reason"] = "not_in_sec_cache"
            comp["reason_detail"] = {"note": "Ticker and name not found in SEC master file; consider exchange delisting or non-SEC registrant"}
            print(f"  → Not found in SEC cache - checking AI investigation")
            if ai_check and primary_ticker:
                try:
                    comp["AI_REASON"] = investigate_company_with_ai(primary_ticker, comp_name)
                    time.sleep(ai_delay_seconds)
                except Exception as e:
                    comp["AI_REASON"] = f"ai_check_error: {e}"
                    
        except Exception as e:
            comp["reason"] = "analysis_exception"
            comp["reason_detail"] = {"error": str(e)}
            print(f"  → Error processing company: {e}")
            continue

    # write back
    data["companies"] = companies
    data["analyzed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    print(f"\nWriting results back to {output_path}...")
    save_json_file(data, output_path)

    # build a short summary
    summary = {}
    for c in companies:
        r = c.get("reason") or "unknown"
        summary[r] = summary.get(r, 0) + 1
    
    print(f"\nProcessing complete! Summary:")
    for reason, count in summary.items():
        print(f"  {reason}: {count}")
    
    return {"total": len(companies), "summary": summary}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--cik":
        # Single CIK analysis mode
        if len(sys.argv) < 3:
            print("Usage: python 1.75_missing_company_investigator.py --cik CIK_NUMBER [--symbol SYMBOL --exchange EXCHANGE]")
            sys.exit(1)
        cik = sys.argv[2]
        symbol = None
        exchange = None
        if "--symbol" in sys.argv:
            symbol_idx = sys.argv.index("--symbol")
            if symbol_idx + 1 < len(sys.argv):
                symbol = sys.argv[symbol_idx + 1]
        if "--exchange" in sys.argv:
            exchange_idx = sys.argv.index("--exchange")
            if exchange_idx + 1 < len(sys.argv):
                exchange = sys.argv[exchange_idx + 1]
        
        result = analyze_cik(cik, symbol=symbol, exchange=exchange)
        print(json.dumps(result, indent=2))
    else:
        # Batch processing mode (default)
        ai_check = "--ai-check" in sys.argv or "--openai" in sys.argv
        result = process_no_cik_file(ai_check=ai_check)
        print(json.dumps(result, indent=2))
