"""
SEC Filing Download Tool
Downloads and saves the latest 10-K filings for individual companies by CIK.

This script focuses on:
- Downloading and saving the latest 10-K filing to organized folders
- Creating filing storage system for reuse across pipeline steps (Step 1.5 → Step 2)
- Building permanent file structure with proper naming conventions

Usage:
python 1.5_ExtractionSEC.py --cik 0000123456 [--symbol TICK --exchange NASDAQ]

Output: 
- 10-K filings saved to sec_filings/10K/ folder
- Investigation report saved to staging/cik_{cik}_10k_download.json
"""
from __future__ import annotations

import json
import time
import datetime
import os
import sys
from typing import Dict, List, Optional, Union, Any
from shared_utils import (
    make_request, 
    fetch_sec_submissions, 
    save_json_file
)


def extract_latest_10k(submissions: Dict) -> Optional[Dict]:
    """
    Finds the most recent 10-K filing from SEC submissions.
    Looks for either 10-K or 10-K/A (amended) forms and returns the latest one.
    """
    if not submissions or "filings" not in submissions:
        return None
    
    filings = submissions["filings"]["recent"]
    forms = filings.get("form", [])
    acc_nums = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])
    primary_docs = filings.get("primaryDocument", [])
    
    latest_10k = None
    latest_date = None
    
    for form, acc, fdate, pdoc in zip(forms, acc_nums, dates, primary_docs):
        if form in ["10-K", "10-K/A"] and fdate:
            try:
                filing_date = datetime.datetime.strptime(fdate, "%Y-%m-%d")
                if latest_date is None or filing_date > latest_date:
                    latest_date = filing_date
                    latest_10k = {
                        "form": form,
                        "accessionNumber": acc,
                        "filingDate": fdate,
                        "primaryDocument": pdoc
                    }
            except ValueError:
                continue
    
    return latest_10k


def build_filing_archives_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Constructs the SEC Archives URL for downloading a specific filing document."""
    cik_int = str(int(str(cik)))  # remove leading zeros
    acc_clean = accession_number.replace('-', '')
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{primary_document}"


def fetch_filing_text(cik: str, accession_number: str, primary_document: str, form_type: str = "unknown") -> Optional[str]:
    """
    Downloads the actual text content of a specific SEC filing document and saves it to organized folders by form type.
    Creates permanent file storage that can be reused across pipeline steps (Step 1.5 → Step 2).
    
    Args:
        cik: Company CIK number
        accession_number: SEC accession number (kept in original format with dashes)
        primary_document: Primary document filename
        form_type: Type of form (10-K, 10-K/A, etc.) for folder organization
    """
    # Normalize CIK for consistent naming
    cik_padded = str(cik).zfill(10)
    acc_clean = accession_number.replace('-', '')  # Only for URL construction
    
    # Create organized folder structure by form type
    form_folder = form_type.replace('-', '').replace('/', '_')  # Clean folder name (10K, 10K_A, etc.)
    storage_dir = f"sec_filings/{form_folder}"
    os.makedirs(storage_dir, exist_ok=True)
    
    # Create README file explaining filename structure (permanent, not deletable)
    readme_path = os.path.join(storage_dir, "README_FILENAME_STRUCTURE.md")
    if not os.path.exists(readme_path):
        readme_content = f"""# SEC Filing Filename Structure - {form_folder} Folder

## Filename Format:
```
{{CIK_padded}}_{{accession_number}}_{{primary_document}}
```

## Component Breakdown:

### 1. **CIK_padded** (10 digits)
- **Format**: `0000123456`
- **Purpose**: Company's Central Index Key, zero-padded to 10 digits
- **Example**: `0001084869` = 1-800-FLOWERS.COM, INC.

### 2. **accession_number** (original format with dashes)
- **Format**: `0001234567-YY-NNNNNN`
- **Components**:
  - `0001234567` = Filing agent/law firm CIK
  - `YY` = Year (e.g., 24 = 2024)
  - `NNNNNN` = Sequential filing number
- **Example**: `0001437749-24-014253`

### 3. **primary_document** (original SEC document name)
- **Format**: Usually `ticker + date + form.htm`
- **Example**: `flws-20240430.htm`
  - `flws` = Ticker symbol
  - `20240430` = Filing date (YYYYMMDD)
  - `.htm` = Document format

## Example Complete Filename:
```
0001084869_0001437749-24-014253_flws-20240430.htm
```

**Translation:**
- Company: 1-800-FLOWERS (CIK: 0001084869)
- Filed by: Agent 0001437749 in 2024 (filing #014253)
- Document: FLWS 10-K form filed on April 30, 2024

## Original SEC URL Reconstruction:
```
https://www.sec.gov/Archives/edgar/data/{{company_cik}}/{{accession_clean}}/{{primary_document}}
```

**Example:**
```
https://www.sec.gov/Archives/edgar/data/1084869/000143774924014253/flws-20240430.htm
```

## Purpose:
- **Unique identification**: No filename conflicts
- **Chronological sorting**: Accession numbers sort by date
- **Company grouping**: CIK prefix groups by company
- **URL reconstruction**: Easy to rebuild original SEC links
- **Cross-pipeline reuse**: Step 2 can find and reuse these files

⚠️ **DO NOT DELETE THIS README** - It provides essential documentation for understanding the filing organization system.
"""
        try:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            print(f"  [INFO] Created README: {form_folder}/README_FILENAME_STRUCTURE.md")
        except Exception as e:
            print(f"  [WARN] Failed to create README: {e}")
    
    # Filename: CIK_accession_document (keeping original accession format with dashes)
    filename = f"{cik_padded}_{accession_number}_{primary_document}"
    file_path = os.path.join(storage_dir, filename)
    
    # Check if file already exists in permanent storage (new format with dashes)
    if os.path.exists(file_path):
        print(f"  [INFO] Using stored {form_type}: {filename}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"  [WARN] File read failed: {e}, re-downloading...")
    
    # Also check for legacy format (without dashes) to avoid duplicate downloads
    acc_clean = accession_number.replace('-', '')
    legacy_filename = f"{cik_padded}_{acc_clean}_{primary_document}"
    legacy_file_path = os.path.join(storage_dir, legacy_filename)
    
    if os.path.exists(legacy_file_path):
        print(f"  [INFO] Using stored {form_type} (legacy format): {legacy_filename}")
        try:
            with open(legacy_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"  [WARN] Legacy file read failed: {e}, re-downloading...")
    
    # Download from SEC Archives (URL still needs cleaned accession number)
    cik_int = str(int(str(cik)))  # remove leading zeros for URL
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{primary_document}"
    
    print(f"  [INFO] Downloading {form_type}: {filename}")
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
            print(f"  [SUCCESS] Saved {form_type}: {filename}")
        except Exception as e:
            print(f"  [WARN] File save failed: {e}")
    
    return content


def analyze_cik(cik: str, symbol: Optional[str] = None, exchange: Optional[str] = None) -> Dict:
    """
    Downloads the latest 10-K filing for a company by CIK.
    Creates permanent file storage for reuse across pipeline steps.
    """
    try:
        print(f"[INFO] Downloading latest 10-K filing for CIK: {cik}")
    except UnicodeEncodeError:
        print(f"Downloading latest 10-K filing for CIK: {cik}")
    
    report: Dict = {
        "cik": str(cik), 
        "requested_at": datetime.datetime.utcnow().isoformat() + "Z", 
        "filing": None, 
        "company_name": None
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

    # Find the latest 10-K filing
    latest_10k = extract_latest_10k(subs)
    
    if not latest_10k:
        report["error"] = "No 10-K filings found"
        return report

    # Download the 10-K filing
    form_type = latest_10k.get("form", "10-K")
    accession = latest_10k.get("accessionNumber")
    primary_doc = latest_10k.get("primaryDocument")
    
    entry = {
        "form": form_type, 
        "filingDate": latest_10k.get("filingDate"), 
        "accessionNumber": accession, 
        "primaryDocument": primary_doc, 
        "link": build_filing_archives_url(cik, accession, primary_doc)
    }
    
    # Download and save the filing
    txt = fetch_filing_text(cik, accession, primary_doc, form_type)
    if txt:
        entry["downloaded"] = True
        entry["size_kb"] = round(len(txt) / 1024, 1)
    else:
        entry["downloaded"] = False
        entry["error"] = "Failed to download filing content"
    
    report["filing"] = entry
    
    # Save report
    os.makedirs('staging', exist_ok=True)
    outname = f"staging/cik_{str(cik).zfill(10)}_10k_download.json"
    save_json_file(report, outname)

    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 1.5_ExtractionSEC.py --cik 0000123456 [--symbol TICK --exchange NASDAQ]")
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
    
    result = analyze_cik(cik, symbol, exchange)
    
    if "error" in result:
        print(f"[ERROR] Error: {result['error']}")
    else:
        print("[INFO] Download Results:")
        if result.get("filing"):
            filing = result["filing"]
            print(f"  Form: {filing.get('form')}")
            print(f"  Date: {filing.get('filingDate')}")
            print(f"  Link: {filing.get('link')}")
            if filing.get('downloaded'):
                print(f"  [SUCCESS] Downloaded: {filing.get('size_kb')} KB")
            else:
                print(f"  [ERROR] Download failed: {filing.get('error', 'Unknown error')}")
        
        if result.get("company_name"):
            print(f"  Company: {result['company_name']}")
    
    print(f"\n[INFO] Report saved to: staging/cik_{str(cik).zfill(10)}_10k_download.json")
