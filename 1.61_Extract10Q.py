#!/usr/bin/env python3
"""
1.61_Extract10Q.py - Extract key equity-related sections from a 10-Q filing into a normalized JSON

This script reads the latest downloaded 10-Q HTML (saved by 1.51_Download10Q.py),
parses the filing, and extracts sections relevant to equity classes:
- Cover page (Section 12(b) table area)
- Stockholders’/Shareholders’ Equity notes
- Capital Stock notes (if present)
- Market for Registrant’s Common Equity (if present; uncommon in 10-Q)
- In-body descriptions of securities/capital stock (non-exhibit, if present)

Output is saved to staging as: staging/cik_{CIK}_equity_extraction_10q.json

Usage:
  python 1.61_Extract10Q.py --cik 0000123456
  python 1.61_Extract10Q.py --file sec_filings/10Q/0000123456_0001234567-24-000001_company-20240630.htm

Notes:
- This is a 10-Q counterpart to the 10-K extractor. It aims to be robust but
  conservative and will return whatever sections it can locate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# --------------------------- Utilities ---------------------------

def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[WARN] Failed reading file: {e}")
        return None


def _html_to_visible_text(html: str) -> str:
    """Convert HTML to visible text, unwrapping ix: tags and removing hidden/script/style."""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        # Unwrap Inline XBRL tags (ix:*) to keep visible text
        for ix_tag in soup.find_all(re.compile(r"^ix:")):
            ix_tag.unwrap()
        text = soup.get_text("\n")
        # Normalize whitespace
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\u00a0", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()
    except Exception:
        # Fallback: naive strip of tags
        txt = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
        txt = re.sub(r"<style[\s\S]*?</style>", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()


# --------------------------- Section Extractors ---------------------------

def _slice_block(lines: List[str], start_idx: int, stop_patterns: List[re.Pattern]) -> str:
    out: List[str] = []
    for i in range(start_idx, len(lines)):
        ln = lines[i]
        if any(p.search(ln) for p in stop_patterns):
            break
        out.append(ln)
    return "\n".join(out).strip()


def _extract_cover_page_sections(full_text: str) -> List[Dict[str, str]]:
    lines = [ln.rstrip() for ln in full_text.splitlines()]
    sections: List[Dict[str, str]] = []

    rx_12b = re.compile(r"Securities\s+registered\s+pursuant\s+to\s+Section\s*12\(b\)", re.IGNORECASE)
    rx_12g = re.compile(r"Securities\s+registered\s+pursuant\s+to\s+Section\s*12\(g\)", re.IGNORECASE)
    stop = [rx_12g, re.compile(r"^PART\s+I|^PART\s+II|^ITEM\s+1", re.IGNORECASE)]

    for i, ln in enumerate(lines):
        if rx_12b.search(ln):
            block = _slice_block(lines, i, stop)
            if block:
                sections.append({
                    "heading": "Cover Page - Section 12(b)",
                    "content": block
                })
            break
    # If not found, try to capture the top page as a fallback (limited length)
    if not sections and lines:
        head = "\n".join(lines[:200]).strip()
        if head:
            sections.append({"heading": "Cover Page (fallback)", "content": head})
    return sections


def _extract_note_like_sections(full_text: str, title_patterns: List[str]) -> List[Dict[str, str]]:
    """Generic note extractor: finds headings and slices until next major boundary."""
    patterns = [re.compile(p, re.IGNORECASE) for p in title_patterns]
    lines = [ln.rstrip() for ln in full_text.splitlines()]
    sections: List[Dict[str, str]] = []

    # Stop boundaries: next NOTE heading, next ITEM/PART, all-caps headings (heuristic)
    stop_patterns = [
        re.compile(r"^note\s+\d+|^\d+\.?\s+note", re.IGNORECASE),
        re.compile(r"^item\s+\d+|^part\s+[ivx]+", re.IGNORECASE),
        re.compile(r"^[A-Z][A-Z\s\-&]{8,}$"),
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        if any(p.search(line) for p in patterns):
            # Capture from here
            block = _slice_block(lines, i, stop_patterns)
            if block:
                # The first line as heading; else use matched pattern name
                heading = line.strip()
                sections.append({"heading": heading, "content": block})
            # Advance a bit to avoid re-matching within block
            i += max(1, len(block.splitlines()))
        else:
            i += 1
    return sections


def _extract_capital_stock(full_text: str) -> List[Dict[str, str]]:
    pats = [
        r"\bcapital\s+stock\b",
        r"\bcapitalization\b",
        r"\bdescription\s+of\s+(?:capital\s+stock|securities)\b",
    ]
    return _extract_note_like_sections(full_text, pats)


def _extract_stockholders_equity(full_text: str) -> List[Dict[str, str]]:
    pats = [
        r"stockholders'?\s+equity",
        r"shareholders'?\s+equity",
        r"statements?\s+of\s+stockholders'?\s+equity",
        r"changes\s+in\s+stockholders'?\s+equity",
    ]
    return _extract_note_like_sections(full_text, pats)


def _extract_market_equity(full_text: str) -> List[Dict[str, str]]:
    pats = [
        r"market\s+for\s+registrant'?s\s+common\s+equity",
        r"market\s+information\s+for\s+common\s+stock",
    ]
    return _extract_note_like_sections(full_text, pats)


def _extract_in_body_descriptions(full_text: str) -> List[Dict[str, str]]:
    pats = [
        r"description\s+of\s+registrant'?s\s+securities",
        r"description\s+of\s+capital\s+stock",
    ]
    return _extract_note_like_sections(full_text, pats)


# --------------------------- Main extraction ---------------------------

def extract_10q_data(html_text: str, cik: str, filing_path: str) -> Dict[str, Any]:
    full_text = _html_to_visible_text(html_text)

    result: Dict[str, Any] = {
        "cik": str(cik),
        "source": "10-Q",
        "filing_path": filing_path,
        "cover_page": {"sections": _extract_cover_page_sections(full_text)},
        "stockholders_equity_notes": {"sections": _extract_stockholders_equity(full_text)},
        "market_equity": {"sections": _extract_market_equity(full_text)},
        "capital_stock": {"sections": _extract_capital_stock(full_text)},
        "in_body_security_descriptions": {"sections": _extract_in_body_descriptions(full_text)},
    }
    return result


def find_latest_10q_html(cik: str) -> Tuple[Optional[str], Optional[str]]:
    """Locate the downloaded 10-Q HTML using the staging download report created by 1.51.
    Returns (file_path, error)."""
    cik_padded = str(cik).zfill(10)
    report = f"staging/cik_{cik_padded}_10q_download.json"
    rep = _read_text_file(report)
    if not rep:
        return None, f"Download report not found: {report}. Run 1.51_Download10Q.py first."
    try:
        j = json.loads(rep)
    except Exception as e:
        return None, f"Invalid JSON in download report: {e}"

    filing = (j or {}).get("filing") or {}
    accession = filing.get("accessionNumber")
    primary = filing.get("primaryDocument")
    form = (filing.get("form") or "10-Q").replace('-', '').replace('/', '_')
    if not accession or not primary:
        return None, "Missing accessionNumber/primaryDocument in download report"

    storage_dir = f"sec_filings/{form}"
    os.makedirs(storage_dir, exist_ok=True)

    file_path = os.path.join(storage_dir, f"{cik_padded}_{accession}_{primary}")
    if os.path.exists(file_path):
        return file_path, None

    # Legacy name without dashes in accession
    acc_clean = accession.replace('-', '')
    legacy_file = os.path.join(storage_dir, f"{cik_padded}_{acc_clean}_{primary}")
    if os.path.exists(legacy_file):
        return legacy_file, None

    return None, f"Downloaded 10-Q HTML not found in {storage_dir}"


def main():
    parser = argparse.ArgumentParser(description="Extract key equity-related sections from a 10-Q filing")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cik", help="CIK number to process (reads staging/cik_{CIK}_10q_download.json to locate HTML)")
    group.add_argument("--file", help="Direct path to a 10-Q HTML file")
    parser.add_argument("--output", help="Output JSON file path (default: staging/cik_{CIK}_equity_extraction_10q.json)")

    args = parser.parse_args()

    if args.file:
        html_path = args.file
        cik_match = re.search(r"cik_(\d+)", html_path)
        cik = cik_match.group(1) if cik_match else (args.cik if args.cik else "unknown")
    else:
        cik = args.cik.zfill(10)
        html_path, err = find_latest_10q_html(cik)
        if err:
            print(f"[ERROR] {err}")
            sys.exit(1)

    # Determine output
    if args.output:
        out_path = args.output
    else:
        out_path = f"staging/cik_{cik}_equity_extraction_10q.json"

    # Read HTML
    html_content = _read_text_file(html_path)
    if not html_content:
        print(f"[ERROR] Could not read 10-Q HTML: {html_path}")
        sys.exit(1)

    print("[INFO] Extracting equity-related sections from 10-Q...")
    print(f"  CIK: {cik}")
    print(f"  HTML: {html_path}")
    print(f"  Output: {out_path}")

    data = extract_10q_data(html_content, cik, html_path)

    # Save JSON
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] Saved extraction JSON: {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save output: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
