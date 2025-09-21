"""
10-K Equity Class Extraction (No AI)
Extracts standardized equity details from 10-K filings using deterministic parsing.

Outputs JSON to staging/cik_{cik}_equity_extraction.json
"""
from __future__ import annotations

import os
import re
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple
from bs4 import BeautifulSoup
from shared_utils import save_json_file

# --------------------------- File discovery ---------------------------

def find_10k_filing(cik: str) -> Optional[str]:
    cik_padded = str(cik).zfill(10)
    base = os.path.join("sec_filings", "10K")
    if not os.path.isdir(base):
        return None
    candidates: List[str] = []
    for name in os.listdir(base):
        if name.startswith(cik_padded) and (name.endswith(".txt") or name.endswith(".htm")):
            candidates.append(os.path.join(base, name))
    if not candidates:
        return None
    # Prefer .txt, newest mtime
    candidates.sort(key=lambda p: (0 if p.endswith(".txt") else 1, os.path.getmtime(p)), reverse=False)
    return candidates[0]


def read_filing_content(path: str) -> Tuple[str, Optional[str]]:
    """Returns plain text and the raw HTML (if available). If given a .txt, try to locate sibling .htm/.html."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    lower = path.lower()
    if lower.endswith((".htm", ".html")):
        html = raw
        soup = BeautifulSoup(raw, "html.parser")
        text = soup.get_text("\n")
        return text, html
    # If a .txt, try to find same-named .htm/.html sibling
    html_path_htm = path[:-4] + ".htm" if lower.endswith(".txt") else None
    html_path_html = path[:-4] + ".html" if lower.endswith(".txt") else None
    html = None
    if html_path_htm and os.path.exists(html_path_htm):
        try:
            with open(html_path_htm, "r", encoding="utf-8") as hf:
                html = hf.read()
        except Exception:
            html = None
    elif html_path_html and os.path.exists(html_path_html):
        try:
            with open(html_path_html, "r", encoding="utf-8") as hf:
                html = hf.read()
        except Exception:
            html = None
    return raw, html

# --------------------------- Helpers ---------------------------

def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --------------------------- Section extraction helpers (text chunks only) ---------------------------

def _extract_cover_text(full_text: str) -> str:
    """Return the visible cover page:
    - Cut everything before a strong cover anchor (e.g., UNITED STATES SECURITIES AND EXCHANGE COMMISSION, FORM 10-K, Commission File Number)
    - Stop at the 'TABLE OF CONTENTS' marker if present
    - Fallback to first 30k chars if no anchors found
    """
    # Determine end bound (TOC if present)
    end_m = re.search(r"\bTABLE\s+OF\s+CONTENTS\b", full_text, re.IGNORECASE)
    end_idx = end_m.start() if end_m else min(len(full_text), 30000)
    candidate = full_text[:end_idx]

    # Find a good start anchor within candidate
    anchors = [
        r"UNITED\s+STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION",
        r"SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION",
        r"FORM\s*10-?K\b",
        r"Commission\s+File\s+Number",
        r"Washington,\s*D\.C\.\s*20549",
    ]
    starts = []
    for rx in anchors:
        m = re.search(rx, candidate, re.IGNORECASE)
        if m:
            starts.append(m.start())
    if starts:
        start_idx = max(0, min(starts))  # start exactly at the first anchor
        return candidate[start_idx:].strip()

    # If no anchor found, try to skip leading ix/xbrl junk lines until first uppercase paragraph
    lines = candidate.splitlines()
    i = 0
    junk_rx = re.compile(r"^(https?://|[a-z]+:[A-Za-z0-9_]+|\d{4}-\d{2}-\d{2}|[0-9]{8,}|[A-Za-z]+:[A-Za-z]+Member|P\d+[MYDQ]|iso4217:|xbrli:|us-gaap:|dei:|srt:|utr:|country:)\b")
    while i < len(lines) and junk_rx.search(lines[i].strip()):
        i += 1
    cleaned = "\n".join(lines[i:]).strip()
    return cleaned if cleaned else candidate.strip()


def _slice_to_next(full_text: str, start_idx: int, stop_rxs: List[str]) -> str:
    """Slice full_text from start_idx up to the earliest stop regex. If none found, to end of text."""
    end = len(full_text)
    tail = full_text[start_idx:]
    for rx in stop_rxs:
        m = re.search(rx, tail, re.IGNORECASE)
        if m:
            idx = start_idx + m.start()
            if idx < end:
                end = idx
    return full_text[start_idx:end].strip()


def _extract_section(full_text: str, title_rx: str, heading: str, stop_rxs: List[str]) -> Optional[Dict[str, str]]:
    m = re.search(title_rx, full_text, re.IGNORECASE)
    if not m:
        return None
    start = m.start()
    content = _slice_to_next(full_text, start, stop_rxs)
    return {"heading": heading, "content": content}


# --------------------------- Main extraction (chunks only) ---------------------------

def extract_10k_data(cik: str) -> Dict[str, Any]:
    filing = find_10k_filing(cik)
    if not filing:
        return {"error": "10-K filing not found"}
    text, html = read_filing_content(filing)

    # Cover page as full pre-TOC content
    cover_text = _extract_cover_text(text)
    cover_sections: List[Dict[str, str]] = []
    if cover_text:
        cover_sections.append({"heading": "Cover Page", "content": cover_text.strip()})
        # Section 12(b) block within the cover
        m12b = re.search(r"Securities registered pursuant to Section\s*12\(b\)", cover_text, re.IGNORECASE)
        if m12b:
            section_12b = _slice_to_next(
                cover_text,
                m12b.start(),
                [
                    r"Securities registered pursuant to Section\s*12\(g\)",
                    r"Indicate by check mark",
                    r"\bTABLE\s+OF\s+CONTENTS\b",
                ],
            )
            if section_12b:
                cover_sections.append({
                    "heading": "Securities registered pursuant to Section 12(b)",
                    "content": section_12b,
                })

    cover_page = {"sections": cover_sections}

    # Notes (Item 8) related chunks — capture full section until next Item or another note title
    note_titles = [
        "stockholders' equity", "stockholders equity",
        "shareholders' equity", "shareholders equity",
        "shareowners' equity", "capital stock", "share capital",
        "earnings per share",
    ]
    note_sections: List[Dict[str, str]] = []
    # Build stop patterns: next Item or any of the other titles
    stop_rxs_base = [r"\n\s*Item\s+\d+\.?"]
    for t in note_titles:
        stop_rxs_base.append(re.escape(t))
    for t in note_titles:
        m = re.search(re.escape(t), text, re.IGNORECASE)
        if not m:
            continue
        start = m.start()
        # Find earliest stop AFTER this start among items and other note titles (not counting itself at same index)
        end = len(text)
        # Search on tail
        tail = text[start + 1:]
        min_idx = None
        for rx in stop_rxs_base:
            sm = re.search(rx, tail, re.IGNORECASE)
            if sm:
                idx = start + 1 + sm.start()
                if idx <= start:  # shouldn't happen due to +1
                    continue
                if min_idx is None or idx < min_idx:
                    min_idx = idx
        if min_idx is not None:
            content = text[start:min_idx].strip()
        else:
            content = text[start:].strip()
        if content:
            # Use a cleaner heading label capitalization
            heading_label = t.title()
            note_sections.append({"heading": heading_label, "content": content})

    stockholders_equity_notes = {"sections": note_sections}

    # Item 5 — full section until next Item 6/7/7A/8/Part III
    item5_section = _extract_section(
        text,
        r"Item\s*5\.?\s*Market\s+for\s+Registrant'?s?\s+Common\s+Equity[\s\S]{0,200}",
        "Item 5 - Market for Registrant's Common Equity",
        [r"\n\s*Item\s*6\b", r"\n\s*Item\s*7\b", r"\n\s*Item\s*7A\b", r"\n\s*Item\s*8\b", r"\bPART\s+III\b"],
    )
    market_equity = {"sections": ([item5_section] if item5_section else [])}

    # Exhibit 4 — Description of Registrant's Securities (until Exhibit 5 or next Item)
    exhibit4_section = None
    # Try a couple of anchors
    for rx in [
        r"Exhibit\s*4\.?[\s\S]{0,120}Description\s+of\s+Registrant'?s?\s+Securities",
        r"Description\s+of\s+Registrant'?s?\s+Securities",
    ]:
        exhibit4_section = _extract_section(
            text,
            rx,
            "Exhibit 4 - Description of Registrant's Securities",
            [r"\n\s*Exhibit\s*5\b", r"\n\s*Item\s+\d+\b", r"\bPART\s+III\b"],
        )
        if exhibit4_section:
            break
    exhibit_4_securities = {"sections": ([exhibit4_section] if exhibit4_section else [])}

    # Charter and Bylaws — capture Exhibits 3.x block until Exhibit 4 or next Item
    charter_section = None
    m3 = re.search(r"\n\s*Exhibit\s*3\.[12]\b", text, re.IGNORECASE)
    if m3:
        content = _slice_to_next(text, m3.start(), [r"\n\s*Exhibit\s*4\b", r"\n\s*Item\s+\d+\b", r"\bPART\s+III\b"])
        if content:
            charter_section = {"heading": "Exhibits 3.x - Charter and Bylaws", "content": content}
    else:
        # Fallback: look for common phrases and take a modest window around them
        hits: List[Dict[str, str]] = []
        for p in [r"Amended and Restated Certificate of Incorporation", r"Amended and Restated By[- ]?laws"]:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                start = max(0, m.start() - 200)
                end = min(len(text), m.end() + 1200)
                hits.append({"heading": "Charter/Bylaws Reference", "content": text[start:end].strip()})
        if hits:
            charter_bylaws = {"sections": hits}
        else:
            charter_bylaws = {"sections": []}
    charter_bylaws = {"sections": ([charter_section] if charter_section else charter_bylaws.get("sections", []))}

    return {
        "cik": str(cik).zfill(10),
        "filing_path": filing,
        "extracted_at": datetime.datetime.utcnow().isoformat() + "Z",
        "cover_page": cover_page,
        "stockholders_equity_notes": stockholders_equity_notes,
        "exhibit_4_securities": exhibit_4_securities,
        "charter_bylaws": charter_bylaws,
        "market_equity": market_equity,
    }

# --------------------------- CLI ---------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract equity class details from 10-K filings (no AI)")
    parser.add_argument("--cik", required=True, help="CIK to process")
    args = parser.parse_args()
    cik = args.cik.zfill(10)
    data = extract_10k_data(cik)
    os.makedirs("staging", exist_ok=True)
    out = os.path.join("staging", f"cik_{cik}_equity_extraction.json")
    if not save_json_file(data, out):
        # Fallback simple write
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    try:
        print(f"[SUCCESS] Saved extraction to {out}")
    except UnicodeEncodeError:
        print(f"Saved extraction to {out}")


if __name__ == "__main__":
    main()
