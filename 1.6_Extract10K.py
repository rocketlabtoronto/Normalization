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
from datetime import UTC
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


def _slice_to_next(full_text: str, start_idx: int, stop_rxs: List[str]) -> str:
    """Slice full_text from start_idx up to the earliest stop regex found AFTER the heading line.
    This avoids stopping immediately on the same line as the start heading.
    """
    end = len(full_text)
    # Start searching for stop markers from the next line (or next character if no newline)
    nl = full_text.find("\n", start_idx)
    search_from = (nl + 1) if nl != -1 else (start_idx + 1)
    tail = full_text[search_from:]
    for rx in stop_rxs:
        m = re.search(rx, tail, re.IGNORECASE)
        if m:
            idx = search_from + m.start()
            if idx < end:
                end = idx
    return full_text[start_idx:end].strip()

def _slice_to_next_multiline(full_text: str, start_idx: int, stop_rxs: List[str], max_window: int = 60000) -> str:
    """Slice from start_idx to the first occurrence of any stop regex that appears AFTER start_idx.
    Uses MULTILINE so ^ matches start of line anywhere. Falls back to max_window if no stop found.
    """
    end_candidates: List[int] = []
    for rx in stop_rxs:
        try:
            cre = re.compile(rx, re.IGNORECASE | re.MULTILINE)
        except re.error:
            # If rx already includes inline flags (?im), fallback to plain IGNORECASE
            cre = re.compile(rx, re.IGNORECASE)
        for m in cre.finditer(full_text):
            pos = m.start()
            if pos > start_idx:
                end_candidates.append(pos)
                break
    if end_candidates:
        end = min(end_candidates)
    else:
        end = min(len(full_text), start_idx + max_window)
    return full_text[start_idx:end].strip()


def _clean_ixbrl_lines(candidate: str) -> str:
    """Remove common iXBRL/namespace and machine tokens to make text human-readable."""
    lines = candidate.splitlines()
    http_rx = re.compile(r"https?://", re.IGNORECASE)
    colon_ns_rx = re.compile(r"\b[a-z][a-z0-9\-]*:[A-Za-z0-9_]+", re.IGNORECASE)
    numeric_only_rx = re.compile(r"^\d{8,}$")
    iso_date_rx = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    period_rx = re.compile(r"^P\d+[MYDQ]$", re.IGNORECASE)

    filtered: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            filtered.append("")
            continue
        if http_rx.search(s):
            continue
        if colon_ns_rx.search(s):
            continue
        if numeric_only_rx.match(s):
            continue
        if iso_date_rx.match(s):
            continue
        if period_rx.match(s):
            continue
        filtered.append(ln)

    cleaned = "\n".join(filtered)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned if cleaned else candidate.strip()

# --------------------------- Section extraction helpers (text chunks only) ---------------------------

def _extract_cover_text(full_text: str) -> str:
    """Return the visible cover page without IX/XBRL junk.
    Strategy:
    - Find the earliest strong cover anchor anywhere in the document.
    - End at the 'TABLE OF CONTENTS'/'INDEX' marker or the next major boundary after the anchor.
    - As a fallback, use a window and aggressively filter IX/XBRL-like lines anywhere in the slice.
    """
    # Strong anchors for cover start (tolerate hyphen variants and common phrasing)
    form10k_rx = r"FORM\s*10[\-\u2010-\u2015\u2212]?\s*K\b"
    anchors = [
        r"UNITED\s+STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION",
        r"SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION",
        form10k_rx,
        r"Commission\s+File\s+(?:Number|No\.?)+",
        r"Washington,\s*D\.?C\.?\s*20549",
        r"Annual\s+Report\s+Pursuant\s+to\s+Section\s+13|15\(d\)",
    ]

    # Search entire document for the first occurrence of any anchor
    start_positions: List[int] = []
    for rx in anchors:
        m = re.search(rx, full_text, re.IGNORECASE)
        if m:
            start_positions.append(m.start())
    if start_positions:
        start_idx = min(start_positions)
    else:
        start_idx = 0  # fallback to beginning

    # Determine end bound by searching AFTER start
    end_markers = [
        r"\bTABLE\s+OF\s+CONTENTS\b",
        r"\bINDEX\b",
        r"\n\s*Item\s*1\b",
        r"\bPART\s+I\b",
    ]
    end_idx = len(full_text)
    tail = full_text[start_idx:]
    for rx in end_markers:
        m = re.search(rx, tail, re.IGNORECASE)
        if m:
            end_idx = min(end_idx, start_idx + m.start())
    # If no end marker found, cap to a reasonable window
    if end_idx == len(full_text):
        end_idx = min(len(full_text), start_idx + 120000)

    candidate = full_text[start_idx:end_idx]

    # Remove IX/XBRL junk lines anywhere in the candidate
    lines = candidate.splitlines()
    http_rx = re.compile(r"https?://", re.IGNORECASE)
    colon_ns_rx = re.compile(r"\b[a-z][a-z0-9\-]*:[A-Za-z0-9_]+", re.IGNORECASE)
    numeric_only_rx = re.compile(r"^\d{8,}$")
    iso_date_rx = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    period_rx = re.compile(r"^P\d+[MYDQ]$", re.IGNORECASE)

    filtered: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            filtered.append("")
            continue
        if http_rx.search(s):
            continue
        if colon_ns_rx.search(s):  # drop any QName-like namespace tokens anywhere in the line
            continue
        if numeric_only_rx.match(s):
            continue
        if iso_date_rx.match(s):
            continue
        if period_rx.match(s):
            continue
        filtered.append(ln)

    cleaned = "\n".join(filtered)
    # Collapse excessive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # If cleaning removed everything, fall back to the original candidate trimmed
    return cleaned if cleaned else candidate.strip()


def _html_to_visible_text(html: str) -> str:
    """Convert HTML to visible text, dropping IX/XBRL, script/style, and hidden elements."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Drop script/style
        for t in soup(["script", "style"]):
            t.decompose()
        # Drop/unwrap any namespaced (prefix:name) elements (ix, dei, us-gaap, etc.)
        for el in list(soup.find_all(True)):
            name = (getattr(el, "name", "") or "")
            if ":" in name:
                # Keep inner text by unwrapping instead of removing
                try:
                    el.unwrap()
                except Exception:
                    el.decompose()
                continue
            # Drop hidden elements
            style = (el.get("style") or "").lower()
            if ("display:none" in style) or ("visibility:hidden" in style) or ("opacity:0" in style):
                el.decompose()
                continue
            if el.has_attr("hidden") or el.get("aria-hidden") in ("true", "1") or el.get("type") == "hidden":
                el.decompose()
                continue
        text = soup.get_text("\n")
        return text
    except Exception:
        return ""

# --------------------------- Main extraction logic ---------------------------

def extract_10k_data(cik: str) -> Dict[str, Any]:
    filing = find_10k_filing(cik)
    if not filing:
        return {"cik": cik, "error": "10-K filing not found"}
    text, html = read_filing_content(filing)

    # Build HTML-visible text first if available; it tends to be cleaner than raw .txt
    html_visible = _html_to_visible_text(html) if html else None
    cover_source_text = html_visible if html_visible else text

    cover_text = _extract_cover_text(cover_source_text)

    # If the result still looks like IX/XBRL junk (many token lines), retry with the alternate source
    alt_source = text if cover_source_text is html_visible else (html_visible or "")
    if alt_source and cover_text:
        lines = [ln.strip() for ln in cover_text.splitlines()[:60] if ln.strip()]
        junk_like = 0
        # Heuristic to detect noisy IXBRL tokens
        junk_rx = re.compile(r"(https?://|\b[a-z][a-z0-9\-]*:[A-Za-z0-9_]+|P\d+[MYDQ]|\b\d{4}-\d{2}-\d{2}\b|^\d{8,}$)", re.IGNORECASE)
        for ln in lines:
            if junk_rx.search(ln):
                junk_like += 1
        if lines and junk_like / max(len(lines), 1) > 0.25:
            retry = _extract_cover_text(alt_source)
            if retry:
                cover_text = retry

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

    # New: extract additional sections from the main 10-K body (only if present)
    body_source_text = html_visible if html_visible else text
    market_sections = _extract_item5_market_equity(body_source_text)
    stock_sections = _extract_stockholders_equity_notes(body_source_text)
    ex4_sections = _extract_description_of_securities_from_body(body_source_text)
    capital_stock_sections = _extract_capital_stock_notes(body_source_text)
    # Fallback: if no capital stock found in HTML-visible text, try raw text
    if not capital_stock_sections and html_visible and text:
        fallback_sections = _extract_capital_stock_notes(text)
        if fallback_sections:
            capital_stock_sections = fallback_sections

    # Build and return payload
    result: Dict[str, Any] = {
        "cik": cik,
        "filing_path": filing.replace("/", "\\"),
        "extracted_at": datetime.datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "cover_page": {"sections": cover_sections},
        "stockholders_equity_notes": {"sections": stock_sections},
        "exhibit_4_securities": {"sections": ex4_sections},
        "charter_bylaws": {"sections": []},
        "market_equity": {"sections": market_sections},
        "capital_stock": {"sections": capital_stock_sections},
    }
    return result

def _extract_item5_market_equity(full_text: str) -> List[Dict[str, str]]:
    """Extract Item 5: Market for Registrant's Common Equity... block from 10-K body if present."""
    # Common anchors with flexible whitespace/quotes
    anchors = [
        r"ITEM\s*5\s*\.\s*Market\s+for\s+Registrant['’`]?s\s+Common\s+Equity[\s\S]{0,60}?\b",
        r"\bMARKET\s+FOR\s+REGISTRANT['’`]?S\s+COMMON\s+EQUITY\b",
    ]
    start_idx = None
    for rx in anchors:
        m = re.search(rx, full_text, re.IGNORECASE)
        if m:
            start_idx = m.start()
            break
    if start_idx is None:
        return []
    block = _slice_to_next(
        full_text,
        start_idx,
        [r"\n\s*Item\s*6\b", r"\n\s*Item\s*7\b", r"\bPART\s+II\b", r"\bINDEX\b", r"\bTABLE\s+OF\s+CONTENTS\b"],
    )
    cleaned = _clean_ixbrl_lines(block)
    if len(cleaned) < 400:  # too short, likely a false positive
        return []
    return [{"heading": "Item 5 — Market for Registrant’s Common Equity", "content": cleaned}]


def _extract_stockholders_equity_notes(full_text: str) -> List[Dict[str, str]]:
    """Extract notes titled Stockholders’/Shareholders’ Equity or Capital Stock from the notes section."""
    results: List[Dict[str, str]] = []
    # Match note headings lines; allow optional "Note N" prefix and various dashes/quotes
    note_heading_rx = re.compile(
        r"(?im)^\s*(?:Note\s+\d+\s*(?:[-—–:]\s*)?)?(Stockholders?['’`]?\s*Equity|Shareholders?['’`]?\s*Equity|Capital\s+Stock)\b.*$"
    )
    # Boundaries: next Note heading, next Item, or structural markers
    stop_rxs = [r"(?im)^\s*Note\s+\d+\b", r"\n\s*Item\s*\d+\b", r"\bREPORT\s+OF\b", r"\bTABLE\s+OF\s+CONTENTS\b", r"\bINDEX\b"]

    for m in note_heading_rx.finditer(full_text):
        start = m.start()
        block = _slice_to_next(full_text, start, stop_rxs)
        cleaned = _clean_ixbrl_lines(block)
        # Require some substance
        if len(cleaned) >= 400:
            # Use the matched line as heading text
            heading_line = m.group(0).strip()
            heading = _clean_ws(heading_line)
            results.append({"heading": heading, "content": cleaned})
    return results


def _extract_description_of_securities_from_body(full_text: str) -> List[Dict[str, str]]:
    """Extract any in-body "Description of Securities/Capital Stock" if present (some issuers include it)."""
    anchors = [
        r"\bDescription\s+of\s+Registrant['’`]?s\s+Securities\b",
        r"\bDescription\s+of\s+Capital\s+Stock\b",
        r"\bDescription\s+of\s+Our\s+Capital\s+Stock\b",
    ]
    start_idx = None
    heading_text = None
    for rx in anchors:
        m = re.search(rx, full_text, re.IGNORECASE)
        if m:
            start_idx = m.start()
            heading_text = _clean_ws(full_text[m.start(): m.end()])
            break
    if start_idx is None:
        return []
    block = _slice_to_next(
        full_text,
        start_idx,
        [r"\n\s*Item\s*\d+\b", r"\bEXHIBIT[S]?\s+INDEX\b", r"\bINDEX\b", r"\bTABLE\s+OF\s+CONTENTS\b"],
    )
    cleaned = _clean_ixbrl_lines(block)
    if len(cleaned) < 400:
        return []
    return [{"heading": heading_text or "Description of Securities", "content": cleaned}]


def _extract_capital_stock_notes(full_text: str) -> List[Dict[str, str]]:
    """Extract notes/sections specifically titled 'Capital Stock' from the financial statements or notes."""
    results: List[Dict[str, str]] = []
    
    capital_stock_rx = re.compile(
        r"(?im)^\s*(?:Note\s+\d+\s*(?:[-—–:]\s*)?)?(?:\d+\.\s*)?Capital\s+[Ss]tock\b.*$"
    )
    numbered_capital_rx = re.compile(
        r"(?im)^\s*\d+\.\s*Capital\s+[Ss]tock\b.*$"
    )
    stop_rxs = [
        r"(?im)^\s*Note\s+\d+\b",
        r"(?im)^\s*\d+\.\s+[A-Z]",
        r"(?im)^\s*Item\s*\d+\b",
        r"\bREPORT\s+OF\b",
        r"\bTABLE\s+OF\s+CONTENTS\b",
        r"\bINDEX\b",
    ]

    debug = os.getenv("LTP_DEBUG_CAPITAL")
    total_matches = 0
    min_len = 200

    for rx in [capital_stock_rx, numbered_capital_rx]:
        for m in rx.finditer(full_text):
            total_matches += 1
            start = m.start()
            # Use robust multiline slicer for this section
            block = _slice_to_next_multiline(full_text, start, stop_rxs, max_window=120000)
            cleaned = _clean_ixbrl_lines(block)
            if debug:
                print("[DEBUG] CapStock heading:", _clean_ws(m.group(0))[:120])
                print("[DEBUG] Raw block len:", len(block), "Cleaned len:", len(cleaned))
                preview = cleaned[:300].replace("\n", " ")
                print("[DEBUG] Cleaned preview:", preview)
            if len(cleaned) >= min_len:
                heading_line = m.group(0).strip()
                heading = _clean_ws(heading_line)
                is_duplicate = any(
                    (heading == ex.get("heading")) or (cleaned[:300] in ex.get("content", ""))
                    for ex in results
                )
                if not is_duplicate:
                    results.append({"heading": heading, "content": cleaned})
    if debug:
        print(f"[DEBUG] CapitalStock matches={total_matches}, extracted={len(results)} (min_len={min_len})")
        for r in results:
            print(f"[DEBUG] Heading: {r['heading'][:120]}")
    return results

# --------------------------- CLI Entry Point ---------------------------

def run_cli():
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Extract 10-K sections into sections-only JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cik", help="CIK to process (10 digits or fewer)")
    group.add_argument("--file", help="Direct path to a 10-K .txt/.htm file")
    parser.add_argument("--output", help="Output path (default: staging/cik_{cik}_equity_extraction.json)")
    args = parser.parse_args()

    if args.file:
        # If file provided, try to infer cik from filename digits
        import re, os
        m = re.search(r"(\d{7,10})", os.path.basename(args.file))
        cik = (m.group(1) if m else "unknown").zfill(10) if m else "unknown"
        # Temporarily write the file path into our discovery logic
        # Bypass find_10k_filing and read directly
        text, html = read_filing_content(args.file)
        html_visible = _html_to_visible_text(html) if html else None
        cover_source_text = html_visible if html_visible else text
        # Build payload using same logic
        cover_text = _extract_cover_text(cover_source_text)
        alt_source = text if cover_source_text is html_visible else (html_visible or "")
        if alt_source and cover_text:
            lines = [ln.strip() for ln in cover_text.splitlines()[:60] if ln.strip()]
            junk_like = 0
            # Heuristic to detect noisy IXBRL tokens
            junk_rx = re.compile(r"(https?://|\b[a-z][a-z0-9\-]*:[A-Za-z0-9_]+|P\d+[MYDQ]|\b\d{4}-\d{2}-\d{2}\b|^\d{8,}$)", re.IGNORECASE)
            for ln in lines:
                if junk_rx.search(ln):
                    junk_like += 1
            if lines and junk_like / max(len(lines), 1) > 0.25:
                retry = _extract_cover_text(alt_source)
                if retry:
                    cover_text = retry
        cover_sections: List[Dict[str, str]] = []
        if cover_text:
            cover_sections.append({"heading": "Cover Page", "content": cover_text.strip()})
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

        # New: additional sections from body
        body_source_text = html_visible if html_visible else text
        market_sections = _extract_item5_market_equity(body_source_text)
        stock_sections = _extract_stockholders_equity_notes(body_source_text)
        ex4_sections = _extract_description_of_securities_from_body(body_source_text)
        capital_stock_sections = _extract_capital_stock_notes(body_source_text)
        # Fallback: try raw text if none found in HTML-visible content
        if not capital_stock_sections and html_visible and text:
            fb = _extract_capital_stock_notes(text)
            if fb:
                capital_stock_sections = fb

        cover_page = {"sections": cover_sections}

        # Final output
        output_path = args.output or f"staging/cik_{cik}_equity_extraction.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        payload = {
            "cik": cik,
            "filing_path": args.file.replace("/", "\\"),
            "extracted_at": datetime.datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "cover_page": cover_page,
            "stockholders_equity_notes": {"sections": stock_sections},
            "exhibit_4_securities": {"sections": ex4_sections},
            "charter_bylaws": {"sections": []},
            "market_equity": {"sections": market_sections},
            "capital_stock": {"sections": capital_stock_sections},
        }
        save_json_file(payload, output_path)
        print(f"[OK] Wrote {output_path}")
        return

    # Handle --cik path
    cik = args.cik.zfill(10)
    data = extract_10k_data(cik)
    data["cik"] = cik
    data["extracted_at"] = datetime.datetime.now(UTC).isoformat().replace("+00:00", "Z")
    fp = find_10k_filing(cik)
    if fp:
        data["filing_path"] = fp.replace("/", "\\")
    out_path = args.output or f"staging/cik_{cik}_equity_extraction.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_json_file(data, out_path)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    run_cli()
