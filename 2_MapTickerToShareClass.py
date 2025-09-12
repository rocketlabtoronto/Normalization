#!/usr/bin/env python3
"""
MapTickerToShareClass.py

A class to pull SEC filings (10-K/10-Q) and parse the cover page table
"Securities registered pursuant to Section 12(b) of the Act" to map
trading symbols to their corresponding share classes.
"""

import requests
import json
import re
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import os
import argparse

# Attempt to load .env if python-dotenv is installed
try:  # optional
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Manual fallback .env loader if OPENAI_API_KEY still unset
if not os.getenv("OPENAI_API_KEY") and os.path.exists(".env"):
    try:
        with open(".env", "r", encoding="utf-8") as _f:
            for _ln in _f:
                if not _ln.strip() or _ln.strip().startswith("#"):
                    continue
                if "=" in _ln:
                    k, v = _ln.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# Set stronger default model for 10-K parsing if not provided
if not os.getenv("LLM_MODEL"):
    # gpt-4o chosen for better reasoning/context window on complex 10-K tables
    os.environ["LLM_MODEL"] = "gpt-4o"


@dataclass
class ShareClassMapping:
    """Data structure for share class information"""
    cik: str
    ticker: str
    title_of_class: str
    exchange: str = ""
    economic_equivalent_to_primary: float = 1.0  # How many economic units per 1 unit of primary (usually Class A)
    share_count: Optional[float] = None          # Parsed shares outstanding (if found)
    relative_weight: Optional[float] = None      # Computed economic weight across classes


class MapTickerToShareClass:
    """
    Class to extract ticker-to-share-class mappings from SEC filings
    """
    
    def __init__(self, user_agent: str = "LookThroughProfits scott@example.com"):
        """
        Initialize the mapper with SEC API configuration
        
        Args:
            user_agent: Required user agent for SEC API requests
        """
        self.user_agent = user_agent
        self.sec_base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
    
    def get_latest_filing(self, cik: str, form_types: List[str] = ["10-K", "10-Q"]) -> Optional[Dict]:
        """
        Get the latest 10-K or 10-Q filing for a given CIK
        
        Args:
            cik: Company CIK (Central Index Key)
            form_types: List of form types to search for (default: ["10-K", "10-Q"])
            
        Returns:
            Dict containing filing information or None if not found
        """
        # Normalize CIK to 10 digits with leading zeros
        cik_padded = str(cik).zfill(10)
        
        # Get company filings
        submissions_url = f"{self.sec_base_url}/submissions/CIK{cik_padded}.json"
        
        try:
            response = requests.get(submissions_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Find the latest 10-K or 10-Q filing
            recent_filings = data.get("filings", {}).get("recent", {})
            
            if not recent_filings:
                return None
            
            forms = recent_filings.get("form", [])
            accession_numbers = recent_filings.get("accessionNumber", [])
            filing_dates = recent_filings.get("filingDate", [])
            
            # Find the most recent 10-K or 10-Q
            for i, form in enumerate(forms):
                if form in form_types:
                    return {
                        "cik": cik_padded,
                        "form_type": form,
                        "accession_number": accession_numbers[i],
                        "filing_date": filing_dates[i],
                    }
            
            return None
            
        except requests.RequestException as e:
            print(f"Error fetching filings for CIK {cik}: {e}")
            return None
    
    def get_filing_documents(self, filing_info: Dict) -> List[Dict]:
        """
        Retrieve list of documents for the filing via index.json only.
        If index.json cannot be retrieved, raise (no placeholder fallback).
        """
        cik = int(filing_info["cik"])
        accession = filing_info["accession_number"]
        accession_nodash = accession.replace('-', '')

        candidate_urls = [
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json",
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/index.json",
        ]

        raw_text = None
        last_err = None
        for url in candidate_urls:
            try:
                resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=30)
                if resp.status_code == 200 and resp.text:
                    raw_text = resp.text
                    break
            except Exception as e:
                last_err = e
        if not raw_text:
            raise RuntimeError(f"index.json not found for {accession}: {last_err if last_err else 'no response'}")

        brace_index = raw_text.find('{')
        if brace_index > 0:
            raw_text = raw_text[brace_index:]

        try:
            data = json.loads(raw_text)
        except Exception as e:
            raise RuntimeError(f"Failed to parse index.json for {accession}: {e}")

        docs: List[Dict] = []
        for item in data.get('directory', {}).get('item', []):
            name = item.get('name') or ''
            if not name:
                continue
            lower = name.lower()
            is_primary = (
                re.search(r'10-(k|q)', lower) is not None or
                lower.endswith('.htm') and 'exhibit' not in lower and 'index' not in lower
            )
            docs.append({
                'name': name,
                'is_html': lower.endswith('.htm') or lower.endswith('.html'),
                'is_txt': lower.endswith('.txt'),
                'is_primary_candidate': is_primary
            })

        if not docs:
            raise RuntimeError(f"No documents listed in index.json for {accession}")

        def score(doc: Dict) -> int:
            name = doc['name'].lower()
            s = 0
            if doc['is_primary_candidate']:
                s += 100
            if name.endswith('.htm'):
                s += 10
            if 'index' in name:
                s -= 100
            return s

        docs.sort(key=score, reverse=True)
        return docs

    def download_filing_content(self, filing_info: Dict, document_name: str, documents: Optional[List[Dict]] = None) -> Optional[str]:
        """Download the best-scoring filing document; raise if none succeed."""
        cik = int(filing_info["cik"])
        accession = filing_info["accession_number"]
        accession_nodash = accession.replace('-', '')
        base_urls = [
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}",
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}"
        ]
        candidate_names: List[str] = []
        if documents:
            candidate_names.extend([d['name'] for d in documents[:8]])
        else:
            candidate_names.append(document_name)

        tried = []
        for name in candidate_names:
            for base_url in base_urls:
                url = f"{base_url}/{name}"
                tried.append(url)
                try:
                    r = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=60)
                    ct = r.headers.get('Content-Type', '').lower()
                    if r.status_code == 200 and ('html' in ct or r.text.strip().startswith('<html')):
                        return r.text
                except Exception:
                    continue
        raise RuntimeError(f"Failed to download primary document after trying: {len(tried)} URLs")

    def parse_cover_page_table(self, content: str) -> List[ShareClassMapping]:
        """Parse cover page table using LLM first; fallback to deterministic HTML parse if LLM fails."""
        soup = BeautifulSoup(content, 'html.parser')
        # Replace deprecated find_all(text=True) usage
        for tag in soup.find_all(string=True):
            if tag and isinstance(tag, str):
                tag.replace_with(tag.replace('\u2014', '-').replace('\u2013', '-').replace('—', '-'))
        candidate_tables = []
        for tbl in soup.find_all('table'):
            txt = tbl.get_text(" ", strip=True).lower()
            if ('securities registered' in txt or '12(b)' in txt or 'trading symbol' in txt) and ('title' in txt or 'class' in txt) and ('symbol' in txt or 'trading' in txt):
                candidate_tables.append((len(txt), tbl))
        table_html = ""
        if candidate_tables:
            candidate_tables.sort(key=lambda x: x[0], reverse=True)
            table_html = str(candidate_tables[0][1])

        context_text = self._collect_context_for_llm(soup)
        prompt = self._build_llm_prompt(table_html, context_text)
        try:
            llm_data = self._call_llm(prompt)
        except Exception as e:
            print(f"    ⚠️ LLM primary parse failed: {e}. Attempting deterministic fallback...", flush=True)
            llm_data = None
        if not llm_data or 'rows' not in llm_data or not isinstance(llm_data.get('rows'), list) or not llm_data['rows']:
            fallback_rows = self._fallback_parse_cover_table(table_html or content)
            if fallback_rows:
                print(f"    🔧 Fallback HTML parser recovered {len(fallback_rows)} rows", flush=True)
                llm_data = {"rows": fallback_rows}
            else:
                raise RuntimeError("Cover page parsing failed (LLM + fallback produced no rows)")

        mappings: List[ShareClassMapping] = []
        for row in llm_data['rows']:
            tkr = (row.get('ticker') or '').upper().strip()
            # Treat punctuation placeholders or em-dashes as non-tradable (empty ticker)
            if tkr in {'—', '--', '-', 'N/A', 'NA'}:
                tkr = ''
            title = (row.get('title_of_class') or '').strip()
            if not title:
                continue
            exch = (row.get('exchange') or '').strip()
            econ = row.get('economic_equivalent_to_primary') or 1.0
            try:
                econ = float(econ)
            except Exception:
                econ = 1.0
            sh = row.get('share_count')
            try:
                sh = float(sh) if sh is not None else None
            except Exception:
                sh = None
            mappings.append(ShareClassMapping(
                cik="",
                ticker=tkr,
                title_of_class=title,
                exchange=exch,
                economic_equivalent_to_primary=econ,
                share_count=sh,
            ))

        if any(m.share_count for m in mappings):
            total = 0.0
            for m in mappings:
                total += (m.share_count or 0.0) * (m.economic_equivalent_to_primary or 1.0)
            if total > 0:
                for m in mappings:
                    m.relative_weight = ((m.share_count or 0.0) * (m.economic_equivalent_to_primary or 1.0)) / total
        return mappings

    def _collect_context_for_llm(self, soup: BeautifulSoup) -> str:
        """Extract sentences likely to contain share counts or conversion ratios."""
        text = soup.get_text('\n')
        # Keep lines with key terms
        keep_terms = re.compile(r'(class\s+a|class\s+b|convert|conversion|outstanding|shares outstanding|each class)', re.IGNORECASE)
        lines = []
        for ln in text.splitlines():
            if keep_terms.search(ln):
                clean = re.sub(r'\s+', ' ', ln).strip()
                if 20 <= len(clean) <= 500:
                    lines.append(clean)
            if len(lines) > 120:
                break
        # Truncate overall length to keep prompt compact
        joined = '\n'.join(lines)
        if len(joined) > 8000:
            joined = joined[:8000]
        return joined

    def _build_llm_prompt(self, table_html: str, context_text: str) -> str:
        instructions = (
            "You are an expert financial document parser. Parse ONLY the cover page table 'Securities registered pursuant to Section 12(b) of the Act'. "
            "Return strict JSON with key 'rows'. Each row: 'ticker', 'title_of_class', 'exchange', 'economic_equivalent_to_primary', 'share_count'. "
            "Rules: (1) If the table shows an em dash/em-dash/— or blank under Symbol, set ticker to ''. (2) Ignore debt/notes without tickers. "
            "(3) Normalize whitespace; do not hallucinate values. (4) Use 1.0 for primary class's economic_equivalent_to_primary. "
            "(5) Use share_count only if explicitly present near the class names in provided context lines; otherwise omit." )
        payload = {
            "instructions": instructions,
            "cover_table_html": table_html[:12000],
            "context_lines": context_text,
            "output_schema_example": {
                "rows": [
                    {"ticker": "EXAMPLE", "title_of_class": "Class A Common Stock", "exchange": "NYSE", "economic_equivalent_to_primary": 1.0, "share_count": 123456789}
                ]
            }
        }
        return json.dumps(payload, indent=2)

    def _call_llm(self, prompt: str) -> Optional[Dict]:
        """Invoke LLM provider with retries & robust JSON extraction. Returns dict or raises."""
        api_key = os.getenv('OPENAI_API_KEY')
        model = os.getenv('LLM_MODEL', 'gpt-4o')
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            import openai  # type: ignore
        except ImportError:
            raise RuntimeError("openai package not installed; run: pip install openai")

        def _extract_json(raw: str) -> Dict:
            cleaned = raw.strip()
            # Remove code fences
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            # Find first '{'
            if '{' not in cleaned:
                raise ValueError('No JSON object found')
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
            if end == -1:
                raise ValueError('Unbalanced braces in JSON candidate')
            candidate = cleaned[start:end]
            return json.loads(candidate)

        last_err = None
        for attempt in range(1, 4):
            try:
                print(f"    📡 Calling OpenAI API attempt {attempt}/3 ({model})...", flush=True)
                if hasattr(openai, 'OpenAI'):
                    client = openai.OpenAI(api_key=api_key)
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You output ONLY valid compact JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0
                    )
                    content = resp.choices[0].message.content or ''  # type: ignore
                else:
                    openai.api_key = api_key  # type: ignore
                    resp = openai.ChatCompletion.create(  # type: ignore
                        model=model,
                        messages=[
                            {"role": "system", "content": "You output ONLY valid compact JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0
                    )
                    content = resp['choices'][0]['message']['content'] or ''  # type: ignore
                if not content.strip():
                    raise ValueError('Empty response from LLM')
                print(f"    ✅ Received response ({len(content)} chars) preview: {content[:100].replace('\n',' ')}...", flush=True)
                parsed = _extract_json(content)
                if 'rows' not in parsed:
                    raise ValueError("Parsed JSON missing 'rows'")
                print(f"    ✅ Successfully parsed AI response with {len(parsed.get('rows', []))} rows", flush=True)
                return parsed
            except Exception as e:
                last_err = e
                print(f"    ⚠️ LLM attempt {attempt} failed: {e}", flush=True)
                if attempt < 3:
                    backoff = 1.2 * attempt
                    print(f"    ⏳ Backing off {backoff:.1f}s then retrying...", flush=True)
                    time.sleep(backoff)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def _fallback_parse_cover_table(self, html_fragment: str) -> List[Dict[str, Any]]:
        """Deterministic parse of cover table HTML (or whole filing) extracting class/ticker pairs.
        Returns list[ {ticker,title_of_class,exchange,economic_equivalent_to_primary,share_count} ].
        """
        rows: List[Dict[str, Any]] = []
        try:
            soup = BeautifulSoup(html_fragment, 'html.parser')
        except Exception:
            return rows
        tables = soup.find_all('table')
        for tbl in tables:
            txt = tbl.get_text(' ', strip=True).lower()
            if not (('trading symbol' in txt or 'trading symbols' in txt) and ('securities registered' in txt or 'section 12' in txt or 'pursuant to' in txt)):
                continue
            # Iterate table rows
            for tr in tbl.find_all('tr'):
                cells = [c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
                if len(cells) < 2:
                    continue
                line = ' '.join(cells).lower()
                if 'trading symbol' in line and 'title' in line:
                    # header row
                    continue
                possible_tickers: List[str] = []
                class_text = None
                for c in cells:
                    c_clean = c.strip()
                    # Identify class description
                    if (('class' in c.lower() or 'common stock' in c.lower()) and len(c_clean) > 5) or (class_text is None and 'stock' in c.lower()):
                        if not class_text or ('class' in c.lower() and len(c_clean) < len(class_text)):
                            class_text = c_clean
                    # Extract tickers tokens
                    for token in re.split(r'[ ,;/]+', c_clean):
                        tok = token.strip().upper()
                        if tok in {'—', '--', '-', 'N/A', 'NA'}:
                            possible_tickers.append('')
                        elif re.fullmatch(r'[A-Z]{1,5}(?:\.[A-Z]{1,2})?', tok):
                            possible_tickers.append(tok)
                if not class_text:
                    continue
                if not possible_tickers:
                    # Non-tradable row
                    rows.append({
                        'ticker': '',
                        'title_of_class': class_text,
                        'exchange': '',
                        'economic_equivalent_to_primary': 1.0,
                        'share_count': None
                    })
                else:
                    for t in possible_tickers:
                        rows.append({
                            'ticker': t,
                            'title_of_class': class_text,
                            'exchange': '',
                            'economic_equivalent_to_primary': 1.0,
                            'share_count': None
                        })
            if rows:
                break  # Use first matching table only
        # Deduplicate (class,ticker)
        seen = set()
        deduped = []
        for r in rows:
            key = (r['title_of_class'].lower(), r['ticker'])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        return deduped

    def get_ticker_to_class_mapping(self, cik: str, company_name: str = "Unknown", ticker: str = "N/A") -> Dict:
        """High-level workflow to obtain ticker/share class mappings for a single CIK.
        Returns a result dict with keys: cik, company_name, primary_ticker, success, error, ticker_mappings, filing_info.
        """
        result = {
            "cik": str(cik).zfill(10),
            "company_name": company_name,
            "primary_ticker": ticker,
            "ticker_mappings": [],
            "filing_info": None,
            "success": False,
            "error": None
        }
        try:
            time.sleep(0.4)  # polite rate limit
            print(f"  📄 Searching for latest 10-K/10-Q filing...", flush=True)
            filing_info = self.get_latest_filing(cik)
            if not filing_info:
                result['error'] = 'No recent 10-K/10-Q filing'
                print(f"  ❌ No recent 10-K/10-Q filing found", flush=True)
                return result
            result['filing_info'] = filing_info
            print(f"  ✅ Found {filing_info['form_type']} from {filing_info['filing_date']}", flush=True)

            print(f"  📋 Retrieving document list...", flush=True)
            documents = self.get_filing_documents(filing_info)
            print(f"  ✅ Found {len(documents)} documents", flush=True)

            print(f"  ⬇️  Downloading filing content...", flush=True)
            primary_doc_name = documents[0]['name'] if documents else ''
            content = self.download_filing_content(filing_info, primary_doc_name, documents)
            if not content:
                result['error'] = 'Failed to download primary document'
                print(f"  ❌ Failed to download filing content", flush=True)
                return result
            print(f"  ✅ Downloaded {len(content):,} characters", flush=True)

            print(f"  🤖 Parsing cover page (LLM + fallback)...", flush=True)
            mappings = self.parse_cover_page_table(content)
            if not mappings:
                result['error'] = 'No mappings returned'
                print(f"  ⚠️ No share class mappings produced", flush=True)
                return result

            for m in mappings:
                m.cik = result['cik']
            result['ticker_mappings'] = [
                {
                    'ticker': m.ticker,
                    'title_of_class': m.title_of_class,
                    'exchange': m.exchange,
                    'economic_equivalent_to_primary': m.economic_equivalent_to_primary,
                    'share_count': m.share_count,
                    'relative_weight': m.relative_weight
                } for m in mappings
            ]
            result['success'] = True
            print(f"  ✅ SUCCESS: {len(mappings)} mappings", flush=True)
            return result
        except Exception as e:
            result['error'] = str(e)
            print(f"  ❌ ERROR: {e}", flush=True)
            return result
        

# Helper functions for augmenting non-traded classes

def _normalize_class_key(label: str) -> str:
    """Normalize class key for matching"""
    low = label.lower()
    # Convert common separators to underscores, collapse multiple underscores
    return re.sub(r'[^a-z0-9]+', '_', low).strip('_') or '__unknown__'


def _augment_with_untraded_classes(result: Dict, company: Dict) -> None:
    """Add placeholder mappings for missing share classes"""
    if not result.get('success') or not result.get('ticker_mappings'):
        return
    
    existing_keys = set()
    for mapping in result['ticker_mappings']:
        key = _normalize_class_key(mapping.get('title_of_class', '') + '|' + mapping.get('ticker', ''))
        existing_keys.add(key)
    
    # Check both untraded_classes and main classes array
    all_classes = []
    
    # Add untraded classes if present
    untraded_classes = company.get('untraded_classes', [])
    for uc in untraded_classes:
        title = uc.get('title_of_class', '').strip()
        if title:
            all_classes.append(title)
    
    # Add main classes if not found on cover page
    main_classes = company.get('classes', [])
    for mc in main_classes:
        title = mc.get('class_name', '').strip()
        if title:
            all_classes.append(title)
    
    # Add missing classes as placeholder mappings
    for title in all_classes:
        key = _normalize_class_key(title + '|')
        if key in existing_keys:
            continue  # Skip if already found on cover page
        # Add placeholder mapping with empty ticker
        result['ticker_mappings'].append({
            "ticker": "",
            "title_of_class": title,
            "exchange": "",
            "economic_equivalent_to_primary": 1.0,
            "share_count": None,
            "relative_weight": None
        })
    # Recalculate weights including placeholders
    if any(m.get('share_count') for m in result['ticker_mappings']):
        total = 0.0
        for m in result['ticker_mappings']:
            total += (m.get('share_count') or 0.0) * (m.get('economic_equivalent_to_primary') or 1.0)
        if total > 0:
            for m in result['ticker_mappings']:
                m['relative_weight'] = ((m.get('share_count') or 0.0) * (m.get('economic_equivalent_to_primary') or 1.0)) / total


def run_batch_processing(test_mode: Union[bool, int] = False):
    """Run Step 2 on all companies from dual_class_output.json"""
    
    print("Starting Step 2: Ticker-to-Share-Class Mapping")
    print("=" * 60)
    
    # Load input data
    input_file = 'dual_class_output.json'
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found. Run Step 1 first.")
        return False
    
    print(f"Loading data from {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except UnicodeDecodeError as e:
        print(f"ERROR: Unicode decode error in {input_file}: {e}")
        # Try alternative encodings
        for encoding in ['utf-8-sig', 'latin1', 'cp1252']:
            try:
                print(f"   Trying encoding: {encoding}")
                with open(input_file, 'r', encoding=encoding) as f:
                    input_data = json.load(f)
                print(f"   Successfully read with {encoding}")
                break
            except Exception as e2:
                print(f"   Failed with {encoding}: {e2}")
                continue
        else:
            print(f"ERROR: Could not read {input_file} with any encoding")
            return False
    except Exception as e:
        print(f"ERROR: Failed to read {input_file}: {e}")
        return False
    
    companies = input_data.get('companies', [])
    
    # Filter to companies with CIKs
    companies_with_cik = [c for c in companies if c.get('cik')]
    
    # Limit for test mode
    if test_mode:
        # If test_mode is an integer, use that as the limit; otherwise default to 5
        limit = test_mode if isinstance(test_mode, int) else 5
        companies_with_cik = companies_with_cik[:limit]
        print(f"Running in test mode - processing first {len(companies_with_cik)} companies with CIKs")
    
    print(f"Found {len(companies_with_cik)} companies with CIKs to process")
    
    if not companies_with_cik:
        print("❌ No companies with CIKs found to process")
        return False
    
    # Initialize mapper
    mapper = MapTickerToShareClass()
    
    # Process each company
    results = []
    successful_mappings = 0
    
    for i, company in enumerate(companies_with_cik, 1):
        company_name = company.get('company_name', 'Unknown')
        primary_ticker = company.get('primary_ticker', 'N/A')
        cik = company.get('cik')
        
        print(f"\n" + "="*80, flush=True)
        print(f"[{i:2}/{len(companies_with_cik)}] 🏢 {company_name} ({primary_ticker})", flush=True)
        print("="*80, flush=True)
        
        try:
            result = mapper.get_ticker_to_class_mapping(cik, company_name, primary_ticker)
            # Augment with any share classes not traded (add with empty ticker)
            _augment_with_untraded_classes(result, company)
            results.append(result)
            
            if result.get('success') and result.get('ticker_mappings'):
                successful_mappings += 1
                print(f"🎯 Company {i} completed successfully!")
            else:
                print(f"⚠️  Company {i} completed with warnings: {result.get('error', 'No mappings found')}")
        
        except Exception as e:
            print(f"❌ ERROR processing {company_name}: {e}")
            results.append({
                "cik": cik,
                "company_name": company_name,
                "primary_ticker": primary_ticker,
                "success": False,
                "error": str(e),
                "ticker_mappings": []
            })
        
        # Progress update every 5 companies
        if i % 5 == 0 or i == len(companies_with_cik):
            print(f"\n📈 PROGRESS UPDATE: {i}/{len(companies_with_cik)} companies processed")
            print(f"   • Successful mappings: {successful_mappings}")
            print(f"   • Success rate: {successful_mappings/i*100:.1f}%")
    
    # Save results
    output_data = {
        "step": "2_MapTickerToShareClass",
        "processed_date": "2025-09-09",
        "total_companies_processed": len(companies_with_cik),
        "successful_mappings": successful_mappings,
        "test_mode": test_mode,
        "results": results
    }
    
    output_file = 'step2_ticker_mappings_test.json' if test_mode else 'step2_ticker_mappings.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Step 2 complete!")
    print(f"📊 Successfully processed {successful_mappings}/{len(companies_with_cik)} companies")
    print(f"💾 Results written to {output_file}")
    
    # Show detailed statistics
    if successful_mappings > 0:
        total_mappings = sum(len(r.get('ticker_mappings', [])) for r in results)
        print(f"📈 Total ticker mappings found: {total_mappings}")
        
        # Show some example mappings
        print(f"\n📋 Example mappings found:")
        count = 0
        for result in results:
            if result.get('success') and result.get('ticker_mappings'):
                print(f"   • {result.get('company_name', 'Unknown')}: {len(result['ticker_mappings'])} mappings")
                for mapping in result['ticker_mappings'][:2]:  # Show first 2 mappings
                    print(f"     - {mapping.get('ticker')}: {mapping.get('title_of_class')}")
                count += 1
                if count >= 3:  # Show 3 examples
                    break
    
    return successful_mappings > 0


def _cli():
    p = argparse.ArgumentParser(description="LLM-based extraction of share class mappings from latest 10-K/10-Q cover table.")
    p.add_argument("--cik", help="CIK (digits) for single company processing")
    p.add_argument("--batch", action="store_true", help="Process all companies from dual_class_output.json")
    p.add_argument("--test", nargs='?', const=3, type=int, help="Test mode - process N companies (default: 3)")
    p.add_argument("--model", help="Override model (sets LLM_MODEL env var for this run)")
    p.add_argument("--raw", action="store_true", help="Print raw JSON only (no extra text)")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON (default if neither raw nor pretty specified)")
    args = p.parse_args()

    if args.model:
        os.environ["LLM_MODEL"] = args.model

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    if args.batch:
        # Batch processing mode
        success = run_batch_processing(test_mode=args.test)
        if not success:
            raise SystemExit(1)
    elif args.cik:
        # Single company mode
        cik = re.sub(r"\D", "", args.cik)
        if not cik:
            raise SystemExit("Provide numeric CIK")

        mapper = MapTickerToShareClass()
        result = mapper.get_ticker_to_class_mapping(cik)

        if args.raw:
            print(json.dumps(result))
        else:
            print(json.dumps(result, indent=2))

        if not result.get("success"):
            raise SystemExit(1)
    else:
        print("Must specify either --cik CIK_NUMBER for single company or --batch for all companies")
        raise SystemExit(1)

if __name__ == "__main__":
    _cli()
