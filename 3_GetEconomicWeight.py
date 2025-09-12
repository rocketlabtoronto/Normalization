#!/usr/bin/env python3
"""
Get economic weights for dual class companies by analyzing their SEC filings for share class information.
This script takes the dual_class_output.json and enriches it with economic weight data from SEC filings.
"""

import os
import json
import re
import requests
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys

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

# Helper: deterministic HTML fallback parsing for cover page trading symbols & classes
def _fallback_parse_cover_page(content: str) -> List[Dict[str, Any]]:
    """Parse the filing HTML deterministically to extract class names and tickers
    from the 'Securities registered pursuant to Section 12(b) of the Act' / 'Trading Symbol(s)' table.
    Returns a list of dicts with class_name, ticker (may be ''), ticker_source.
    """
    results: List[Dict[str, Any]] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return results

    try:
        soup = BeautifulSoup(content[:400000], 'lxml')  # limit parse scope
    except Exception:
        try:
            soup = BeautifulSoup(content[:400000], 'html.parser')
        except Exception:
            return results

    # Normalize text to search anchor nodes
    full_text_lower = soup.get_text(" \n", strip=True).lower()
    if 'trading symbol' not in full_text_lower and 'securities registered pursuant to section 12' not in full_text_lower:
        return results

    # Candidate tables: must contain 'Trading Symbol' and some 'Class' or 'Common Stock'
    tables = soup.find_all('table')
    for tbl in tables:
        try:
            txt = tbl.get_text(" ", strip=True)
            low = txt.lower()
            if ('trading symbol' in low or 'trading symbols' in low) and ('class' in low or 'common stock' in low):
                # Process rows
                rows = tbl.find_all('tr')
                for r in rows:
                    cells = r.find_all(['td', 'th'])
                    if len(cells) < 2:
                        continue
                    row_text = ' '.join(c.get_text(" ", strip=True) for c in cells)
                    rlow = row_text.lower()
                    if 'trading symbol' in rlow and len(cells) <= 3:
                        # Header row
                        continue
                    # Heuristic: class description usually longer, ticker cell short with A-Z chars
                    # Build candidate pairs by scanning cells
                    class_part = None
                    ticker_part = None
                    for c in cells:
                        ctext = c.get_text(" ", strip=True)
                        cstrip = ctext.strip()
                        # Potential tickers separated by commas/spaces
                        if 1 <= len(cstrip) <= 8 and re.fullmatch(r'[A-Za-z\.\-]{1,8}', cstrip):
                            ticker_part = cstrip.upper()
                        elif re.search(r'class\s+[a-z0-9]', cstrip.lower()) or 'common stock' in cstrip.lower():
                            class_part = cstrip
                    if not class_part:
                        # Try first cell as class if it contains stock keywords
                        first = cells[0].get_text(" ", strip=True)
                        if 'stock' in first.lower():
                            class_part = first
                    if class_part:
                        results.append({
                            'class_name': class_part,
                            'ticker': ticker_part if ticker_part else '',
                            'ticker_source': 'cover page fallback HTML'
                        })
        except Exception:
            continue

    # Deduplicate by normalized class key keeping first with ticker if any
    dedup: Dict[str, Dict[str, Any]] = {}
    def _norm(name: str) -> str:
        n = name.lower().strip()
        n = re.split(r'[:;(]', n)[0]
        n = re.sub(r'[^a-z0-9\s]', ' ', n)
        n = re.sub(r'\s+', ' ', n).strip()
        m = re.search(r'class\s*([a-z0-9]+)', n)
        if m:
            tok = m.group(1)
            if tok.isdigit():
                return f'class_num_{tok}'
            return f'class_{tok[0].upper()}'
        if 'common stock' in n and 'class' not in n:
            return 'common'
        return n[:60]

    for entry in results:
        key = _norm(entry['class_name'])
        if key not in dedup:
            dedup[key] = entry
        else:
            # Prefer one with ticker
            if not dedup[key].get('ticker') and entry.get('ticker'):
                dedup[key] = entry
    return list(dedup.values())

@dataclass
class EconomicWeight:
    """Data structure for economic weight information"""
    class_name: str
    ticker: str = ""
    shares_outstanding: Optional[float] = None
    economic_weight: Optional[float] = None  # Proportion of total economic value (0.0 to 1.0)
    conversion_ratio: Optional[float] = None  # How many units of primary class this represents
    source: str = ""  # Source of the data (10-K, 10-Q, etc.)
    ticker_source: Optional[str] = None  # Where the ticker came from (e.g., Trading Symbol(s) cover page)

def fetch_sec_cik_map() -> Dict[str, str]:
    """Fetch SEC ticker to CIK mapping."""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {'User-Agent': 'Economic Weight Analysis Tool (contact@example.com)'}
    
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

def get_latest_filing(cik: str, user_agent: str = "Economic Weight Analysis Tool (contact@example.com)") -> Optional[Dict]:
    """Get the latest 10-K or 10-Q filing for a given CIK"""
    cik_padded = str(cik).zfill(10)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    
    try:
        response = requests.get(submissions_url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Find the latest 10-K or 10-Q filing
        recent_filings = data.get("filings", {}).get("recent", {})
        
        if not recent_filings:
            return None
        
        forms = recent_filings.get("form", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        filing_dates = recent_filings.get("filingDate", [])
        primary_documents = recent_filings.get("primaryDocument", [])
        
        # Find the most recent 10-K or 10-Q
        for i, form in enumerate(forms):
            if form in ["10-K", "10-Q"]:
                primary_doc = primary_documents[i] if i < len(primary_documents) else None
                return {
                    "cik": cik_padded,
                    "form_type": form,
                    "accession_number": accession_numbers[i],
                    "filing_date": filing_dates[i],
                    "primary_document": primary_doc
                }
        
        return None
        
    except requests.RequestException as e:
        print(f"Error fetching filings for CIK {cik}: {e}")
        return None

def get_filing_content(filing_info: Dict, user_agent: str = "Economic Weight Analysis Tool (contact@example.com)") -> Optional[str]:
    """Download the filing content"""
    cik = int(filing_info["cik"])
    accession = filing_info["accession_number"]
    accession_nodash = accession.replace('-', '')
    primary_document = filing_info.get("primary_document")
    
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive"
    }
    
    # If we have the primary document name, use it first
    if primary_document:
        primary_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_document}"
        try:
            print(f"    → Trying primary document: {primary_url}")
            response = requests.get(primary_url, headers=headers, timeout=60)
            if response.status_code == 200 and len(response.text.strip()) > 1000:
                print(f"    → Successfully downloaded filing content ({len(response.text)} characters)")
                return response.text
            else:
                print(f"    → Primary document failed with status {response.status_code}")
        except Exception as e:
            print(f"    → Primary document error: {e}")
    
    # Try the standard SEC EDGAR URL pattern
    standard_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession_nodash}.htm"
    try:
        print(f"    → Trying standard URL: {standard_url}")
        response = requests.get(standard_url, headers=headers, timeout=60)
        if response.status_code == 200 and len(response.text.strip()) > 1000:
            print(f"    → Successfully downloaded filing content ({len(response.text)} characters)")
            return response.text
        else:
            print(f"    → Standard URL failed with status {response.status_code}")
    except Exception as e:
        print(f"    → Standard URL error: {e}")
    
    # Try alternative patterns
    alternative_urls = [
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession}.htm",
        f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{cik}/{accession_nodash}/{primary_document}" if primary_document else None,
    ]
    
    for alt_url in alternative_urls:
        if alt_url is None:
            continue
        try:
            print(f"    → Trying alternative: {alt_url}")
            response = requests.get(alt_url, headers=headers, timeout=60)
            if response.status_code == 200 and len(response.text.strip()) > 1000:
                print(f"    → Alternative URL succeeded ({len(response.text)} characters)")
                return response.text
        except Exception as e:
            print(f"    → Alternative URL error: {e}")
            continue
    
    print(f"    → All download attempts failed")
    return None

def extract_economic_weights_with_ai(content: str, company_name: str, ticker: str) -> List[EconomicWeight]:
    """Use OpenAI to extract economic weight information from SEC filing content
    Implements:
      - Content snippet reduction
      - Retry (3 attempts)
      - Raw response logging & empty guard
      - Regex / brace-scan JSON extraction
      - Deterministic HTML fallback if AI fails
    """
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('LLM_MODEL', 'gpt-4o')

    if not api_key:
        print(f"    ❌ OPENAI_API_KEY not set, skipping AI analysis")
        return []

    # Extract a focused snippet containing cover page + first occurrences of share keywords
    def _extract_relevant_snippet(html: str) -> str:
        upper_limit = 250000  # hard safety cap
        snippet = html[:upper_limit]
        # Try to isolate cover page section
        cover_idx = re.search(r'Securities registered pursuant to Section 12', snippet, re.IGNORECASE)
        if cover_idx:
            start = max(0, cover_idx.start() - 3000)
            end = min(len(snippet), cover_idx.start() + 60000)
            snippet = snippet[start:end]
        # Append first note area mentioning conversion or shares if outside clip
        m2 = re.search(r'conversion ratio|convertible into|each class', html[:upper_limit], re.IGNORECASE)
        if m2 and m2.group(0) not in snippet:
            seg_start = max(0, m2.start() - 3000)
            seg_end = min(len(html), m2.start() + 12000)
            snippet += '\n\n' + html[seg_start:seg_end]
        return snippet[:120000]  # final cap

    snippet = _extract_relevant_snippet(content)
    print(f"    🧪 Prompt snippet length: {len(snippet)} chars (original {len(content)} chars)")

    print(f"    🤖 Preparing AI prompt for OpenAI {model}...")
    prompt = f"""You are analyzing a SEC filing for {company_name} ({ticker}). This is a dual-class company.

Follow instructions precisely and return ONLY JSON (no markdown, no commentary). If data absent, use empty arrays.

REQUIRED JSON FORMAT:
{{
  "classes": [
    {{
      "class_name": "Class A Common Stock" or similar,
      "ticker": "TICKER" or empty string if not traded,
      "shares_outstanding": number or null,
      "conversion_ratio": number or null,
      "voting_rights": number or null,
      "economic_weight": number between 0 and 1 or null
    }}
  ]
}}

INSTRUCTIONS:
1. Extract information for ALL share classes mentioned in the filing
2. Look for share class names like "Class A", "Class B", "Common Stock", "Preferred Stock"
3. Find shares outstanding numbers from the cover page table or notes
4. Look for conversion ratios between classes
5. Calculate economic weights as: shares_outstanding / total_shares_all_classes
6. If shares outstanding not found, set to null
7. Return valid JSON only - no explanations, markdown, or code blocks

FILING CONTENT SNIPPET (HTML/text, sanitized):
{snippet[:100000]}"""

    try:
        try:
            import openai  # type: ignore
        except ImportError:
            print(f"    ❌ openai package not installed, skipping AI analysis")
            # Fallback directly
            fallback_entries = _fallback_parse_cover_page(content)
            if fallback_entries:
                print(f"    🔧 Fallback HTML extracted {len(fallback_entries)} classes")
                return [EconomicWeight(class_name=e['class_name'], ticker=e['ticker'], source='HTML fallback', ticker_source=e['ticker_source']) for e in fallback_entries]
            return []

        result: Optional[str] = None
        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                print(f"    📡 Calling OpenAI API attempt {attempt}/3 ({model})...")
                if hasattr(openai, 'OpenAI'):
                    client = openai.OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert financial analyst. Return ONLY strict JSON as specified."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    result = response.choices[0].message.content if response.choices else ''
                else:
                    openai.api_key = api_key  # type: ignore
                    response = openai.ChatCompletion.create(  # type: ignore
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert financial analyst. Return ONLY strict JSON as specified."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    result = response['choices'][0]['message']['content'] if response.get('choices') else ''  # type: ignore
                if not result or not str(result).strip():
                    raise ValueError('Empty LLM response')
                result = result.strip()
                print(f"    ✅ Received response ({len(result)} chars) | preview: {result[:120].replace('\n',' ')}...")
                # Attempt JSON parse
                parsed_weights = _parse_ai_json_to_weights(result, company_name, ticker)
                if parsed_weights is not None:
                    return parsed_weights
                else:
                    raise ValueError('JSON parse failed after extraction logic')
            except Exception as e:  # capture failure for retry
                last_error = e
                print(f"    ⚠️ Attempt {attempt} failed: {e}")
                if attempt < 3:
                    sleep_for = 1.5 * attempt
                    print(f"    ⏳ Backoff {sleep_for:.1f}s before retry...")
                    time.sleep(sleep_for)
                continue
        print(f"    ❌ All AI attempts failed: {last_error}")
        # Fallback deterministic parse
        fallback_entries = _fallback_parse_cover_page(content)
        if fallback_entries:
            print(f"    🔧 HTML fallback succeeded with {len(fallback_entries)} classes")
            return [EconomicWeight(class_name=e['class_name'], ticker=e['ticker'], source='HTML fallback', ticker_source=e['ticker_source']) for e in fallback_entries]
        return []
    except Exception as outer:
        print(f"Warning: AI analysis failed for {company_name}: {outer}")
        fallback_entries = _fallback_parse_cover_page(content)
        if fallback_entries:
            print(f"    🔧 HTML fallback succeeded with {len(fallback_entries)} classes")
            return [EconomicWeight(class_name=e['class_name'], ticker=e['ticker'], source='HTML fallback', ticker_source=e['ticker_source']) for e in fallback_entries]
        return []

# Helper: robust JSON extraction

def _parse_ai_json_to_weights(raw: str, company_name: str, primary_ticker: str) -> Optional[List[EconomicWeight]]:
    try:
        cleaned = raw.strip()
        # Strip markdown fences
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        # If multiple objects / commentary, isolate first JSON object using brace scan
        if '{' in cleaned:
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
            if end != -1:
                candidate = cleaned[start:end]
            else:
                candidate = cleaned[start:]
        else:
            # Try regex fallback
            m = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if not m:
                raise ValueError('No JSON braces found')
            candidate = m.group(0)
        data = json.loads(candidate)
        classes = data.get('classes', []) if isinstance(data, dict) else []
        weights: List[EconomicWeight] = []
        for c in classes:
            if not isinstance(c, dict):
                continue
            weights.append(EconomicWeight(
                class_name=c.get('class_name', ''),
                ticker=c.get('ticker', ''),
                shares_outstanding=c.get('shares_outstanding'),
                economic_weight=c.get('economic_weight'),
                conversion_ratio=c.get('conversion_ratio'),
                source='AI analysis of SEC filing',
                ticker_source=c.get('ticker_source')
            ))
        print(f"    🧩 Parsed {len(weights)} classes from AI JSON")
        return weights
    except Exception as e:
        print(f"    ❌ JSON parse helper failed: {e}")
        print(f"       Raw snippet (first 200 chars): {raw[:200].replace('\n',' ')}...")
        return None

def calculate_economic_weights(weights: List[EconomicWeight]) -> List[EconomicWeight]:
    """Calculate relative economic weights if not already provided"""
    if not weights:
        return weights
    
    # If we have shares outstanding data, calculate weights
    total_economic_value = 0.0
    weights_to_calculate = []
    
    for weight in weights:
        if weight.shares_outstanding:
            # Use conversion ratio if available, otherwise assume 1.0
            conversion_ratio = weight.conversion_ratio if weight.conversion_ratio else 1.0
            
            # For economic calculation, we need to understand the economic equivalence
            # Class A shares typically have higher economic value per share
            # If conversion_ratio < 1.0, it means this class converts to fewer of the other class
            # If conversion_ratio > 1.0, it means this class converts to more of the other class
            
            # Calculate economic value based on shares outstanding
            # For now, use shares_outstanding directly - conversion ratios will be handled
            # when we calculate relative weights between classes
            economic_value = weight.shares_outstanding * conversion_ratio
            total_economic_value += economic_value
            weights_to_calculate.append((weight, economic_value))
    
    # Calculate proportions
    if total_economic_value > 0:
        for weight, economic_value in weights_to_calculate:
            weight.economic_weight = economic_value / total_economic_value
            
            # Debug output
            print(f"    → {weight.class_name}: {weight.shares_outstanding:,} shares × {weight.conversion_ratio} = {economic_value:,.0f} economic units ({weight.economic_weight:.1%})")
    
    return weights

def analyze_company_economic_weights(company: Dict[str, Any], ticker_map: Dict[str, str]) -> Dict[str, Any]:
    """Analyze economic weights for a single company"""
    company_name = company.get('company_name', 'Unknown')
    primary_ticker = company.get('primary_ticker', 'N/A')
    
    print(f"📊 Starting analysis: {company_name} ({primary_ticker})", flush=True)
    
    # Add rate limiting - be respectful to SEC servers
    time.sleep(0.5)
    
    # Get CIK
    cik = company.get('cik')
    if not cik:
        print(f"  🔍 No CIK in data, looking up ticker {primary_ticker}...", flush=True)
        if primary_ticker and primary_ticker in ticker_map:
            cik = ticker_map[primary_ticker]
            company['cik'] = cik
            print(f"  ✅ Found CIK: {cik}", flush=True)
        else:
            print(f"  ❌ Ticker {primary_ticker} not found in SEC mapping", flush=True)
    else:
        print(f"  ✅ Using existing CIK: {cik}", flush=True)
    
    if not cik:
        print(f"  ⏭️  Skipping {company_name} - no CIK available", flush=True)
        return company
    
    # Get latest filing
    print(f"  📄 Searching for recent SEC filings (10-K/10-Q)...", flush=True)
    filing_info = get_latest_filing(cik)
    
    if not filing_info:
        print(f"  ❌ No recent 10-K/10-Q filings found for {company_name}", flush=True)
        return company
    
    print(f"  ✅ Found {filing_info['form_type']} filing from {filing_info['filing_date']}", flush=True)
    
    # Get filing content
    print(f"  ⬇️  Downloading filing content...", flush=True)
    content = get_filing_content(filing_info)
    
    if not content:
        print(f"  ❌ Could not download filing content for {company_name}", flush=True)
        return company
    
    print(f"  ✅ Downloaded {len(content):,} characters of filing data", flush=True)
    
    # Extract economic weights using AI
    print(f"  🤖 Running AI analysis to extract share class data...", flush=True)
    weights = extract_economic_weights_with_ai(
        content, 
        company_name, 
        primary_ticker
    )
    
    # Calculate relative weights if needed
    if weights:
        print(f"  🧮 Calculating economic weights...")
        weights = calculate_economic_weights(weights)
    else:
        print(f"  ⚠️  No economic data extracted by AI for {company_name}")
    
    # Merge economic data into existing classes array
    updated_classes = []
    existing_classes = company.get('classes', [])
    
    print(f"  🔗 Merging AI results with existing voting data...")
    print(f"     • AI found: {len(weights)} share classes with economic data")
    print(f"     • Existing: {len(existing_classes)} voting classes from input data")

    # Helper: load SEC ticker cache
    def _load_sec_cache():
        cache_path = 'sec_company_tickers_cache.json'
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as cf:
                return json.load(cf)
        except Exception:
            return None

    sec_cache = _load_sec_cache()

    # Helper: load Step 2 ticker mappings (test file preferred)
    def _load_step2_mappings():
        candidates = ['step2_ticker_mappings_test.json', 'step2_ticker_mappings.json']
        for c in candidates:
            if os.path.exists(c):
                try:
                    with open(c, 'r', encoding='utf-8') as sf:
                        data = json.load(sf)
                        results = data.get('results') or data.get('mappings') or []
                        mapping = {}
                        for entry in results:
                            cik_key = entry.get('cik')
                            if not cik_key:
                                continue
                            cik_z = str(cik_key).zfill(10)
                            cm = mapping.setdefault(cik_z, {})
                            for tm in entry.get('ticker_mappings', []):
                                title = tm.get('title_of_class') or tm.get('class_name') or ''
                                tkr = tm.get('ticker')
                                if not title or not tkr:
                                    continue
                                k = _norm_class_name(title)
                                # store multiple variants for robustness
                                variants = set([k, k.lower(), k.upper()])
                                # if class_ prefix present, add letter-only variants
                                if k.startswith('class_'):
                                    tok = k[len('class_'):]
                                    variants.add(tok)
                                    variants.add(tok.lower())
                                    variants.add(tok.upper())
                                    if tok.startswith('num_'):
                                        variants.add('class_' + tok)
                                for v in variants:
                                    if v not in cm:
                                        cm[v] = tkr
                        return mapping
                except Exception:
                    continue
        return None

    step2_map = _load_step2_mappings()
    print(f"    🔎 Step2 mappings loaded for {len(step2_map) if step2_map else 0} CIKs")

    def _pick_ticker(class_key: str, ai_tkr: Optional[str], ai_src: Optional[str], input_tkr: Optional[str], cik_val: Optional[str], primary: Optional[str]) -> str:
        # Debug: show lookup inputs
        print(f"      ▶ Picking ticker for class_key={class_key}, ai_tkr={ai_tkr}, ai_src={ai_src}, cik={cik_val}, primary={primary}")
        # 0) If AI explicitly found ticker from Trading Symbol(s) / cover page, prefer it
        try:
            if ai_tkr and ai_src and any(x in ai_src.lower() for x in ['trading symbol', 'cover page']):
                print(f"      ✅ Using AI ticker from cover page: {ai_tkr}")
                return ai_tkr
        except Exception:
            pass
        # 1) Step 2 mapping for this CIK and class (authoritative class-specific tickers)
        try:
            if cik_val and step2_map:
                cik_z = str(cik_val).zfill(10)
                if cik_z in step2_map:
                    print(f"      🔁 Step2 mappings for CIK {cik_z}: {list(step2_map[cik_z].keys())}")
                if cik_z in step2_map and class_key in step2_map[cik_z]:
                    print(f"      ✅ Using STEP2 ticker {step2_map[cik_z][class_key]} for class {class_key}")
                    return step2_map[cik_z][class_key]
        except Exception:
            pass
        # 2) No authoritative ticker found -> treat as non-tradable
        print(f"      ⚪ No class-specific ticker found; marking as non-tradable (ticker='')")
        return ""

    # Helper: normalize class name to a stable key
    def _norm_class_name(name: str) -> str:
        if not name:
            return "__unknown__"
        raw_lower = name.lower().strip()
        # Special cases first (issuer-agnostic)
        if 'preferred' in raw_lower and 'stock' in raw_lower:
            return 'preferred'
        if 'common stock' in raw_lower and 'class' not in raw_lower:
            return 'common'
        # Trim everything after obvious separators like ':' or ';' or '('
        short = name.split(':', 1)[0].split(';', 1)[0].split('(', 1)[0]
        s = short.lower().strip()
        # Normalize non/ voting
        s = re.sub(r"\b(non[- ]?voting|nonvoting)\b", 'non_voting', s)
        # Remove punctuation and extra spaces (keep words for class detection)
        s = re.sub(r"[^a-z0-9\s]", ' ', s)
        s = re.sub(r"\s+", ' ', s).strip()

        # Prefer explicit 'Class X' patterns (letter or numeric)
        m = re.search(r"\bclass\s*([a-z0-9]+)\b", name.lower())
        if m:
            token = m.group(1)
            if token.isdigit():
                return f"class_num_{token}"
            letter = re.search(r"[a-z]", token)
            if letter:
                return f"class_{letter.group(0).upper()}"
            return f"class_{token}"

        # Recognize plain 'common stock'
        if 'common stock' in s:
            return 'common'

        # If short phrase begins with a single-letter token like 'a' or 'b', treat as Class A/B
        m2 = re.match(r"^([a-z])\b", s)
        if m2:
            return f"class_{m2.group(1).upper()}"
    
    # Update existing classes with economic data
    for weight in weights:
        class_key = _norm_class_name(weight.class_name)
        ai_ticker = weight.ticker
        ai_source = weight.source
        input_ticker = None
        cik_value = None
        
        # Look up existing class data
        existing_class = next((c for c in existing_classes if _norm_class_name(c.get('class_name', '')) == class_key), None)
        
        if existing_class:
            print(f"    • Found existing class data for {existing_class.get('class_name')}")
            input_ticker = existing_class.get('ticker')
            cik_value = existing_class.get('cik')
            
            # Update with AI-extracted economic data
            existing_class['shares_outstanding'] = weight.shares_outstanding
            existing_class['economic_weight'] = weight.economic_weight
            existing_class['conversion_ratio'] = weight.conversion_ratio
            existing_class['source'] = weight.source
            existing_class['ticker_source'] = weight.ticker_source
            
            updated_classes.append(existing_class)
        else:
            print(f"    • No existing data for {weight.class_name}, adding new class")
            # New class entry
            new_class = {
                'class_name': weight.class_name,
                'ticker': _pick_ticker(class_key, ai_ticker, ai_source, input_ticker, cik_value, primary_ticker),
                'cik': cik,
                'shares_outstanding': weight.shares_outstanding,
                'economic_weight': weight.economic_weight,
                'conversion_ratio': weight.conversion_ratio,
                'source': weight.source,
                'ticker_source': weight.ticker_source
            }
            
            updated_classes.append(new_class)
    
    # Preserve class-specific tickers; do not override with primary_ticker
    
    # Update company data
    company['classes'] = updated_classes
    
    return company

# Main processing logic
if __name__ == "__main__":
    print(f"=== Economic Weight Analysis Pipeline ===")
    
    # Load input data
    input_file = 'dual_class_output.json'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run 1_dual_class_ingest.py first.")
        # Write minimal output to satisfy pipeline expectation
        output_data = {
            'economic_analysis_date': '2025-09-06',
            'companies_analyzed': 0,
            'successful_analyses': 0,
            'test_mode': False,
            'companies': []
        }
        # Use the test filename as a safe default placeholder
        output_file = 'dual_class_economic_weights_test.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Wrote empty output to {output_file} due to missing input.")
        sys.exit(1)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Extract companies and test mode flag
    companies = input_data.get('companies', [])
    test_mode = input_data.get('test_mode', False)
    
    print(f"Loaded {len(companies)} companies from {input_file}")
    
    # Fetch latest CIK mappings from SEC
    ticker_map = fetch_sec_cik_map()
    
    # Analyze each company
    results = []
    for company in companies:
        result = analyze_company_economic_weights(company, ticker_map)
        results.append(result)
    
    # Prepare output data
    output_data = {
        'economic_analysis_date': '2025-09-06',
        'companies_analyzed': len(results),
        'successful_analyses': sum(1 for r in results if r.get('cik')),
        'test_mode': test_mode,
        'companies': results
    }
    
    # Determine output file
    output_file = 'dual_class_economic_weights_test.json' if test_mode else 'dual_class_economic_weights.json'
    
    # Write output data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Analysis complete. Results written to {output_file}")
