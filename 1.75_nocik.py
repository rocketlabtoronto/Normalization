"""
Minimal automation for investigating a CIK's recent filings and exchange delisting signals.

Given a CIK (optionally symbol and exchange), performs:
- SEC submissions API: pull recent filings (default 18 months) and filter for 8-K, Form 25 / 25-NSE, Form 15
- For 8-Ks: fetch filing text (from Archives) and parse HTML/text for Items 1.03, 2.01, 3.01
- Order events by filing date and produce a simple narrative and structured output

Exchange checks are minimal and best-effort (require symbol):
- NASDAQ: placeholder to query Trader Corporate Actions Alerts for symbol/date range
- NYSE: placeholder to check NYSE Market Notices/Delistings
- OTC: placeholder to query FINRA Daily List around event dates

Usage:
python 1.75_nocik.py --cik 0000123456 [--symbol TICK --exchange NASDAQ]
python 1.75_nocik.py [--ai-check]

Output: writes "cik_{cik}_events.json" in the current directory and prints a short summary.
"""
from __future__ import annotations

import json
import time
import re
import datetime
import os
from typing import List, Dict, Optional

try:
    import requests
except Exception:
    requests = None

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


SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "LookThroughProfitsBot/1.0 (mailto:you@example.com)"


def sec_headers():
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def fetch_submissions_json(cik: str) -> Dict:
    """Fetch the SEC submissions JSON for a CIK (zero-padded to 10 digits)."""
    if requests is None:
        raise RuntimeError("requests required")
    cik10 = str(cik).zfill(10)
    url = SEC_SUBMISSIONS_URL.format(cik=cik10)
    r = requests.get(url, headers=sec_headers(), timeout=20)
    r.raise_for_status()
    return r.json()


def extract_recent_filings(submissions: Dict, months: int = 18) -> List[Dict]:
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


def fetch_filing_text(cik: str, accession_number: str, primary_document: str) -> Optional[str]:
    """Fetch the filing primary document text/html. Returns text or None."""
    if requests is None:
        return None
    url = build_filing_archives_url(cik, accession_number, primary_document)
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        if r.status_code == 200 and r.text:
            time.sleep(0.2)
            return r.text
        # fallback: IX viewer
        ix = f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{int(str(cik))}/{accession_number.replace('-', '')}/{primary_document}"
        r2 = requests.get(ix, headers={"User-Agent": USER_AGENT}, timeout=20)
        if r2.status_code == 200:
            time.sleep(0.2)
            return r2.text
    except Exception:
        return None
    return None


def parse_8k_items(text: str) -> Dict[str, str]:
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
    """Run the minimal workflow and return a structured report dict."""
    report: Dict = {"cik": str(cik), "requested_at": datetime.datetime.utcnow().isoformat() + "Z", "filings": [], "narrative": [], "exchange_checks": {}}
    try:
        subs = fetch_submissions_json(cik)
    except Exception as e:
        report["error"] = f"Could not fetch submissions JSON: {e}"
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
        entry = {"form": f.get("form"), "filingDate": f.get("filingDate"), "accessionNumber": f.get("accessionNumber"), "primaryDocument": f.get("primaryDocument"), "link": build_filing_archives_url(cik, f.get("accessionNumber"), f.get("primaryDocument"))}
        txt = fetch_filing_text(cik, f.get("accessionNumber"), f.get("primaryDocument"))
        if f.get("form", "").upper().startswith("8-"):
            items = parse_8k_items(txt or "")
            entry["8k_items"] = items
        # detect Form 15 presence
        if f.get("form", "").startswith("15") or f.get("form", "").upper() == "FORM 15":
            entry["deregistration"] = True
            # note 90-day clock cannot be reliably computed here without filingDate vs effective date parsing
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
    outname = f"cik_{str(cik).zfill(10)}_events.json"
    with open(outname, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

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


def build_sec_ticker_map(cache_path: str = "sec_company_tickers_cache.json", force_download: bool = False) -> Dict[str, Dict]:
    """Load or download SEC company_tickers.json and build maps: ticker_map and name_map.
    ticker_map maps ticker variant (upper) -> {'cik':..., 'ticker':..., 'title':...}
    name_map maps normalized company title -> cik
    """
    # try cache
    if not force_download and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # raw may be dict keyed by numeric ids
            ticker_map = {}
            name_map = {}
            items = raw.items() if isinstance(raw, dict) else enumerate(raw)
            for _, v in items:
                if not isinstance(v, dict):
                    continue
                cik_str = v.get("cik_str") or v.get("cik")
                ticker = (v.get("ticker") or v.get("symbol") or "").strip()
                title = v.get("title") or v.get("company_name") or v.get("name") or ""
                if not ticker or not cik_str:
                    continue
                cik = str(cik_str).zfill(10)
                for cand in generate_ticker_variants(ticker):
                    key = cand.upper()
                    if key not in ticker_map:
                        ticker_map[key] = {"cik": cik, "ticker": ticker, "title": title}
                if title:
                    name_map[title.lower()] = cik
            return {"ticker_map": ticker_map, "name_map": name_map}
        except Exception:
            pass

    if requests is None:
        raise RuntimeError("requests is required to fetch SEC company tickers; provide a local cache file.")

    r = requests.get(SEC_COMPANY_TICKERS_URL, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    raw = r.json()
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    ticker_map = {}
    name_map = {}
    items = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for _, v in items:
        if not isinstance(v, dict):
            continue
        cik_str = v.get("cik_str") or v.get("cik")
        ticker = (v.get("ticker") or v.get("symbol") or "").strip()
        title = v.get("title") or v.get("company_name") or v.get("name") or ""
        if not ticker or not cik_str:
            continue
        cik = str(cik_str).zfill(10)
        for cand in generate_ticker_variants(ticker):
            key = cand.upper()
            if key not in ticker_map:
                ticker_map[key] = {"cik": cik, "ticker": ticker, "title": title}
        if title:
            name_map[title.lower()] = cik
    return {"ticker_map": ticker_map, "name_map": name_map}


def generate_ticker_variants(ticker: str) -> List[str]:
    """Simple variant generation for tickers (handles dots/dashes and stripped forms)."""
    if not ticker:
        return []
    t = ticker.strip()
    variants = {t, t.upper(), t.replace('.', ''), t.replace('.', '-'), t.replace('-', ''), t.replace('-', '.')}
    # also produce uppercase, deduplicated list
    return [v.upper() for v in variants if v]

# --- OpenAI investigation helpers ---
import urllib.parse

def openai_investigate_company(symbol: str, company_name: str = None) -> str:
    """Use OpenAI to investigate what happened to a company/ticker.
    Returns a concise explanation of delisting, bankruptcy, acquisition, etc.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
    
    if not api_key:
        return "OpenAI API key not set in environment (OPENAI_API_KEY)"
    
    # Construct a focused prompt
    company_ref = f"{company_name} ({symbol})" if company_name else symbol
    
    prompt = f"""What happened to the company {company_ref}? Specifically, was it:
- Delisted from a stock exchange? If so, when and why?
- Acquired or merged with another company? If so, by whom and when?
- Filed for bankruptcy? If so, when?
- Suspended from trading? If so, when and why?
- Still actively trading but under a different ticker?

Please provide a brief, factual 1-2 sentence summary focusing on corporate actions that would explain why it might not appear in current SEC filings or have missing CIK data. If the company is still active and trading normally, just say "Company appears to be active and trading normally."

Answer:"""

    try:
        # Try new OpenAI client first
        try:
            import openai  # type: ignore
        except ImportError:
            return "openai package not installed; run: pip install openai"
        
        try:
            if hasattr(openai, 'OpenAI'):
                # New client pattern (v1.0+)
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.1  # Low temperature for factual responses
                )
                result = response.choices[0].message.content.strip()
            else:
                # Legacy client pattern
                openai.api_key = api_key  # type: ignore
                response = openai.ChatCompletion.create(  # type: ignore
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.1
                )
                result = response['choices'][0]['message']['content'].strip()  # type: ignore
            
            return result
        except Exception as e:
            return f"OpenAI API call failed: {e}"
    
    except Exception as e:
        return f"OpenAI investigation failed: {e}"


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

def fetch_google_news(symbol: str, max_results: int = 5, timeout: int = 10) -> List[Dict[str, str]]:
    """Fetch top Google News search snippets for a ticker. Returns list of {title, snippet, url}.
    This is a best-effort HTML scrape of google.com news search results. May fail if Google blocks requests.
    """
    if requests is None:
        return []
    q = f"{symbol} delist OR delisted OR delisting OR suspended OR \"stopped trading\" OR \"ceased trading\" OR \"filed to delist\" OR \"Form 25\" OR bankruptcy OR acquired OR merger"
    url = "https://www.google.com/search?tbm=nws&q=" + urllib.parse.quote(q)
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 429:
            print(f"  ⚠️ Google rate limited us (HTTP 429) - may need longer delays")
            return []
        elif r.status_code != 200:
            print(f"  ⚠️ Google returned HTTP {r.status_code}")
            return []
        
        r.raise_for_status()
        html = r.text
        
        # Check for blocking messages
        if "Our systems have detected unusual traffic" in html or "blocked" in html.lower():
            print(f"  ⚠️ Google is blocking automated requests")
            return []
            
    except Exception as e:
        print(f"  ⚠️ Network error: {e}")
        return []

    # crude parsing: find titles and snippets
    results: List[Dict[str, str]] = []
    # Titles: look for <div class="BNeawe vvjwJb AP7Wnd">Title</div>
    titles = re.findall(r"<div[^>]*class=[\"']?BNeawe vvjwJb AP7Wnd[\"']?[^>]*>(.*?)</div>", html)
    snippets = re.findall(r"<div[^>]*class=[\"']?BNeawe s3v9rd AP7Wnd[\"']?[^>]*>(.*?)</div>", html)
    # If patterns not found use simpler tags
    if not titles:
        titles = re.findall(r"<a [^>]*>([^<]{10,200}?)</a>", html)
    # pair up
    for i in range(min(max_results, max(len(titles), len(snippets)))):
        title = titles[i] if i < len(titles) else ""
        snippet = snippets[i] if i < len(snippets) else ""
        # strip tags
        title = re.sub(r"<.*?>", "", title)
        snippet = re.sub(r"<.*?>", "", snippet)
        results.append({"title": title.strip(), "snippet": snippet.strip(), "source": url})
    return results


def summarize_google_snippets(symbol: str, snippets: List[Dict[str, str]]) -> str:
    """Create a concise 1-2 sentence summary from Google News snippets using simple heuristics.
    Prioritize delisting/bankruptcy/merger/acquisition phrases.
    """
    if not snippets:
        return "Google News search failed or returned no results (possibly rate limited)"

    text = " ".join((s.get("title", "") + " " + s.get("snippet", "") ) for s in snippets).lower()
    # priority checks
    if any(k in text for k in ("delist", "delisted", "delisting", "filed to delist", "delisting proceedings")):
        # try extract short phrase
        m = re.search(r"(delist(ed|ing)? .*?)(?:\.|,|;|\n|$)", text)
        phrase = m.group(1) if m else "reported delisting"
        return f"News search shows reports that {symbol} was {phrase}."
    if any(k in text for k in ("acquir", "acquired", "merg", "merger")):
        m = re.search(r"(acquir(ed|ing)?|merg(er|ed)?).*?(?:\.|,|;|\n|$)", text)
        phrase = m.group(0).strip() if m else "involved in an acquisition/merger"
        return f"News indicates {symbol} was {phrase}."
    if any(k in text for k in ("bankrupt", "bankruptcy", "filed for bankruptcy")):
        return f"News indicates {symbol} filed for bankruptcy or bankruptcy-related proceedings."
    if any(k in text for k in ("suspend", "suspended trading", "halted trading")):
        return f"News indicates trading in {symbol} was suspended or halted."
    # fallback: return the top headline as summary
    top = snippets[0]
    short = (top.get("title") or top.get("snippet") or "").strip()
    if len(short) > 200:
        short = short[:200].rsplit(' ', 1)[0] + '...'
    return f"Top news: {short}"


def google_investigate_ticker(symbol: str, company_name: str = None) -> str:
    """Run Google News fetch and summarizer; return a concise 1-2 sentence reason string.
    First tries ticker, then falls back to company name if no results found.
    If network fails, returns a short error note.
    """
    try:
        print(f"  → Fetching Google News for ticker {symbol}...", flush=True)
        snippets = fetch_google_news(symbol, max_results=5)
        summary = summarize_google_snippets(symbol, snippets)
        
        # If no meaningful results found and we have a company name, try searching by company name
        if ("search failed" in summary or "returned no results" in summary or not snippets) and company_name:
            print(f"  → No ticker results found, trying company name '{company_name}'...", flush=True)
            company_snippets = fetch_google_news(company_name, max_results=5)
            if company_snippets:
                company_summary = summarize_google_snippets(company_name, company_snippets)
                if "search failed" not in company_summary and "returned no results" not in company_summary:
                    summary = f"(searched by company name) {company_summary}"
                    print(f"  → Google News result (by company name): {summary[:100]}{'...' if len(summary) > 100 else ''}", flush=True)
                    return summary
        
        print(f"  → Google News result: {summary[:100]}{'...' if len(summary) > 100 else ''}", flush=True)
        return summary
    except Exception as e:
        print(f"  → Google News check failed for {symbol}: {e}", flush=True)
        return f"Google News check failed: {e}"

# --- end Google helpers ---

def process_no_cik_file(input_path: str = "dual_class_output_nocik.json",
                        output_path: Optional[str] = None,
                        cache_path: str = "sec_company_tickers_cache.json",
                        force_download: bool = False,
                        months: int = 18,
                        run_filings_on_found: bool = True,
                        ai_check: bool = True,  # Use AI investigation instead of Google News
                        ai_delay_seconds: float = 0.5) -> Dict:
    """Load the no-CIK JSON, attempt to resolve reasons, and write back with a 'reason' and 'reason_detail' for every company.

    If a CIK is discovered, and run_filings_on_found is True, the script will call analyze_cik to fetch recent filings and attach a short filing summary.
    If ai_check is True and a primary_ticker exists, the script will use OpenAI to investigate the company and attach a concise summary into 'AI_REASON'.
    """
    if output_path is None:
        output_path = input_path

    print(f"Loading input file: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading SEC company ticker mappings (cache: {cache_path})...")
    maps = build_sec_ticker_map(cache_path=cache_path, force_download=force_download)
    ticker_map = maps.get("ticker_map", {})
    name_map = maps.get("name_map", {})
    print(f"Loaded {len(ticker_map)} ticker mappings and {len(name_map)} company name mappings")
    
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
            # ensure reason exists for every company
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
            if comp_name.lower() in ("company name", "company") or (not primary_ticker and comp.get("classes") and len(comp.get("classes"))==1 and comp.get("classes")[0].get("class_name",""
                ).lower() in ("unequal","unequal voting","n/a")):
                comp["reason"] = "header_row_or_placeholder"
                comp["reason_detail"] = {"note": "appears to be header/placeholder row"}
                print(f"  → Detected as header/placeholder row")
                # AI check
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
                    info = ticker_map.get(v.upper())
                    if info:
                        found = (v.upper(), info)
                        break
            if found:
                key, info = found
                cik = info.get("cik")
                comp["cik"] = cik
                comp["reason"] = "found_by_ticker_variant"
                comp["reason_detail"] = {"matched_variant": key, "sec_ticker": info.get("ticker"), "sec_title": info.get("title"), "cik": cik}
                print(f"  → Found CIK {cik} by ticker variant {key}")
                # optionally fetch recent filings and attach a short summary
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

            # 3) try exact name match
            nm = comp_name.lower()
            if nm in name_map:
                cik = name_map.get(nm)
                comp["cik"] = cik
                comp["reason"] = "found_by_name_exact"
                comp["reason_detail"] = {"matched_name": nm, "cik": cik}
                print(f"  → Found CIK {cik} by exact name match")
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

            # 4) fuzzy name match
            # use simple difflib against name_map keys
            import difflib
            candidates = list(name_map.keys())
            matches = difflib.get_close_matches(nm, candidates, n=1, cutoff=0.82)
            if matches:
                best = matches[0]
                score = difflib.SequenceMatcher(None, nm, best).ratio()
                comp["reason"] = "found_by_name_fuzzy"
                comp["reason_detail"] = {"matched_name": best, "score": score, "cik": name_map.get(best)}
                print(f"  → Found fuzzy name match: {best} (score: {score:.3f})")
                # only auto-assign if high confidence
                if score >= 0.9:
                    cik = name_map.get(best)
                    comp["cik"] = cik
                    print(f"  → High confidence match - assigned CIK {cik}")
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
                else:
                    print(f"  → Low confidence match - not auto-assigning CIK")
                if ai_check and primary_ticker:
                    try:
                        comp["AI_REASON"] = investigate_company_with_ai(primary_ticker, comp_name)
                        time.sleep(ai_delay_seconds)
                    except Exception as e:
                        comp["AI_REASON"] = f"ai_check_error: {e}"
                continue

            # 5) unable to find in SEC cache - mark for manual/exchange checks
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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
            print("Usage: python 1.75_nocik.py --cik CIK_NUMBER [--symbol SYMBOL --exchange EXCHANGE]")
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
