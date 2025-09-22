import os
import re
import sys
import json
import argparse
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def load_extraction_data(file_path: str) -> Dict[str, Any]:
    """Load the equity extraction JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {file_path}: {e}")
        sys.exit(1)


def clean_json_response(response_text: str) -> str:
    """Clean OpenAI response to extract valid JSON array text."""
    if not response_text:
        return "[]"
    # Remove markdown code fences
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    elif response_text.startswith('```'):
        response_text = response_text[3:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]

    response_text = response_text.strip()

    # Find JSON array bounds
    start = response_text.find('[')
    end = response_text.rfind(']')
    if start != -1 and end != -1 and end > start:
        response_text = response_text[start:end + 1]
    return response_text


def save_results(equity_classes: List[Dict[str, Any]], output_file: str, cik: str):
    """Save the extracted equity class data to JSON file with metadata."""
    result = {
        "extracted_at": datetime.now(UTC).isoformat(),
        "cik": cik,
        "total_classes": len(equity_classes),
        "extraction_method": "OpenAI " + os.getenv('LLM_MODEL', 'gpt-4o'),
        "normalization_note": "Voting and conversion weights normalized to weakest class = 1.0",
        "equity_classes": equity_classes,
    }
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Results saved to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        sys.exit(1)

def _compact_extraction(extraction: Dict[str, Any], max_total_chars: int = None, max_section_chars: int = None) -> Dict[str, Any]:
    """Return a trimmed copy of the extraction JSON to fit model limits.
    Keeps key groups and truncates long section contents while preserving headings.
    """
    if not extraction:
        return {}
    max_total_chars = max_total_chars or int(os.getenv("OPENAI_MAX_CONTEXT_CHARS", "60000"))
    # Use a conservative default for holistic pass
    max_total_chars = min(max_total_chars, 45000)
    max_section_chars = max_section_chars or int(os.getenv("OPENAI_MAX_SECTION_CHARS", "4000"))

    keep_groups = [
        "cover_page",
        "stockholders_equity_notes",
        "capital_stock",
        "exhibit_4_securities",
        "charter_bylaws",
        # market_equity can be large/noisy; include last
        "market_equity",
    ]

    out: Dict[str, Any] = {
        "cik": extraction.get("cik"),
        "source": extraction.get("source"),
        "filing_path": extraction.get("filing_path"),
    }
    total = 0

    def add_text_len(s: str) -> int:
        return len(s or "")

    for group in keep_groups:
        sec_group = (extraction.get(group) or {}).get("sections", []) or []
        if not sec_group:
            continue
        trimmed_secs: List[Dict[str, str]] = []
        for sec in sec_group:
            if total >= max_total_chars:
                break
            heading = (sec.get("heading") or "").strip()
            content = (sec.get("content") or "").strip()
            if not content:
                continue
            # collapse whitespace to save tokens
            content = re.sub(r"\s+", " ", content).strip()
            if len(content) > max_section_chars:
                content = content[:max_section_chars]
            trimmed_secs.append({"heading": heading, "content": content})
            total += add_text_len(content)
        if trimmed_secs:
            out[group] = {"sections": trimmed_secs}
        if total >= max_total_chars:
            break
    return out

def extract_with_openai(extraction_10k: Dict[str, Any], extraction_10q: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Use OpenAI to holistically extract and normalize equity class data from full 10-K and 10-Q extraction JSONs.
    Pass both documents to the LLM (read 10-K first, then 10-Q) and let it reconcile using more up-to-date information.
    """

    # Initialize OpenAI client
    try:
        import openai
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("[ERROR] OPENAI_API_KEY not found in environment variables")
            print("Please check your .env file")
            sys.exit(1)
        if hasattr(openai, 'OpenAI'):
            client = openai.OpenAI(api_key=api_key)
        else:
            openai.api_key = api_key
            client = None
    except ImportError:
        print("[ERROR] OpenAI library not installed. Install with: pip install openai")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI client: {e}")
        sys.exit(1)

    # Compact inputs to stay within token limits
    comp_10k = _compact_extraction(extraction_10k or {})
    comp_10q = _compact_extraction(extraction_10q or {}) if extraction_10q else {}

    cik = comp_10k.get('cik') or (comp_10q or {}).get('cik') or ""
    filing_k = comp_10k.get('filing_path', '')
    filing_q = (comp_10q or {}).get('filing_path', '')

    holistic = {
        "cik": cik,
        "ten_k": comp_10k,
        "ten_q": comp_10q or {},
        "note": "Sections may be truncated for length; use disclosed values and most recent dates."
    }
    holistic_json = json.dumps(holistic, ensure_ascii=False)

    # Compose prompt (LLM-only; no deterministic post-processing)
    prompt = (
        "You are a financial data analyst specializing in equity structures. "
        "You will be given the ENTIRE equity_extraction JSONs for a company: the 10-K (read first) and the 10-Q (read second). "
        "Read ALL sections holistically (cover page, stockholders' equity notes, market, capital stock, exhibits, charter/bylaws, etc.) and synthesize a final, most up-to-date view of share classes.\n\n"
        f"FILING INFO: CIK={cik}, 10-K file={filing_k}, 10-Q file={filing_q}\n\n"
        "INPUT OBJECT (JSON with keys {cik, ten_k, ten_q}):\n"
        f"{holistic_json}\n\n"
        "TASK: Return a JSON array, one object per share class, with fields: "
        "ticker_symbol (string), class_name (string), shares_outstanding (integer), conversion_weight (float), "
        "voting_weight (float), voting_rights (string), conversion_rights (string), dividend_rights (string), "
        "other_rights (string), par_value (string), authorized_shares (integer).\n\n"
        "Important requirements (LLM is the single source of truth; avoid regex and do not rely on external logic): \n"
        "- Include ALL share classes described anywhere in the provided text, including authorized but unissued classes.\n"
        "- Include ALL share classes described anywhere in the provided text, including preferred shares.\n"
        "- Do NOT invent ticker symbols. Only the cover page Section 12(b) determines tradeable tickers. If a class is not listed in 12(b), set ticker_symbol = null. If the 12(b) section shows 'None', set all ticker_symbol = null.\n"
        "- Use the most up-to-date numbers when 10-K and 10-Q differ. Generally, the 10-Q (being later) supersedes the prior 10-K for shares outstanding and similar figures as of its reporting date.\n"
        "Normalization: set the weakest class = 1.0 for voting_weight and conversion_weight and scale others accordingly.\n"
        "Return ONLY a valid JSON array with no extra text."
    )

    def _call_llm(messages, model):
        if client:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "3000"))
            ).choices[0].message.content.strip()
        else:
            import openai as _openai
            resp = _openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "3000"))
            )
            return resp['choices'][0]['message']['content'].strip()

    model = os.getenv('LLM_MODEL', 'gpt-4o')
    system_msg = {"role": "system", "content": "Extract and normalize share class data by holistically reading 10-K then 10-Q JSONs. Return only a JSON array."}
    user_msg = {"role": "user", "content": prompt}

    try:
        response_text = _call_llm([system_msg, user_msg], model)
    except Exception as e:
        # Retry once with more aggressive compaction and/or smaller model if token/rate limit
        emsg = str(e)
        print(f"[WARN] LLM call failed: {emsg}\n[INFO] Retrying with tighter context and fallback model...")
        comp_10k_small = _compact_extraction(extraction_10k or {}, max_total_chars=25000, max_section_chars=2000)
        comp_10q_small = _compact_extraction(extraction_10q or {}, max_total_chars=20000, max_section_chars=1500) if extraction_10q else {}
        holistic_small = {
            "cik": cik,
            "ten_k": comp_10k_small,
            "ten_q": comp_10q_small,
            "note": "Aggressively truncated for length."
        }
        holistic_json_small = json.dumps(holistic_small, ensure_ascii=False)
        prompt_small = prompt.replace(holistic_json, holistic_json_small)
        fallback_model = os.getenv('LLM_MODEL_FALLBACK', 'gpt-4o-mini')
        try:
            response_text = _call_llm([system_msg, {"role": "user", "content": prompt_small}], fallback_model)
        except Exception as e2:
            print(f"[ERROR] OpenAI API error: {e2}")
            traceback.print_exc()
            return []

    # Clean the response text
    response_text = clean_json_response(response_text)

    # Parse JSON
    try:
        equity_classes = json.loads(response_text)
        if not isinstance(equity_classes, list):
            print("[ERROR] OpenAI response is not a JSON array")
            print(f"Response: {response_text}")
            return []
        return [c for c in equity_classes if isinstance(c, dict) and 'class_name' in c]
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse OpenAI response as JSON: {e}")
        print(f"Response was: {response_text}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Extract normalized equity class data using OpenAI (10-K + 10-Q holistic)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cik", help="CIK number to process (reads staging/cik_{cik}_equity_extraction.json and staging/cik_{cik}_equity_extraction_10q.json if available)")
    group.add_argument("--file", help="Direct path to a single equity extraction JSON file (fallback single-doc mode)")
    parser.add_argument("--output", help="Output file path (default: fileoutput/equity_classes/cik_{cik}_equity_classes.json")

    args = parser.parse_args()

    # Determine input and CIK
    extraction_10k: Optional[Dict[str, Any]] = None
    extraction_10q: Optional[Dict[str, Any]] = None

    if args.file:
        input_file = args.file
        cik_match = re.search(r'cik_(\d+)', args.file)
        cik = cik_match.group(1) if cik_match else "unknown"
        extraction_10k = load_extraction_data(input_file)
    else:
        cik = args.cik.zfill(10)
        # 10-K extraction
        file_k = f"staging/cik_{cik}_equity_extraction.json"
        # 10-Q extraction (optional)
        file_q = f"staging/cik_{cik}_equity_extraction_10q.json"
        if os.path.exists(file_k):
            extraction_10k = load_extraction_data(file_k)
        else:
            print(f"[WARN] Missing 10-K extraction file: {file_k}")
        if os.path.exists(file_q):
            extraction_10q = load_extraction_data(file_q)
        else:
            print(f"[INFO] 10-Q extraction not found (optional): {file_q}")

    if not extraction_10k and not extraction_10q:
        print("[ERROR] No extraction inputs found (neither 10-K nor 10-Q)")
        sys.exit(1)

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = f"fileoutput/equity_classes/cik_{cik}_equity_classes.json"

    print(f"[INFO] Processing equity extraction data (holistic 10-K + 10-Q)...")
    print(f"  CIK: {cik}")
    if extraction_10k:
        print("  Input 10-K: present")
    if extraction_10q:
        print("  Input 10-Q: present")
    print(f"  Output: {output_file}")

    # Analyze with OpenAI (LLM-only)
    equity_classes = extract_with_openai(extraction_10k or {}, extraction_10q)

    if not equity_classes:
        print("[ERROR] No equity classes extracted")
        sys.exit(1)

    print(f"[SUCCESS] Extracted {len(equity_classes)} equity classes:")
    for i, equity_class in enumerate(equity_classes, 1):
        class_name = equity_class.get("class_name", "Unknown")
        shares = equity_class.get("shares_outstanding", "N/A")
        voting_weight = equity_class.get("voting_weight", "N/A")
        conversion_weight = equity_class.get("conversion_weight", "N/A")
        ticker = equity_class.get("ticker_symbol", None)

        if isinstance(shares, (int, float)):
            shares_str = f"{shares:,}"
        else:
            shares_str = str(shares)

        print(f"  {i}. {class_name} ({ticker})")
        print(f"     Shares: {shares_str}")
        print(f"     Voting weight: {voting_weight}")
        print(f"     Conversion weight: {conversion_weight}")

    # Save results
    save_results(equity_classes, output_file, cik)

    print("[SUCCESS] Equity class data extraction complete!")
    print(f"[INFO] Raw normalized data ready for analysis in {output_file}")


if __name__ == "__main__":
    main()