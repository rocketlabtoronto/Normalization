import os
import re
import sys
import json
import argparse
import traceback
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, UTC
from zoneinfo import ZoneInfo

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


def save_results(equity_classes: List[Dict[str, Any]], output_file: str, cik: str, openai_cost_usd: Optional[float] = None):
    """Save the extracted equity class data to JSON file with metadata.
    - extracted_at is formatted in Eastern Time (EST) without seconds.
    - openai_cost_usd is an estimated cost based on model token usage (if available).
    """
    # Human-readable EST timestamp without seconds
    try:
        now_et = datetime.now(UTC).astimezone(ZoneInfo("America/New_York"))
    except Exception:
        # Fallback if tz database isn't available; use local time
        now_et = datetime.now()
    extracted_at_human = now_et.strftime("%Y-%m-%d %I:%M %p EST")

    result = {
        "extracted_at": extracted_at_human,
        "cik": cik,
        "total_classes": len(equity_classes),
        "extraction_method": "OpenAI " + os.getenv('LLM_MODEL', 'gpt-4o'),
        "normalization_note": "Voting and conversion weights normalized to weakest class = 1.0",
        "equity_classes": equity_classes,
    }
    if openai_cost_usd is not None:
        # Round to 6 decimal places for readability
        result["openai_cost_usd"] = round(float(openai_cost_usd), 6)

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

def extract_with_openai(extraction_10k: Dict[str, Any], extraction_10q: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], float]:
    """Use OpenAI to holistically extract and normalize equity class data from full 10-K and 10-Q extraction JSONs.
    Pass both documents to the LLM (read 10-K first, then 10-Q) and let it reconcile using more up-to-date information.
    Returns (equity_classes, estimated_cost_usd).
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

    # Pricing per 1M tokens (USD). Override via env OPENAI_PRICE_IN_<MODEL>, OPENAI_PRICE_OUT_<MODEL>
    def _pricing_for_model(model_name: str) -> Tuple[float, float]:
        key = model_name.lower()
        in_env = os.getenv(f"OPENAI_PRICE_IN_{model_name.replace('-', '_').upper()}")
        out_env = os.getenv(f"OPENAI_PRICE_OUT_{model_name.replace('-', '_').upper()}")
        if in_env and out_env:
            try:
                return float(in_env), float(out_env)
            except ValueError:
                pass
        defaults = {
            'gpt-4o': (5.0, 15.0),
            'gpt-4o-mini': (0.15, 0.60),
            'gpt-4.1': (5.0, 15.0),
            'gpt-4.1-mini': (1.0, 5.0),
        }
        return defaults.get(key, (5.0, 15.0))

    def _estimate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        in_rate, out_rate = _pricing_for_model(model_name)
        return (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate

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
        "ticker_symbol (string), class_name (string), shares_outstanding (string with thousands separators or null), conversion_weight (float), "
        "voting_weight (float), voting_rights (string), conversion_rights (string), dividend_rights (string), "
        "other_rights (string), par_value (string), authorized_shares (string with thousands separators or null).\n\n"
        "Important requirements (LLM is the single source of truth; avoid regex and do not rely on external logic): \n"
        "- Include ALL share classes described anywhere in the provided text, including authorized but unissued classes.\n"
        "- Include ALL share classes described anywhere in the provided text, including preferred shares.\n"
        "- Do NOT invent ticker symbols. Only the cover page Section 12(b) determines tradeable tickers. If a class is not listed in 12(b), set ticker_symbol = null. If the 12(b) section shows 'None', set all ticker_symbol = null.\n"
        "- Use the most up-to-date numbers when 10-K and 10-Q differ. Generally, the 10-Q (being later) supersedes the prior 10-K for shares outstanding and similar figures as of its reporting date.\n"
        "- UNIT SCALING AND MULTIPLIERS: Carefully detect and apply table/unit qualifiers like 'In thousands', 'In millions', 'In billions', or phrases such as 'in thousands of shares'.\n"
        "  • If a figure is labeled 'In thousands', multiply by 1,000. If 'In millions', multiply by 1,000,000. If 'In billions', multiply by 1,000,000,000.\n"
        "  • Apply scaling to share count fields only (authorized_shares and shares_outstanding). Do NOT scale par_value.\n"
        "  • Look for unit qualifiers in nearby headings, column headers, footnotes, or captions of the exact table/section providing the number.\n"
        "  • Example: If a table states 'In thousands' and shows authorized 1,900,000, then authorized_shares = 1,900,000,000 (1.9 billion), not 1,900,000.\n"
        "  • If conflicting or ambiguous, prefer the most specific qualifier closest to the number; if still unclear, choose the interpretation consistent with other disclosed figures (e.g., shares outstanding, issuance activity).\n"
        "- FORMATTING: After applying any necessary scaling, format shares_outstanding and authorized_shares as strings using U.S. thousands separators (e.g., '549,964,000' or '1,900,000,000'). If a value is unknown or not disclosed, use null.\n"
        "Normalization: set the weakest class = 1.0 for voting_weight and conversion_weight and scale others accordingly.\n"
        "Return ONLY a valid JSON array with no extra text. Ensure shares_outstanding and authorized_shares are strings with commas when present, not numbers."
    )

    def _call_llm(messages, model) -> Tuple[str, int, int]:
        if client:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "3000"))
            )
            content = resp.choices[0].message.content.strip()
            usage = getattr(resp, 'usage', None)
            prompt_tokens = getattr(usage, 'prompt_tokens', None) or getattr(usage, 'input_tokens', 0) or 0
            completion_tokens = getattr(usage, 'completion_tokens', None) or getattr(usage, 'output_tokens', 0) or 0
            return content, int(prompt_tokens), int(completion_tokens)
        else:
            import openai as _openai
            resp = _openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "3000"))
            )
            content = resp['choices'][0]['message']['content'].strip()
            usage = resp.get('usage', {})
            prompt_tokens = int(usage.get('prompt_tokens', 0))
            completion_tokens = int(usage.get('completion_tokens', 0))
            return content, prompt_tokens, completion_tokens

    model = os.getenv('LLM_MODEL', 'gpt-4o')
    system_msg = {"role": "system", "content": "Extract and normalize share class data by holistically reading 10-K then 10-Q JSONs. Return only a JSON array. Format share counts as strings with thousands separators."}
    user_msg = {"role": "user", "content": prompt}

    total_cost_usd = 0.0

    try:
        response_text, p_tokens, c_tokens = _call_llm([system_msg, user_msg], model)
        total_cost_usd += _estimate_cost_usd(model, p_tokens, c_tokens)
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
            response_text, p_tokens, c_tokens = _call_llm([system_msg, {"role": "user", "content": prompt_small}], fallback_model)
            total_cost_usd += _estimate_cost_usd(fallback_model, p_tokens, c_tokens)
        except Exception as e2:
            print(f"[ERROR] OpenAI API error: {e2}")
            traceback.print_exc()
            return [], total_cost_usd

    # Clean the response text
    response_text = clean_json_response(response_text)

    # Parse JSON
    try:
        equity_classes = json.loads(response_text)
        if not isinstance(equity_classes, list):
            print("[ERROR] OpenAI response is not a JSON array")
            print(f"Response: {response_text}")
            return [], total_cost_usd
        return [c for c in equity_classes if isinstance(c, dict) and 'class_name' in c], total_cost_usd
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse OpenAI response as JSON: {e}")
        print(f"Response was: {response_text}")
        return [], total_cost_usd


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
    equity_classes, cost_usd = extract_with_openai(extraction_10k or {}, extraction_10q)

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

        shares_str = str(shares)

        print(f"  {i}. {class_name} ({ticker})")
        print(f"     Shares: {shares_str}")
        print(f"     Voting weight: {voting_weight}")
        print(f"     Conversion weight: {conversion_weight}")

    # Save results with cost and human-readable timestamp
    save_results(equity_classes, output_file, cik, openai_cost_usd=cost_usd)

    print("[SUCCESS] Equity class data extraction complete!")
    if cost_usd is not None:
        print(f"[INFO] Estimated OpenAI cost: ${cost_usd:.6f} USD")
    print(f"[INFO] Raw normalized data ready for analysis in {output_file}")


if __name__ == "__main__":
    main()