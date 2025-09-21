#!/usr/bin/env python3
"""
2_RetrieveData.py - Extract and normalize equity class data using OpenAI

This script reads the equity extraction JSON files from staging/ and uses OpenAI 
to extract raw data for each share class, normalizing voting weights and conversion 
ratios relative to the weakest class (set to 1.0).

Usage:
    python 2_RetrieveData.py --cik 0000123456
    python 2_RetrieveData.py --file staging/cik_0001084869_equity_extraction.json
"""

import json
import argparse
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, UTC
import re

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_extraction_data(file_path: str) -> Dict[str, Any]:
    """Load the equity extraction JSON data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {file_path}: {e}")
        sys.exit(1)


# --------------------------- New helpers for chunk-based extraction ---------------------------

def flatten_sections(extraction_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Flatten all section groups into a list of { group, heading, content } chunks."""
    chunks: List[Dict[str, str]] = []
    for group in [
        "cover_page",
        "stockholders_equity_notes",
        "exhibit_4_securities",
        "charter_bylaws",
        "market_equity",
    ]:
        section_group = extraction_data.get(group) or {}
        for sec in section_group.get("sections", []) or []:
            heading = (sec.get("heading") or "").strip()
            content = (sec.get("content") or "").strip()
            if content:
                chunks.append({
                    "group": group,
                    "heading": heading,
                    "content": content,
                })
    return chunks


def filter_relevant_chunks(chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep chunks most relevant to equity classes and rights."""
    if not chunks:
        return []
    # Always keep cover page and section 12(b)
    keep: List[Dict[str, str]] = []
    keywords = [
        "stockholders", "shareholders", "shareowners", "capital stock", "share capital",
        "equity", "earnings per share", "exhibit 4", "description of registrant",
        "item 5", "market for registrant", "securities registered", "section 12(b)",
        "charter", "bylaws",
    ]
    for ch in chunks:
        h = (ch.get("heading") or "").lower()
        g = ch.get("group") or ""
        text = f"{h} {g}"
        if g == "cover_page":
            keep.append(ch)
            continue
        if any(k in text for k in keywords):
            keep.append(ch)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[Dict[str, str]] = []
    for ch in keep:
        key = (ch.get("group"), ch.get("heading"))
        if key not in seen:
            uniq.append(ch)
            seen.add(key)
    return uniq if uniq else chunks


def cap_total_chars(chunks: List[Dict[str, str]], max_total: int = 60000) -> List[Dict[str, str]]:
    """Cap total characters included in the prompt to avoid context overflow, truncating individual contents as needed."""
    total = 0
    out: List[Dict[str, str]] = []
    for ch in chunks:
        content = ch.get("content", "")
        remaining = max_total - total
        if remaining <= 0:
            break
        if len(content) <= remaining:
            out.append(ch)
            total += len(content)
        else:
            # Truncate content; note to model that it is truncated
            out.append({
                "group": ch.get("group", ""),
                "heading": ch.get("heading", ""),
                "content": content[:remaining]
            })
            total += remaining
            break
    return out


def extract_with_openai(extraction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Use OpenAI to extract and normalize equity class data from chunked text sections."""

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

    # Build chunked context from new 1.6 output structure
    all_chunks = flatten_sections(extraction_data)
    relevant_chunks = filter_relevant_chunks(all_chunks)
    prompt_chunks = cap_total_chars(relevant_chunks, max_total=int(os.getenv("OPENAI_MAX_CONTEXT_CHARS", "60000")))

    # Compose prompt
    prompt = (
        "You are a financial data analyst specializing in equity structures. "
        "You will be given a list of text chunks extracted from a 10-K filing. "
        "Each chunk has a heading and full content. Parse ALL provided chunks together to extract "
        "share class details and normalize voting/conversion weights.\n\n"
        f"FILING INFO: CIK={extraction_data.get('cik','')}, FILE={extraction_data.get('filing_path','')}\n\n"
        "TEXT CHUNKS (JSON Array of {group, heading, content}):\n"
        f"{json.dumps(prompt_chunks, ensure_ascii=False)}\n\n"
        "TASK: Return a JSON array, one object per share class, with fields: \n"
        "ticker_symbol (string), class_name (string), shares_outstanding (integer), conversion_weight (float), "
        "voting_weight (float), voting_rights (string), conversion_rights (string), dividend_rights (string), "
        "other_rights (string), par_value (string), authorized_shares (integer).\n\n"
        "Normalization: set the weakest class = 1.0 for voting_weight and conversion_weight and scale others accordingly.\n"
        "Derive all values from the text; if unknown, use null and leave description fields empty strings.\n"
        "Return ONLY a valid JSON array with no extra text."
    )

    try:
        model = os.getenv('LLM_MODEL', 'gpt-4o')
        system_msg = {
            "role": "system",
            "content": "Extract and normalize share class data from SEC 10-K text chunks. Return only a JSON array."
        }
        user_msg = {"role": "user", "content": prompt}

        if client:  # New OpenAI library
            response = client.chat.completions.create(
                model=model,
                messages=[system_msg, user_msg],
                temperature=0.1,
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "3000"))
            )
            response_text = response.choices[0].message.content.strip()
        else:  # Old OpenAI library
            response = openai.ChatCompletion.create(
                model=model,
                messages=[system_msg, user_msg],
                temperature=0.1,
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "3000"))
            )
            response_text = response['choices'][0]['message']['content'].strip()

        # Clean the response text
        response_text = clean_json_response(response_text)

        # Try to parse the JSON response
        try:
            equity_classes = json.loads(response_text)
            if not isinstance(equity_classes, list):
                print("[ERROR] OpenAI response is not a JSON array")
                print(f"Response: {response_text}")
                return []

            # Validate each class has required fields
            validated_classes = []
            for cls in equity_classes:
                if isinstance(cls, dict) and 'class_name' in cls:
                    validated_classes.append(cls)
            return validated_classes

        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse OpenAI response as JSON: {e}")
            print(f"Response was: {response_text}")
            return []

    except Exception as e:
        print(f"[ERROR] OpenAI API error: {e}")
        return []


def clean_json_response(response_text: str) -> str:
    """Clean OpenAI response to extract valid JSON"""
    # Remove markdown code blocks
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
        response_text = response_text[start:end+1]

    return response_text


def save_results(equity_classes: List[Dict[str, Any]], output_file: str, cik: str):
    """Save the extracted equity class data to JSON file"""

    result = {
        "extracted_at": datetime.now(UTC).isoformat(),
        "cik": cik,
        "total_classes": len(equity_classes),
        "extraction_method": "OpenAI " + os.getenv('LLM_MODEL', 'gpt-4o'),
        "normalization_note": "Voting and conversion weights normalized to weakest class = 1.0",
        "equity_classes": equity_classes
    }

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Results saved to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract normalized equity class data using OpenAI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cik", help="CIK number to process (looks for staging/cik_{cik}_equity_extraction.json)")
    group.add_argument("--file", help="Direct path to equity extraction JSON file")
    parser.add_argument("--output", help="Output file path (default: staging/cik_{cik}_equity_classes.json)")

    args = parser.parse_args()

    # Determine input file and CIK
    if args.file:
        input_file = args.file
        # Extract CIK from filename for output
        cik_match = re.search(r'cik_(\d+)', args.file)
        cik = cik_match.group(1) if cik_match else "unknown"
    else:
        cik = args.cik.zfill(10)  # Pad with zeros to 10 digits
        input_file = f"staging/cik_{cik}_equity_extraction.json"

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = f"fileoutput/equity_classes/cik_{cik}_equity_classes.json"

    print(f"[INFO] Processing equity extraction data...")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  CIK: {cik}")

    # Load the extraction data
    extraction_data = load_extraction_data(input_file)

    # Extract equity class data using OpenAI
    print("[INFO] Analyzing equity data with OpenAI...")
    equity_classes = extract_with_openai(extraction_data)

    if not equity_classes:
        print("[ERROR] No equity classes extracted")
        sys.exit(1)

    print(f"[SUCCESS] Extracted {len(equity_classes)} equity classes:")
    for i, equity_class in enumerate(equity_classes, 1):
        class_name = equity_class.get("class_name", "Unknown")
        shares = equity_class.get("shares_outstanding", "N/A")
        voting_weight = equity_class.get("voting_weight", "N/A")
        conversion_weight = equity_class.get("conversion_weight", "N/A")
        ticker = equity_class.get("ticker_symbol", "N/A")

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
