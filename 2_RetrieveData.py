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
from datetime import datetime
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
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {file_path}: {e}")
        sys.exit(1)

def extract_with_openai(extraction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Use OpenAI to extract and normalize equity class data"""
    
    # Initialize OpenAI client
    try:
        import openai
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment variables")
            print("Please check your .env file")
            sys.exit(1)
        
        # Initialize client based on openai library version
        if hasattr(openai, 'OpenAI'):
            client = openai.OpenAI(api_key=api_key)
        else:
            openai.api_key = api_key
            client = None
            
    except ImportError:
        print("❌ OpenAI library not installed. Install with: pip install openai")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        sys.exit(1)
    
    # Prepare the context for OpenAI - include all relevant sections
    context = {
        "cover_page": extraction_data.get("cover_page", {}),
        "stockholders_equity_notes": extraction_data.get("stockholders_equity_notes", {}),
        "exhibit_4_securities": extraction_data.get("exhibit_4_securities", {}),
        "charter_bylaws": extraction_data.get("charter_bylaws", {}),
        "market_equity": extraction_data.get("market_equity", {}),
        "cik": extraction_data.get("cik", ""),
        "filing_path": extraction_data.get("filing_path", "")
    }
    
    prompt = f"""
Analyze this 10-K equity extraction data and extract normalized share class information.

EQUITY EXTRACTION DATA:
{json.dumps(context, indent=2)}

TASK: Extract raw data for each share class and return a JSON array. Each element represents one class of shares.

For each share class, extract:
1. ticker_symbol: The trading symbol (e.g., "FLWS")
2. class_name: The class designation (e.g., "Class A Common Stock", "Class B Common Stock") 
3. shares_outstanding: Number of shares outstanding (as integer, no commas)
4. conversion_weight: Conversion ratio relative to the WEAKEST class (weakest class = 1.0)
5. voting_weight: Voting power relative to the WEAKEST voting class (weakest = 1.0)
6. voting_rights: Description of voting rights from the filing
7. conversion_rights: Description of conversion rights from the filing
8. dividend_rights: Description of dividend rights from the filing
9. other_rights: Any other special rights, preferences, or restrictions
10. par_value: Par value per share if available
11. authorized_shares: Number of authorized shares if available

CRITICAL NORMALIZATION RULES:
- For conversion_weight: Find the class with the weakest conversion rights and set it to 1.0. Scale others accordingly.
- For voting_weight: Find the class with the fewest votes per share and set it to 1.0. Scale others accordingly.
- Example: If Class A has 1 vote/share and Class B has 10 votes/share, then Class A = 1.0, Class B = 10.0
- Example: If Class B converts to Class A 1:1, and Class A has no conversion, then both = 1.0
- If only one class exists, all weights = 1.0

EXTRACT FROM ALL SECTIONS:
- Use cover_page for basic share counts and ticker symbols
- Use stockholders_equity_notes for detailed voting/conversion rights
- Use exhibit_4_securities for additional rights descriptions
- Combine information from multiple sections for complete picture

Return ONLY a valid JSON array with no markdown formatting or additional text:
[
  {{
    "ticker_symbol": "string",
    "class_name": "string", 
    "shares_outstanding": integer,
    "conversion_weight": float,
    "voting_weight": float,
    "voting_rights": "string",
    "conversion_rights": "string", 
    "dividend_rights": "string",
    "other_rights": "string",
    "par_value": "string",
    "authorized_shares": integer
  }}
]
"""

    try:
        model = os.getenv('LLM_MODEL', 'gpt-4o')
        
        if client:  # New OpenAI library
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a financial data analyst specializing in equity structures. Extract and normalize share class data from SEC filings. Return only valid JSON arrays with no additional formatting."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=3000
            )
            response_text = response.choices[0].message.content.strip()
        else:  # Old OpenAI library
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a financial data analyst specializing in equity structures. Extract and normalize share class data from SEC filings. Return only valid JSON arrays with no additional formatting."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=3000
            )
            response_text = response['choices'][0]['message']['content'].strip()
        
        # Clean the response text
        response_text = clean_json_response(response_text)
        
        # Try to parse the JSON response
        try:
            equity_classes = json.loads(response_text)
            if not isinstance(equity_classes, list):
                print("❌ OpenAI response is not a JSON array")
                print(f"Response: {response_text}")
                return []
            
            # Validate each class has required fields
            validated_classes = []
            for cls in equity_classes:
                if isinstance(cls, dict) and 'class_name' in cls:
                    validated_classes.append(cls)
                    
            return validated_classes
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse OpenAI response as JSON: {e}")
            print(f"Response was: {response_text}")
            return []
            
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
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
        "extracted_at": datetime.utcnow().isoformat() + "Z",
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
        print(f"💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

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
        output_file = f"staging/cik_{cik}_equity_classes.json"
    
    print(f"🔍 Processing equity extraction data...")
    print(f"  📄 Input: {input_file}")
    print(f"  💾 Output: {output_file}")
    print(f"  🆔 CIK: {cik}")
    
    # Load the extraction data
    extraction_data = load_extraction_data(input_file)
    
    # Extract equity class data using OpenAI
    print("🤖 Analyzing equity data with OpenAI...")
    equity_classes = extract_with_openai(extraction_data)
    
    if not equity_classes:
        print("❌ No equity classes extracted")
        sys.exit(1)
    
    print(f"✅ Extracted {len(equity_classes)} equity classes:")
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
    
    print("✅ Equity class data extraction complete!")
    print(f"📊 Raw normalized data ready for analysis in {output_file}")

if __name__ == "__main__":
    main()
