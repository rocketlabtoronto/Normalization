# 2_RetrieveData: Rules, Heuristics, and Logic

Purpose

- Normalize share class information from 10‑K extraction JSON (Step 1.6 output) using an AI‑first, LLM‑only holistic approach. The LLM interprets the entire extraction JSON, and writes a final equity_classes JSON per CIK.

Inputs and Outputs

- Input: staging/cik\_{CIK}\_equity_extraction.json (from Step 1.6)
- Output: fileoutput/equity_classes/cik\_{CIK}\_equity_classes.json
- CLI:
  - python 2_RetrieveData.py --cik 0000123456
  - python 2_RetrieveData.py --file staging/cik_0000123456_equity_extraction.json [--output path]

Environment

- OPENAI_API_KEY: required
- LLM_MODEL: default gpt-4o
- OPENAI_MAX_TOKENS: default 3000
- .env is supported via python-dotenv

High‑Level Flow (LLM‑only)

1. Load the full Step 1.6 extraction JSON into memory
2. Send the entire object to the LLM with explicit instructions/rules
3. The LLM returns ONLY a JSON array of classes with required fields
4. Save the final result with metadata

AI Prompt: Core Requirements

- Include ALL share classes described in the text, including authorized but unissued classes (e.g., Preferred Stock)
- If the text explicitly says a class is not issued (e.g., "no preferred stock issued"), include the class with shares_outstanding = 0, but DO NOT zero out a class that already has shares outstanding merely because an amendment authorized additional unissued shares
- Extract authorized_shares and par_value per class from Capital Stock, Stockholders’ Equity notes, Charter/Bylaws, Exhibit 4, and anywhere else in the JSON
- Do NOT invent ticker symbols. Only the cover page Section 12(b) determines tradeable tickers; if not listed there, set ticker_symbol = null. If 12(b) shows "None", set all ticker_symbol = null
- If a field is not stated, use null; keep rights fields as empty strings
- Normalize voting_weight and conversion_weight to the weakest class = 1.0

Context Handling

- The model receives the full extraction JSON (holistic). Extremely large files may exceed model context limits. If that occurs, increase OPENAI_MAX_TOKENS and consider reducing upstream extraction verbosity.

Output Schema (per class)

- ticker_symbol: string or null (strictly from cover page 12(b))
- class_name: string
- shares_outstanding: integer or null
- conversion_weight: float or null (weakest = 1.0)
- voting_weight: float or null (weakest = 1.0)
- voting_rights: string (may be empty)
- conversion_rights: string (may be empty)
- dividend_rights: string (may be empty)
- other_rights: string (may be empty)
- par_value: string (e.g., "$0.0001") or null
- authorized_shares: integer or null

Key Rules Summary

- Authorized but unissued classes must be included (e.g., Preferred Stock)
- Do not zero out existing outstanding shares when an amendment authorizes additional unissued shares; only set to 0 when text explicitly says none issued/outstanding
- Tickers are assigned strictly from cover page Section 12(b). If none listed, all tickers are null
- LLM‑only extraction; no deterministic post‑processing is applied

Usage Examples

- python 2_RetrieveData.py --cik 0001799448
- python 2_RetrieveData.py --file staging/cik_0001799448_equity_extraction.json --output fileoutput/equity_classes/cik_0001799448_equity_classes.json

Limitations and Notes

- Cover layouts and disclosure phrasing vary; the LLM interprets holistically but may still need larger context window for very large filings
- authorized_shares and par_value may remain null if not reasonably inferable from the provided JSON
- Rights fields depend on disclosure quality; the LLM may return brief summaries or empty strings
- The script passes only the Step 1.6 extraction JSON (no additional crawling)

Contact/Debugging

- Adjust LLM_MODEL/OPENAI_MAX_TOKENS as needed
- Review console output for errors and the saved JSON in fileoutput/equity_classes/
