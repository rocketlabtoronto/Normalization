# Function Call Flow Analysis

## 1️⃣ **`1_dual_class_csv_to_json_converter.py`**

```
Entry Point: if __name__ == "__main__":
    └── main()                          # Orchestrates CSV to JSON conversion
        └── load_csv_data()            # Reads dual-class companies from CSV file
        └── process_companies()        # Normalizes and validates company data
        └── save_json_file() [from shared_utils]  # Saves structured JSON output
```

## 1️⃣🔎 **`1.5_Download10K.py`**

```
Entry Point: if __name__ == "__main__":
    └── args[1] == "--cik":
        └── analyze_cik(cik, symbol, exchange)  # Downloads latest 10-K filing for the company
            ├── fetch_sec_submissions() [from shared_utils]  # Downloads company's filing history from SEC
            ├── extract_latest_10k()            # Finds the most recent 10-K or 10-K/A filing
            ├── build_filing_archives_url()     # Constructs SEC Archives URL for document
            ├── fetch_filing_text()             # Downloads actual 10-K document content and saves to sec_filings/10K/ folder
            │   └── make_request() [from shared_utils]
            └── save_json_file() [from shared_utils]  # Saves download report to staging/cik_{cik}_10k_download.json
```

## 1️⃣📥 **`1.51_Download10Q.py`**

```
Entry Point: if __name__ == "__main__":
    └── args[1] == "--cik":
        └── analyze_cik(cik, symbol, exchange)  # Downloads latest 10-Q filing for the company (skips if already saved)
            ├── fetch_sec_submissions() [from shared_utils]
            ├── extract_latest_10q()            # Finds the most recent original 10-Q
            ├── build_filing_archives_url()
            ├── fetch_filing_text()             # Saves to sec_filings/10Q/
            │   └── make_request() [from shared_utils]
            └── save_json_file() [from shared_utils]  # staging/cik_{cik}_10q_download.json
```

## 1️⃣🔍 **`1.6_Extract10K.py`**

```
Entry Point: if __name__ == "__main__":
    └── args[1] == "--cik":
        └── analyze_cik_equity(cik, symbol, exchange)  # Extracts equity class details from downloaded 10-K filing
            ├── find_10k_filing(cik)            # Locates downloaded 10-K file in sec_filings/10K/ folder
            ├── extract_equity_data(cik, filing_path)  # Main extraction coordinator
            │   ├── extract_cover_page_data()    # Extracts registered securities and outstanding shares from cover page
            │   ├── extract_stockholders_equity_notes()  # Parses Part II, Item 8 - Stockholders' Equity notes
            │   ├── extract_exhibit_4_description()     # Extracts Exhibit 4 - Description of Registrant's Securities
            │   ├── extract_charter_bylaws_info()       # Parses Exhibits 3.1/3.2 - Charter & Bylaws
            │   ├── extract_market_equity_info()        # Extracts Item 5 - Market for Common Equity
            │   └── extract_capital_stock()             # Captures Capital Stock sections and in-body descriptions
            └── save_json_file() [from shared_utils]    # staging/cik_{cik}_equity_extraction.json
```

## 1️⃣📘 **`1.61_Extract10Q.py`**

```
Entry Point: if __name__ == "__main__":
    └── args[1] == "--cik":
        └── analyze_cik_equity_10q(cik, symbol, exchange)  # Extracts equity class details from downloaded 10-Q filing
            ├── find_latest_10q_html(cik)       # Uses 10-Q download report to locate saved HTML in sec_filings/10Q/
            ├── extract_equity_data_10q(cik, filing_path)
            │   ├── extract_cover_page_data()         # Same cover logic as 10-K
            │   ├── extract_stockholders_equity_notes()
            │   ├── extract_market_equity_info()
            │   ├── extract_capital_stock()
            │   └── extract_in_body_security_descriptions()
            └── save_json_file() [from shared_utils]   # staging/cik_{cik}_equity_extraction_10q.json
```

## 1️⃣🔎 **`1.75_missing_company_investigator.py`**

```
Entry Point: if __name__ == "__main__":
    └── process_no_cik_file()              # Investigates companies missing CIK numbers (batch mode only)
        ├── load_json_file() [from shared_utils]  # Loads companies without CIKs from staging/1.75_dual_class_output_nocik.json
        ├── fetch_sec_ticker_map() [from shared_utils]  # Gets SEC ticker-to-CIK mapping
        └── FOR EACH COMPANY:
            ├── generate_ticker_variants() [from shared_utils]  # Creates ticker variations for matching
            ├── IF ai_check enabled:
            │   └── investigate_company_with_ai()   # Uses AI to research what happened to missing company
            │       └── openai_investigate_company()  # Asks AI about delisting/acquisition/bankruptcy
            │           └── query_openai() [from shared_utils]
            └── save_json_file() [from shared_utils]   # Saves investigation results to staging/1.75_dual_class_output_investigated.json
```

## 2️⃣ **`2_RetrieveData.py`** (Holistic 10-K + 10-Q)

```
Entry Point: if __name__ == "__main__":
    └── main()
        ├── argparse setup                     # --cik or --file
        ├── load_extraction_data()             # Loads 10-K and, if present, 10-Q extraction JSONs from staging/
        ├── extract_with_openai()              # LLM-only reconciliation of all sections
        │   ├── OpenAI client initialization   # Reads .env (API key, model, limits)
        │   ├── Prompt enforces policies:      # Strict rules baked into instructions
        │   │   ├── Tickers only from cover page Section 12(b); set null if none
        │   │   ├── Include authorized-but-unissued classes (e.g., Preferred) with ticker null
        │   │   ├── Do not zero out issued classes due to authorization-only amendments
        │   │   └── Prefer most recent numbers; 10-Q supersedes 10-K when applicable
        │   ├── Context compaction              # Trims long sections to fit token limits; retries with tighter caps
        │   ├── Fallback model option           # Switches if initial request hits TPM/context limits
        │   ├── clean_json_response()           # Validates and sanitizes AI JSON
        │   └── Field validation                # Ensures required fields present
        └── save_results()                      # Writes to fileoutput/equity_classes/cik_{cik}_equity_classes.json
            └── Metadata tracking               # Extraction timestamp, model used, notes
```

## 🛠️ **`shared_utils.py` Functions Used Throughout:**

```
Core Utilities:
├── make_request() - Makes HTTP requests with retries, proper headers, and error handling
├── load_json_file() - Safely loads and parses JSON files with error handling
├── save_json_file() - Saves Python objects to JSON files with proper formatting
├── setup_openai() - Initializes OpenAI client with API key and configuration
├── query_openai() - Makes OpenAI API calls with retry logic and token management
├── fetch_sec_submissions() - Downloads company filing history from SEC EDGAR API
├── fetch_sec_ticker_map() - Retrieves and caches SEC's official ticker-to-CIK mapping
└── generate_ticker_variants() - Creates ticker variations (suffixes, prefixes) for matching
```

## 🔍 **How to Trace Function Calls:**

### **Method 1: Static Analysis (Reading Code)**

1. Start at `if __name__ == "__main__":`
2. Follow the main function calls
3. Look for function definitions and their calls
4. Track imports from `shared_utils`

### **Method 2: Add Debug Logging**

```python
def my_function():
    print(f"🔵 ENTERING: {__name__}.my_function()")
    # ... function code ...
    print(f"🔴 EXITING: {__name__}.my_function()")
```

### **Method 3: Use Python's `trace` Module**

```bash
python -m trace --trace script.py
```

### **Method 4: Use `pdb` Debugger**

```python
import pdb; pdb.set_trace()
```

### **Method 5: IDE Debugging**

- Set breakpoints in VS Code
- Use "Step Into" to follow function calls
- Use call stack to see execution path

## 📋 **Typical Execution Order:**

1. Step 1: `1_dual_class_csv_to_json_converter.py` → Creates `staging/1_dual_class_output.json`
2. Step 1.5: `1.5_Download10K.py --cik CIK_NUMBER` → Downloads latest 10-K
3. Step 1.6: `1.6_Extract10K.py --cik CIK_NUMBER` → Extracts 10-K equity data → `staging/cik_{cik}_equity_extraction.json`
4. Step 1.51 (optional but recommended): `1.51_Download10Q.py --cik CIK_NUMBER` → Downloads latest 10-Q
5. Step 1.61 (optional but recommended): `1.61_Extract10Q.py --cik CIK_NUMBER` → Extracts 10-Q equity data → `staging/cik_{cik}_equity_extraction_10q.json`
6. Step 2: `2_RetrieveData.py --cik CIK_NUMBER` → Holistic normalization (10-K + 10-Q) → `fileoutput/equity_classes/cik_{cik}_equity_classes.json`
7. Step 1.75 (as needed): `1.75_missing_company_investigator.py` → Investigates companies missing CIKs

## 🎯 **Key Function Patterns:**

- Data Loading: Always starts with `load_json_file()`
- SEC API Calls: Use `make_request()` with proper headers
- AI Processing: Use `query_openai()` with error handling
- Data Saving: Always ends with `save_json_file()`
- Batch Processing: Loop through companies with progress reporting

## 📊 **Pipeline Summary:**

- Step 1 (CSV → JSON): Converts CSV to structured JSON with company data
- Step 1.5 (10-K Download): Downloads latest 10-K filing and saves to organized folders
- Step 1.6 (10-K Extraction): Extracts comprehensive equity class details from 10-K into structured JSON
- Step 1.51 (10-Q Download): Downloads latest 10-Q filing if not already present
- Step 1.61 (10-Q Extraction): Extracts analogous equity details from 10-Q into structured JSON
- Step 2 (OpenAI Normalization): LLM-only holistic synthesis of 10-K + 10-Q; enforces cover-page ticker rule, includes unissued classes, and prefers most recent data; outputs normalized share classes
- Step 1.75 (Investigate Missing): Researches companies without CIKs using AI and ticker variants

Each step builds on the previous, with shared utilities handling common operations like SEC API calls, file I/O, and AI interactions. The new holistic Step 2 reads both 10-K and 10-Q extraction JSONs to produce a single reconciled equity class output.
