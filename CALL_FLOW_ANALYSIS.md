# Function Call Flow Analysis

## 1️⃣ **`1_dual_class_csv_to_json_converter.py`**

```
Entry Point: if __name__ == "__main__":
    └── main()                          # Orchestrates CSV to JSON conversion
        └── load_csv_data()            # Reads dual-class companies from CSV file
        └── process_companies()        # Normalizes and validates company data
        └── save_json_file() [from shared_utils]  # Saves structured JSON output
```

## 1️⃣🔎 **`1.75_missing_company_investigator.py`**

```
Entry Point: if __name__ == "__main__":
    ├── args[1] == "--cik":
    │   └── analyze_cik(cik, symbol, exchange)  # Analyzes single company's SEC filings for corporate events
    │       ├── fetch_sec_submissions() [from shared_utils]  # Downloads company's filing history from SEC
    │       ├── extract_recent_filings()        # Filters to recent 8-K, Form 25, Form 15 filings
    │       ├── FOR EACH FILING:
    │       │   ├── build_filing_archives_url()  # Constructs SEC Archives URL for document
    │       │   ├── fetch_filing_text()         # Downloads actual filing document content
    │       │   │   └── make_request() [from shared_utils]
    │       │   └── parse_8k_items()           # Extracts specific event items (1.03, 2.01, 3.01) from 8-K forms
    │       ├── discover_nasdaq_alerts() / discover_nyse_notices()  # Checks exchange-specific delisting notices
    │       └── save_json_file() [from shared_utils]  # Saves investigation report
    │
    └── DEFAULT (batch mode):
        └── process_no_cik_file()              # Investigates companies missing CIK numbers
            ├── load_json_file() [from shared_utils]  # Loads companies without CIKs
            ├── fetch_sec_ticker_map() [from shared_utils]  # Gets SEC ticker-to-CIK mapping
            └── FOR EACH COMPANY:
                ├── generate_ticker_variants() [from shared_utils]  # Creates ticker variations for matching
                ├── IF ai_check enabled:
                │   └── investigate_company_with_ai()   # Uses AI to research what happened to missing company
                │       └── openai_investigate_company()  # Asks AI about delisting/acquisition/bankruptcy
                │           └── query_openai() [from shared_utils]
                └── analyze_cik() [if CIK found]       # Runs filing analysis if CIK discovered
            └── save_json_file() [from shared_utils]   # Saves investigation results
```

## 2️⃣ **`2_sec_filing_ticker_mapper.py`**

```
Entry Point: if __name__ == "__main__":
    └── _cli()                          # Parses command line arguments for batch or single-company mode
        ├── args.batch = True:
        │   └── run_batch_processing(test_mode)  # Processes all companies from step 1 output
        │       ├── load_json_file() [from shared_utils]  # Loads companies with CIKs from step 1
        │       ├── MapTickerToShareClass() [class initialization]  # Creates SEC filing parser instance
        │       └── FOR EACH COMPANY:
        │           └── mapper.get_ticker_to_class_mapping(cik, name, ticker)  # Extracts ticker-to-share-class mappings
        │               ├── get_latest_filing(cik)          # Downloads most recent 10-K/10-Q filing metadata
        │               │   └── make_request() [from shared_utils]
        │               ├── get_filing_documents(filing_info)  # Lists all documents in the filing
        │               │   └── make_request() [from shared_utils]
        │               ├── download_filing_content(filing_info, doc_name, docs)  # Downloads primary filing document
        │               │   └── make_request() [from shared_utils]
        │               └── parse_cover_page_table(content)  # Extracts Section 12(b) securities table
        │                   ├── _collect_context_for_llm()   # Gathers context about share classes and counts
        │                   ├── _build_llm_prompt()         # Creates structured prompt for AI parsing
        │                   ├── _call_llm()                 # Uses AI to parse complex cover page tables
        │                   │   └── query_openai() [from shared_utils]
        │                   └── _fallback_parse_cover_table() [if LLM fails]  # Deterministic HTML parsing backup
        │       ├── _augment_with_untraded_classes() [for each result]  # Adds non-traded share classes as placeholders
        │       └── save_json_file() [from shared_utils]    # Saves ticker mappings to step 2 output
        │
        └── args.cik = "123456":
            └── mapper.get_ticker_to_class_mapping(cik)     # Single company analysis mode
                └── [same flow as above for single company]
```

## 3️⃣ **`3_ai_powered_financial_analyzer.py`**

```
Entry Point: if __name__ == "__main__":
    └── main function logic              # Orchestrates AI-powered economic weight analysis
        ├── load_json_file() [from shared_utils - loads step 1 output]  # Loads companies with CIKs
        ├── fetch_sec_ticker_map() [from shared_utils]  # Gets SEC ticker-to-CIK mapping for lookups
        └── FOR EACH COMPANY:
            ├── lookup_cik_by_name_or_ticker()      # Attempts to find CIK using company name or ticker variants
            ├── IF CIK found:
            │   └── query_ai_for_economic_weights()  # Uses AI to determine economic weights of share classes
            │       ├── setup_openai() [from shared_utils]  # Initializes OpenAI client
            │       ├── query_openai() [from shared_utils]   # Asks AI about voting rights and economic distribution
            │       ├── _parse_ai_json_to_weights()         # Converts AI response to structured weight data
            │       └── _deduplicate_classes()              # Removes duplicate share class entries
            └── IF NO CIK:
                └── investigate_no_cik_with_ai()    # Uses AI to investigate why company has no CIK
                    └── query_openai() [from shared_utils]  # Asks AI about delisting/private status
        ├── save_json_file() [main results to results/]    # Saves economic weights analysis
        └── save_json_file() [no-CIK companies to results/]  # Saves companies without CIKs for step 1.75
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

1. **Step 1**: `1_dual_class_csv_to_json_converter.py` → Creates `staging/1_dual_class_output.json`
2. **Step 1.75**: `1.75_missing_company_investigator.py` → Creates `staging/1.75_dual_class_output_nocik.json`
3. **Step 2**: `2_sec_filing_ticker_mapper.py --batch` → Creates `staging/2_step2_ticker_mappings.json`
4. **Step 3**: `3_ai_powered_financial_analyzer.py` → Creates `results/3_dual_class_economic_weights.json`

## 🎯 **Key Function Patterns:**

- **Data Loading**: Always starts with `load_json_file()`
- **SEC API Calls**: Use `make_request()` with proper headers
- **AI Processing**: Use `query_openai()` with error handling
- **Data Saving**: Always ends with `save_json_file()`
- **Batch Processing**: Loop through companies with progress reporting

## 📊 **Pipeline Summary:**

**Step 1** (CSV → JSON): Converts CSV to structured JSON with company data
**Step 1.75** (Investigate Missing): Researches companies without CIKs using AI and SEC filings  
**Step 2** (Ticker Mapping): Extracts ticker-to-share-class mappings from SEC filings
**Step 3** (Economic Weights): Uses AI to determine economic weight distribution across share classes

Each step builds on the previous, with shared utilities handling common operations like SEC API calls, file I/O, and AI interactions.
