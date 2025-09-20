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
            │   └── extract_market_equity_info()        # Extracts Item 5 - Market for Common Equity
            └── save_json_file() [from shared_utils]    # Saves structured equity data to staging/cik_{cik}_equity_extraction.json
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

## 2️⃣ **`2_RetrieveData.py`**

```
Entry Point: if __name__ == "__main__":
    └── main()                          # Orchestrates OpenAI-powered equity class normalization
        ├── argparse setup              # Handles --cik or --file input arguments
        ├── load_extraction_data()      # Loads equity extraction JSON from staging/
        ├── extract_with_openai()       # Uses OpenAI to normalize equity class data
        │   ├── OpenAI client initialization  # Loads API key from .env file
        │   ├── Comprehensive prompt engineering  # Sends all extraction sections to AI
        │   ├── AI analysis of equity structure   # Normalizes voting/conversion weights
        │   ├── clean_json_response()    # Cleans and validates AI JSON response
        │   └── Field validation        # Ensures required fields are present
        └── save_results()              # Saves normalized equity classes to staging/cik_{cik}_equity_classes.json
            └── Metadata tracking       # Includes extraction timestamp, model used, normalization notes
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
2. **Step 1.5**: `1.5_Download10K.py --cik CIK_NUMBER` → Downloads latest 10-K filing for individual companies
3. **Step 1.6**: `1.6_Extract10K.py --cik CIK_NUMBER` → Extracts equity class details from downloaded 10-K filings
4. **Step 2**: `2_RetrieveData.py --cik CIK_NUMBER` → Uses OpenAI to normalize equity class data into structured arrays
5. **Step 1.75**: `1.75_missing_company_investigator.py` → Investigates companies missing CIKs

## 🎯 **Key Function Patterns:**

- **Data Loading**: Always starts with `load_json_file()`
- **SEC API Calls**: Use `make_request()` with proper headers
- **AI Processing**: Use `query_openai()` with error handling
- **Data Saving**: Always ends with `save_json_file()`
- **Batch Processing**: Loop through companies with progress reporting

## 📊 **Pipeline Summary:**

**Step 1** (CSV → JSON): Converts CSV to structured JSON with company data  
**Step 1.5** (10-K Download): Downloads latest 10-K filing for individual companies, saves to organized folders  
**Step 1.6** (Equity Extraction): Extracts comprehensive equity class details from 10-K filings into structured JSON  
**Step 2** (OpenAI Normalization): Uses OpenAI to normalize equity extraction data into clean share class arrays with voting/conversion weights  
**Step 1.75** (Investigate Missing): Researches companies without CIKs using AI and ticker variants

Each step builds on the previous, with shared utilities handling common operations like SEC API calls, file I/O, and AI interactions. The new Step 2 provides a direct OpenAI-powered normalization path from raw 10-K extractions to clean equity class arrays, bypassing the legacy ticker mapping approach.
