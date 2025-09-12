# LookThroughProfits Dual-Class Share Normalization Pipeline

This repository contains a Python-based pipeline for analyzing dual-class share structures of public companies. The pipeline processes a list of dual-class companies, enriches them with SEC data, maps ticker symbols to share classes, and calculates economic weights for each share class.

## Overview

The pipeline consists of several sequential steps that transform raw dual-class company data into normalized economic weight information suitable for investment analysis.

**Key Features:**

- Automated SEC filing retrieval and analysis
- AI-powered share class extraction from 10-K/10-Q filings
- Economic weight calculation across share classes
- Missing CIK resolution with AI investigation
- Comprehensive error handling and data quality checks

## Pipeline Steps

### Step 1: `1_dual_class_ingest.py` - Initial Data Ingestion

**Purpose:** Ingests raw dual-class company data from CSV and enriches it with SEC CIK numbers.

**Input:**

- `DualClassList_19Aug2025.csv` - Raw CSV file containing:
  - Company Name
  - Primary Ticker Symbol
  - Voting Structure (text descriptions of share classes)

**Process:**

1. Parses CSV file with multiple encoding support
2. Normalizes ticker symbols using SEC validation rules
3. Fetches SEC's master ticker-to-CIK mapping from `company_tickers.json`
4. Parses voting structure text to extract:
   - Class names (Class A, Class B, etc.)
   - Voting rights per share (1 vote, 10 votes, non-voting, etc.)
   - Associated ticker symbols
5. Maps company tickers to SEC CIK numbers
6. Determines primary ticker based on voting structure priority

**Output:**

- `dual_class_output.json` - Structured JSON containing:
  ```json
  {
    "as_of": "2025-08-19",
    "total_companies": 290,
    "companies_with_cik": 290,
    "companies": [
      {
        "company_name": "1-800-FLOWERS.COM",
        "primary_ticker": "FLWS",
        "classes": [
          {
            "class_name": "Class A: 1 vote per share",
            "ticker": "FLWS",
            "votes_per_share": 1.0
          }
        ],
        "cik": "0001084869"
      }
    ]
  }
  ```

**Key Functions:**

- `normalize_ticker()` - Cleans and validates ticker symbols
- `parse_voting_rights()` - Extracts numeric voting rights from text
- `parse_voting_structure()` - Converts text descriptions to structured data

---

### Step 1.75: `1.75_nocik.py` - Missing CIK Resolution

**Purpose:** Investigates companies that couldn't be mapped to SEC CIK numbers and determines why (delisting, acquisition, bankruptcy, etc.).

**Input:**

- `dual_class_output_nocik.json` - Companies without CIK numbers
- SEC company ticker cache

**Process:**

1. **Ticker Variant Matching:** Tries multiple ticker variations (with/without dots, dashes)
2. **Company Name Matching:** Uses exact and fuzzy string matching against SEC company names
3. **SEC Filing Analysis:** For found CIKs, fetches recent 8-K, Form 25, and Form 15 filings
4. **AI Investigation:** Uses OpenAI to research companies not found in SEC data
5. **Filing Event Detection:** Identifies delisting events, bankruptcies, and corporate actions

**Output:**

- Enhanced JSON with resolution data:
  ```json
  {
    "company_name": "Example Corp",
    "primary_ticker": "EXAM",
    "reason": "found_by_ticker_variant",
    "reason_detail": {
      "matched_variant": "EXAM",
      "cik": "0001234567"
    },
    "filings_summary": ["2025-08-15: 8-K (Items: 1.03)", "2025-07-30: 10-Q"],
    "deregistration_flag": false,
    "AI_REASON": "Company appears active and trading normally."
  }
  ```

**Key Features:**

- **Filing Analysis:** Parses 8-K items 1.03 (bankruptcy), 2.01 (completion of acquisition), 3.01 (delistings)
- **AI Integration:** GPT-4 powered investigation of corporate events
- **Exchange Monitoring:** Placeholder hooks for NASDAQ/NYSE delisting alerts
- **Fuzzy Matching:** Uses `difflib` for approximate company name matching

---

### Step 2: `2_MapTickerToShareClass.py` - SEC Filing Share Class Mapping

**Purpose:** Extracts precise ticker-to-share-class mappings directly from SEC filings' cover page tables.

**Input:**

- CIK numbers (individual or from previous steps)
- Company SEC filings (10-K/10-Q)

**Process:**

1. **Filing Retrieval:** Downloads latest 10-K or 10-Q from SEC EDGAR
2. **Table Detection:** Locates "Securities registered pursuant to Section 12(b)" table
3. **AI Parsing:** Uses GPT-4o to extract structured data from HTML tables:
   - Ticker symbols
   - Share class titles
   - Exchange listings
   - Economic equivalence ratios
   - Share counts (if available)
4. **Data Validation:** Filters out debt instruments, validates ticker formats

**Output:**

```json
{
  "cik": "0001652044",
  "ticker_mappings": [
    {
      "ticker": "GOOGL",
      "title_of_class": "Class A Common Stock, $0.001 par value",
      "exchange": "Nasdaq Stock Market LLC",
      "economic_equivalent_to_primary": 1.0,
      "share_count": null,
      "relative_weight": null
    },
    {
      "ticker": "GOOG",
      "title_of_class": "Class C Capital Stock, $0.001 par value",
      "exchange": "Nasdaq Stock Market LLC",
      "economic_equivalent_to_primary": 1.0,
      "share_count": null,
      "relative_weight": null
    }
  ],
  "filing_info": {
    "form_type": "10-Q",
    "accession_number": "0001652044-25-000062",
    "filing_date": "2025-07-24"
  },
  "success": true
}
```

**Usage:**

```bash
python 2_MapTickerToShareClass.py --cik 0001652044
python 2_MapTickerToShareClass.py --cik 0001326801 --model gpt-4
```

**Limitations:**

- Requires companies to have standard SEC cover page table format
- Some filings may not contain parseable share class information
- Success rate varies by company and filing structure

---

### Step 3: `3_GetEconomicWeight.py` - Economic Weight Analysis

**Purpose:** Calculates economic weights (ownership percentages) for each share class by analyzing SEC filings for share outstanding counts.

**Input:**

- `dual_class_output.json` - Companies with CIK numbers and voting structures

**Process:**

1. **SEC Filing Download:** Retrieves latest 10-K/10-Q filings
2. **AI Content Analysis:** Uses GPT-4o to extract from filing text:
   - Shares outstanding by class
   - Conversion ratios between classes
   - Economic equivalence relationships
3. **Economic Weight Calculation:**
   ```python
   economic_weight = (shares_outstanding * conversion_ratio) / total_economic_shares
   ```
4. **Data Merging:** Combines AI-extracted economic data with existing voting rights
5. **Quality Validation:** Flags inconsistencies and data quality issues

**Output:**

- `dual_class_economic_weights.json` - Complete economic analysis:
  ```json
  {
    "company_name": "1-800-FLOWERS.COM",
    "primary_ticker": "FLWS",
    "classes": [
      {
        "class_name": "Class A",
        "votes_per_share": 1.0,
        "shares_outstanding": 64660147,
        "economic_weight": 0.9186,
        "conversion_ratio": 1.0,
        "source": "AI analysis of SEC filing"
      },
      {
        "class_name": "Class B",
        "votes_per_share": 10.0,
        "shares_outstanding": 5730000,
        "economic_weight": 0.0814,
        "conversion_ratio": 1.0,
        "source": "AI analysis of SEC filing"
      }
    ],
    "cik": "0001084869"
  }
  ```

**Key Features:**

- **Company-Specific Analysis:** Extracts actual conversion ratios (not hardcoded values)
- **Test Mode:** Can process subset of companies for validation
- **Error Handling:** Graceful handling of filing download failures and parsing errors
- **AI Prompt Engineering:** Sophisticated prompts for accurate financial data extraction

**Usage:**

```bash
# Process all companies
python 3_GetEconomicWeight.py

# Test mode with 10 companies
python 3_GetEconomicWeight.py --test 10

# Process specific date
python 3_GetEconomicWeight.py --date "20Aug2025"
```

## Data Flow Summary

```
Raw CSV Data → Step 1 → JSON with CIKs → Step 1.75 → Resolved CIKs
                ↓
Step 3 ← Step 2 (Optional: Enhanced ticker mappings)
   ↓
Final Economic Weights JSON
```

## File Structure

```
├── DualClassList_19Aug2025.csv          # Input: Raw company data
├── dual_class_output.json               # Step 1 output: Companies with CIKs
├── dual_class_output_nocik.json         # Companies without CIKs (for Step 1.75)
├── dual_class_economic_weights.json     # Step 3 output: Final analysis
├── 1_dual_class_ingest.py              # Step 1: Data ingestion
├── 1.75_nocik.py                       # Step 1.75: Missing CIK resolution
├── 2_MapTickerToShareClass.py          # Step 2: SEC filing ticker mapping
└── 3_GetEconomicWeight.py              # Step 3: Economic weight calculation
```

## Environment Setup

### Required Dependencies

```bash
pip install requests beautifulsoup4 openai python-dotenv
```

### Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o  # Optional: defaults to gpt-4o
```

### SEC API Requirements

- All scripts include proper User-Agent headers as required by SEC
- Built-in rate limiting to respect SEC API guidelines
- Automatic retry logic for network failures

## Common Usage Patterns

### Full Pipeline Execution

```bash
# Step 1: Ingest raw data
python 1_dual_class_ingest.py

# Step 1.75: Resolve missing CIKs (if needed)
python 1.75_nocik.py --ai-check

# Step 3: Calculate economic weights
python 3_GetEconomicWeight.py
```

### Individual Company Analysis

```bash
# Analyze specific company's share structure
python 2_MapTickerToShareClass.py --cik 0001652044

# Investigate missing CIK
python 1.75_nocik.py --cik 0001234567 --symbol TICK --exchange NASDAQ
```

### Testing and Validation

```bash
# Test economic weight calculation on 10 companies
python 3_GetEconomicWeight.py --test 10

# AI investigation with delays
python 1.75_nocik.py --ai-check
```

## Data Quality Considerations

### Known Issues

1. **Empty Class Names:** Some CSV parsing creates invalid entries
2. **Duplicate Classes:** Company restructuring can create multiple similar entries
3. **AI Parsing Accuracy:** Complex SEC filings may not parse correctly
4. **Missing Share Counts:** Not all filings contain explicit share outstanding numbers

### Quality Controls

- Input validation at each step
- Error logging and graceful degradation
- Test modes for validation
- Manual review flags for edge cases

## Output Analysis

The final economic weights enable several analyses:

- **Voting vs Economic Power:** Compare voting control to economic ownership
- **Control Premium Calculation:** Quantify the value of superior voting rights
- **Portfolio Normalization:** Adjust holdings for economic rather than voting exposure
- **Corporate Governance Analysis:** Assess management entrenchment through dual-class structures

## Technical Notes

### AI Integration

- Uses OpenAI GPT-4o for complex document parsing
- Structured prompts ensure consistent JSON output
- Built-in error handling for API failures
- Optional fallback to alternative models

### SEC API Compliance

- Proper User-Agent identification
- Rate limiting (0.1-0.2 second delays)
- Robust error handling for API changes
- Caching to minimize API calls

### Performance

- Concurrent processing where possible
- Intelligent caching of SEC data
- Configurable batch sizes
- Progress tracking for long-running operations

This pipeline provides a comprehensive solution for normalizing dual-class share structures, enabling more accurate investment analysis and portfolio management.
