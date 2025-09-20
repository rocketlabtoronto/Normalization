# Dual-Class Share Normalization Pipeline

An automated Python pipeline that analyzes dual-class companies to extract accurate share class structures, economic weights, and voting ratios from SEC filings. Uses AI-powered document analysis with deterministic fallback parsing for reliable data extraction.

## 🎯 Purpose

This pipeline solves the challenge of normalizing dual-class share structures across public companies by:

- **Extracting trading symbols** for each share class from SEC cover pages
- **Calculating economic weights** based on shares outstanding and conversion ratios
- **Mapping voting structures** to determine control vs. economic ownership
- **Providing normalized data** for investment analysis and portfolio construction

## 🔧 Pipeline Architecture

### Step 1: Company Ingestion (`1_dual_class_csv_to_json_converter.py`)

- Processes dual-class company lists with CIK resolution
- Validates SEC ticker mappings and company metadata
- Outputs structured company data with voting classifications

### Step 1.5: 10-K Download (`1.5_Download10K.py`)

- Downloads latest 10-K filings for individual companies from SEC EDGAR
- Implements proper rate limiting and SEC compliance headers
- Saves filings to organized folder structure for reuse
- Usage: `python 1.5_Download10K.py --cik CIK_NUMBER`

### Step 1.6: Equity Extraction (`1.6_Extract10K.py`)

- Extracts comprehensive equity class details from downloaded 10-K filings
- Parses cover page, stockholders' equity notes, and exhibits
- Uses BeautifulSoup with XBRL metadata filtering for clean content
- Usage: `python 1.6_Extract10K.py --cik CIK_NUMBER`

### Step 2: OpenAI Normalization (`2_RetrieveData.py`)

- Uses OpenAI (GPT-4o) to normalize equity extraction data into structured arrays
- Automatically calculates voting and conversion weights (weakest class = 1.0)
- Produces clean JSON arrays with ticker, shares outstanding, and rights details
- Usage: `python 2_RetrieveData.py --cik CIK_NUMBER`

### Step 1.75: Missing Company Investigation (`1.75_missing_company_investigator.py`)

- Investigates companies missing CIK numbers using AI analysis
- Downloads 8-K, Form 25, and Form 15 filings for delisting research
- Provides explanations for missing companies (private, acquired, delisted)

## 🚀 Key Features

- **Direct 10-K Parsing**: Comprehensive extraction from actual SEC filings with XBRL filtering
- **OpenAI-Powered Normalization**: GPT-4o analysis for accurate equity class structure mapping
- **Automated Weight Calculation**: Normalizes voting and conversion weights relative to weakest class
- **Organized File Storage**: Reusable SEC filing downloads with consistent folder structure
- **Individual Company Processing**: Process specific companies by CIK for targeted analysis
- **Robust Error Handling**: Retry logic, content optimization, and comprehensive validation
- **Rate Limiting**: SEC EDGAR API compliance with proper headers and delays
- **Structured JSON Output**: Clean, normalized arrays ready for investment analysis

## 📋 Usage

### Individual Company Analysis (Recommended Workflow)

```bash
# Step 1: Download latest 10-K filing
python 1.5_Download10K.py --cik 1084869

# Step 2: Extract equity class details
python 1.6_Extract10K.py --cik 1084869

# Step 3: Normalize with OpenAI
python 2_RetrieveData.py --cik 1084869
```

### Alternative Input Methods

```bash
# Use direct file path instead of CIK
python 2_RetrieveData.py --file staging/cik_0001084869_equity_extraction.json

# Custom output location
python 2_RetrieveData.py --cik 1084869 --output my_analysis.json
```

### Batch Processing

```bash
# Convert CSV to JSON (Step 1)
python 1_dual_class_csv_to_json_converter.py

# Investigate missing companies (Step 1.75)
python 1.75_missing_company_investigator.py
```

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure OpenAI API key in .env file
echo "OPENAI_API_KEY=your_key_here" > .env
echo "LLM_MODEL=gpt-4o" >> .env
```

## 💼 Use Cases

- **Investment Research**: Analyze control vs. economic ownership in dual-class companies
- **Portfolio Construction**: Weight positions based on economic rather than market cap
- **Governance Analysis**: Understand voting control concentration
- **Risk Assessment**: Evaluate dual-class premium and structural risks

## 📊 Output

The pipeline produces structured JSON datasets with normalized equity class information:

### Example Output (`staging/cik_0001084869_equity_classes.json`)

```json
{
  "extracted_at": "2025-09-20T03:42:57.804399Z",
  "cik": "0001084869",
  "total_classes": 2,
  "extraction_method": "OpenAI gpt-4o",
  "normalization_note": "Voting and conversion weights normalized to weakest class = 1.0",
  "equity_classes": [
    {
      "ticker_symbol": "FLWS",
      "class_name": "Class A Common Stock",
      "shares_outstanding": 36550679,
      "conversion_weight": 1.0,
      "voting_weight": 1.0,
      "voting_rights": "Holders of Class A common stock have one vote per share.",
      "conversion_rights": "None",
      "par_value": "$0.01",
      "authorized_shares": 200000000
    },
    {
      "ticker_symbol": "FLWS",
      "class_name": "Class B Common Stock",
      "shares_outstanding": 27068221,
      "conversion_weight": 1.0,
      "voting_weight": 10.0,
      "voting_rights": "Holders of Class B common stock have 10 votes per share.",
      "conversion_rights": "Each share converts to one share of Class A upon transfer.",
      "par_value": "$0.01",
      "authorized_shares": 200000000
    }
  ]
}
```

### Key Data Points

- **Normalized Weights**: Voting and conversion ratios relative to weakest class (= 1.0)
- **Complete Rights**: Voting, conversion, dividend, and other rights descriptions
- **Corporate Details**: Par value, authorized shares, and share class structure
- **Trading Symbols**: Actual ticker symbols for each class of shares
- **Metadata**: Extraction timestamp, AI model used, and normalization methodology
