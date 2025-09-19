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

### Step 2: Ticker-to-Share-Class Mapping (`2_sec_filing_ticker_mapper.py`)

- Parses SEC filing cover pages to extract trading symbols
- Uses GPT-4o analysis with HTML table fallback parsing
- Augments missing share classes from input definitions
- Implements retry logic and content optimization

### Step 3: Economic Weight Analysis

Two analysis modes are available:

#### **OpenQuestion Mode (Default)** (`3_ai_powered_financial_analyzer.py`)

- **Token-efficient**: Directly queries AI about company share structures
- **Faster processing**: No SEC filing downloads required
- **Knowledge-based**: Leverages AI's training data about dual-class companies
- **Lower bandwidth**: Reduces API calls and download times

#### **SEC Filing Mode** (`3_placeholder_economic_weight_analyzer.py`)

- **Document-based**: Downloads recent SEC filings (10-K/10-Q) via EDGAR API
- **Comprehensive parsing**: Extracts shares outstanding and conversion ratios from actual filings
- **Deterministic fallback**: HTML table parsing when AI analysis fails
- **Most current data**: Uses latest filed information

Both modes:

- Calculate economic weights and relative ownership percentages
- Merge with existing voting data for complete analysis
- Support retry logic and error handling

## 🚀 Key Features

- **Dual Analysis Modes**: Choose between OpenQuestion (fast, token-efficient) or SEC Filing (comprehensive document parsing)
- **AI-Powered Extraction**: GPT-4o analysis with structured JSON output
- **Robust Error Handling**: Retry logic, content optimization, and deterministic fallbacks
- **Rate Limiting**: SEC EDGAR API compliance with proper headers and delays
- **Data Validation**: JSON schema validation and comprehensive logging
- **Modular Design**: Step-based pipeline with JSON file chaining

## 📋 Usage

### Basic Usage (OpenQuestion Mode - Default)

```bash
python main.py                    # Fast AI-based analysis
python main.py --test            # Test mode with sample data
```

### SEC Filing Mode

```bash
python main.py --noOpenQuestion  # Download and parse actual SEC filings
```

### Additional Options

```bash
python main.py --ai-check        # Enable AI investigation for missing CIKs
python main.py --skip-step 3     # Skip economic analysis
python main.py --resume-from 3   # Resume from specific step
```

## 💼 Use Cases

- **Investment Research**: Analyze control vs. economic ownership in dual-class companies
- **Portfolio Construction**: Weight positions based on economic rather than market cap
- **Governance Analysis**: Understand voting control concentration
- **Risk Assessment**: Evaluate dual-class premium and structural risks

## 📊 Output

Produces normalized JSON datasets containing:

- Trading symbols for each share class
- Shares outstanding and economic weights
- Voting ratios and conversion factors
- SEC filing metadata and data sources
