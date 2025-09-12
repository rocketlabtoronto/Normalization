# Dual-Class Share Normalization Pipeline

An automated Python pipeline that analyzes dual-class companies to extract accurate share class structures, economic weights, and voting ratios from SEC filings. Uses AI-powered document analysis with deterministic fallback parsing for reliable data extraction.

## 🎯 Purpose

This pipeline solves the challenge of normalizing dual-class share structures across public companies by:

- **Extracting trading symbols** for each share class from SEC cover pages
- **Calculating economic weights** based on shares outstanding and conversion ratios
- **Mapping voting structures** to determine control vs. economic ownership
- **Providing normalized data** for investment analysis and portfolio construction

## 🔧 Pipeline Architecture

### Step 1: Company Ingestion (`1_dual_class_ingest.py`)

- Processes dual-class company lists with CIK resolution
- Validates SEC ticker mappings and company metadata
- Outputs structured company data with voting classifications

### Step 2: Ticker-to-Share-Class Mapping (`2_MapTickerToShareClass.py`)

- Parses SEC filing cover pages to extract trading symbols
- Uses GPT-4o analysis with HTML table fallback parsing
- Augments missing share classes from input definitions
- Implements retry logic and content optimization

### Step 3: Economic Weight Analysis (`3_GetEconomicWeight.py`)

- Downloads recent SEC filings (10-K/10-Q) via EDGAR API
- Extracts shares outstanding and conversion ratios using AI
- Calculates economic weights and relative ownership percentages
- Merges with existing voting data for complete analysis

## 🚀 Key Features

- **AI-Powered Extraction**: GPT-4o analysis of SEC documents with structured JSON output
- **Robust Error Handling**: Retry logic, content optimization, and deterministic fallbacks
- **Rate Limiting**: SEC EDGAR API compliance with proper headers and delays
- **Data Validation**: JSON schema validation and comprehensive logging
- **Modular Design**: Step-based pipeline with JSON file chaining

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
