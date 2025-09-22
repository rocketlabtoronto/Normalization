# SEC Filing Storage Structure

## Overview

The pipeline saves SEC filings to organized folders to eliminate duplicate downloads between pipeline steps. The OpenAI-powered workflow spans 10-K and 10-Q (Steps 1.5, 1.6, 1.51, 1.61, 2) to produce normalized equity class data.

## Folder Structure

```
sec_filings/
├── 8K/                           # 8-K filings from Step 1.75
│   ├── 0000123456_000012345621000123_document.htm
│   ├── 0000789012_000078901221000456_8k.htm
│   └── ...
├── 10K/                          # 10-K filings (downloaded by Steps 1.5, 1.75, or legacy processes)
│   ├── 0000123456_000012345621000789_10k.htm
│   ├── 0000789012_000078901221000987_document.htm
│   ├── 0001084869_0001084869-25-000017_flws-20250629.htm  # Example
│   └── README_FILENAME_STRUCTURE.md    # Documentation for filename format
├── 10Q/                          # 10-Q filings (downloaded by Step 1.51 or legacy processes)
│   ├── 0000123456_000012345621000345_10q.htm
│   ├── 0001799448_0000950170-25-104042_algs-20250630.htm  # Example
│   └── ...
├── 25/                           # Form 25 filings from Step 1.75
│   └── ...
├── 15/                           # Form 15 filings from Step 1.75
│   └── ...
└── 8k_items/                     # Extracted 8-K items in JSON format
    ├── CIK0000123456/
    │   ├── 0000123456_000012345621000123_items.json
    │   ├── 0000123456_000012345621000456_items.json
    │   └── ...
    ├── CIK0000789012/
    │   └── ...
    └── ...
```

## Staging Folder Structure

The `staging/` folder contains intermediate results and metadata from each pipeline step:

```
staging/
├── 1_dual_class_output.json                     # Step 1: Converted CSV data
├── cik_{cik}_10k_download.json                  # Step 1.5: 10-K download metadata
├── cik_{cik}_equity_extraction.json             # Step 1.6: 10-K equity extraction
├── cik_{cik}_10q_download.json                  # Step 1.51: 10-Q download metadata
├── cik_{cik}_equity_extraction_10q.json         # Step 1.61: 10-Q equity extraction
├── cik_{cik}_events.json                        # Step 1.75: Company events/status
├── 1.75_dual_class_output_nocik.json            # Step 1.75: Companies without CIKs
├── 1.75_dual_class_output_investigated.json     # Step 1.75: Investigation results
└── 2_step2_ticker_mappings.json                 # Legacy Step 2: Ticker mappings
```

## Final Output Folder Structure

The `fileoutput/` folder contains the final normalized results:

```
fileoutput/
└── equity_classes/
    ├── cik_{cik}_equity_classes.json            # Step 2: Normalized equity classes
    ├── cik_0001084869_equity_classes.json       # Example
    └── cik_0001799448_equity_classes.json       # Example
```

## File Naming Convention

### SEC Filing Documents

- Format: `{CIK_padded}_{accession_clean}_{document_name}`
- Example: `0000123456_000012345621000123_document.htm`
- Components:
  - CIK_padded: 10-digit zero-padded CIK number
  - accession_clean: Accession number with dashes removed
  - document_name: Original SEC document filename

### 8-K Items JSON Files

- Location: `sec_filings/8k_items/CIK{cik_padded}/`
- Format: `{CIK_padded}_{accession_clean}_items.json`
- Example: `0000123456_000012345621000123_items.json`

## JSON Structure for 8-K Items

```json
{
  "cik": "0000123456",
  "accession_number": "0001234562-21-000123",
  "filing_date": "2021-03-15",
  "form_type": "8-K",
  "extracted_items": {
    "1.03": "Bankruptcy proceedings excerpt...",
    "2.01": "Merger agreement details...",
    "3.01": "Material agreement information..."
  },
  "extraction_date": "2025-09-13T10:30:00.123456"
}
```

## Pipeline Integration

### Step 1.5 (Download 10-K)

- Downloads: Latest 10-K filings for individual companies
- Saves to: `sec_filings/10K/`
- Output: Download metadata → `staging/cik_{cik}_10k_download.json`
- Usage: `python 1.5_Download10K.py --cik CIK_NUMBER`

### Step 1.6 (Extract 10-K Equity Data)

- Reads from: `sec_filings/10K/` (files downloaded by Step 1.5)
- Extracts: Cover page, market equity, stockholders' equity notes, capital stock, and in-body descriptions
- Output: Structured equity data → `staging/cik_{cik}_equity_extraction.json`
- Usage: `python 1.6_Extract10K.py --cik CIK_NUMBER`

### Step 1.51 (Download 10-Q)

- Downloads: Latest 10-Q filing (skips if already downloaded)
- Saves to: `sec_filings/10Q/`
- Output: Download metadata → `staging/cik_{cik}_10q_download.json`
- Usage: `python 1.51_Download10Q.py --cik CIK_NUMBER`

### Step 1.61 (Extract 10-Q Equity Data)

- Reads from: `sec_filings/10Q/` (files downloaded by Step 1.51)
- Extracts: Cover page, market equity, stockholders' equity notes, capital stock, and in-body descriptions
- Output: Structured equity data → `staging/cik_{cik}_equity_extraction_10q.json`
- Usage: `python 1.61_Extract10Q.py --cik CIK_NUMBER`

### Step 2 (OpenAI Normalization, Holistic 10-K + 10-Q)

- Reads from: `staging/cik_{cik}_equity_extraction.json` and `staging/cik_{cik}_equity_extraction_10q.json` (if present)
- Processes: LLM-only holistic synthesis; enforces cover-page ticker rule; includes authorized-but-unissued classes; prefers most recent (10-Q) numbers
- Output: Normalized share classes → `fileoutput/equity_classes/cik_{cik}_equity_classes.json`
- Usage: `python 2_RetrieveData.py --cik CIK_NUMBER`

### Step 1.75 (Missing Company Investigator)

- Downloads: 8-K, Form 25, Form 15 filings
- Saves to: `sec_filings/{form_type}/`
- Extracts: 8-K items → `sec_filings/8k_items/CIK{cik}/`

### Step 2 (Legacy - SEC Filing Ticker Mapper)

- Downloads: 10-K, 10-Q filings (if not already saved)
- Saves to: `sec_filings/{form_type}/`
- Reuses: Files from other pipeline steps when available

## Benefits

1. No Duplicate Downloads: Pipeline steps reuse 10-K/10-Q files when available
2. Organized Storage: Files grouped by form type and company with consistent naming
3. Easy Access: Structured folders and naming make files easy to find and process
4. JSON Extracts: 8-K items and equity data saved in structured format for analysis
5. Cross-Step Sharing: Pipeline steps share downloaded files and intermediate results
6. OpenAI Pipeline: Direct path from 10-K/10-Q downloads → equity extraction → normalized data
7. Incremental Processing: Each step can be run independently or as part of full pipeline

## Usage Notes

- Files are permanent storage (not cache) and persist between runs
- New holistic pipeline (1.5 → 1.6 → 1.51 → 1.61 → 2) provides combined 10-K + 10-Q equity analysis
- Legacy steps check multiple form folders (8K, 10K, 10Q) to find existing files
- 8-K items are only saved if they contain actual extracted content
- All downloads include proper SEC rate limiting and error handling
- Staging files contain structured JSON for easy programmatic access
- Each CIK-specific file follows consistent naming: `cik_{cik_padded}_{purpose}.json`
