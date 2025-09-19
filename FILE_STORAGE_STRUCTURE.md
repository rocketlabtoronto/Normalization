# SEC Filing Storage Structure

## Overview

The pipeline now saves SEC filings to organized folders to eliminate duplicate downloads between Step 1.75 and Step 2.

## Folder Structure

```
sec_filings/
├── 8K/                           # 8-K filings from Step 1.75
│   ├── 0000123456_000012345621000123_document.htm
│   ├── 0000789012_000078901221000456_8k.htm
│   └── ...
├── 10K/                          # 10-K filings (downloaded by Step 1.75 or Step 2)
│   ├── 0000123456_000012345621000789_10k.htm
│   ├── 0000789012_000078901221000987_document.htm
│   └── ...
├── 10Q/                          # 10-Q filings (downloaded by Step 1.75 or Step 2)
│   ├── 0000123456_000012345621000345_10q.htm
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

## File Naming Convention

### SEC Filing Documents

- **Format**: `{CIK_padded}_{accession_clean}_{document_name}`
- **Example**: `0000123456_000012345621000123_document.htm`
- **Components**:
  - `CIK_padded`: 10-digit zero-padded CIK number
  - `accession_clean`: Accession number with dashes removed
  - `document_name`: Original SEC document filename

### 8-K Items JSON Files

- **Location**: `sec_filings/8k_items/CIK{cik_padded}/`
- **Format**: `{CIK_padded}_{accession_clean}_items.json`
- **Example**: `0000123456_000012345621000123_items.json`

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

### Step 1.75 (Missing Company Investigator)

- **Downloads**: 8-K, Form 25, Form 15 filings
- **Saves to**: `sec_filings/{form_type}/`
- **Extracts**: 8-K items → `sec_filings/8k_items/CIK{cik}/`

### Step 2 (SEC Filing Ticker Mapper)

- **Downloads**: 10-K, 10-Q filings (if not already saved)
- **Saves to**: `sec_filings/{form_type}/`
- **Reuses**: Files from Step 1.75 when available

## Benefits

1. **No Duplicate Downloads**: Step 2 reuses 10-K/10-Q files if already downloaded
2. **Organized Storage**: Files grouped by form type and company
3. **Easy Access**: Consistent naming makes files easy to find
4. **JSON Extracts**: 8-K items saved in structured format for analysis
5. **Cross-Step Sharing**: Pipeline steps can share downloaded files

## Usage Notes

- Files are permanent storage (not cache) and persist between runs
- Step 2 checks multiple form folders (8K, 10K, 10Q) to find existing files
- 8-K items are only saved if they contain actual extracted content
- All downloads include proper SEC rate limiting and error handling
