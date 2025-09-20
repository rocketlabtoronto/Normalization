# SEC Filing Filename Structure - 10K Folder

## Filename Format:
```
{CIK_padded}_{accession_number}_{primary_document}
```

## Component Breakdown:

### 1. **CIK_padded** (10 digits)
- **Format**: `0000123456`
- **Purpose**: Company's Central Index Key, zero-padded to 10 digits
- **Example**: `0001084869` = 1-800-FLOWERS.COM, INC.

### 2. **accession_number** (original format with dashes)
- **Format**: `0001234567-YY-NNNNNN`
- **Components**:
  - `0001234567` = Filing agent/law firm CIK
  - `YY` = Year (e.g., 24 = 2024)
  - `NNNNNN` = Sequential filing number
- **Example**: `0001437749-24-014253`

### 3. **primary_document** (original SEC document name)
- **Format**: Usually `ticker + date + form.htm`
- **Example**: `flws-20240430.htm`
  - `flws` = Ticker symbol
  - `20240430` = Filing date (YYYYMMDD)
  - `.htm` = Document format

## Example Complete Filename:
```
0001084869_0001437749-24-014253_flws-20240430.htm
```

**Translation:**
- Company: 1-800-FLOWERS (CIK: 0001084869)
- Filed by: Agent 0001437749 in 2024 (filing #014253)
- Document: FLWS 10-K form filed on April 30, 2024

## Original SEC URL Reconstruction:
```
https://www.sec.gov/Archives/edgar/data/{company_cik}/{accession_clean}/{primary_document}
```

**Example:**
```
https://www.sec.gov/Archives/edgar/data/1084869/000143774924014253/flws-20240430.htm
```

## Purpose:
- **Unique identification**: No filename conflicts
- **Chronological sorting**: Accession numbers sort by date
- **Company grouping**: CIK prefix groups by company
- **URL reconstruction**: Easy to rebuild original SEC links
- **Cross-pipeline reuse**: Step 2 can find and reuse these files

⚠️ **DO NOT DELETE THIS README** - It provides essential documentation for understanding the filing organization system.
