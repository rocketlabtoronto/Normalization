#!/usr/bin/env python3
"""
Batch Pipeline Runner
Runs the complete OpenAI equity normalization pipeline for all companies with CIKs.

This script orchestrates the full workflow:
1. Automatically cleans workspace (removes old intermediate/output files)
2. Loads companies from staging/1_dual_class_output.json
3. For each company with a CIK:
   - Step 1.5: Downloads latest 10-K filing
   - Step 1.6: Extracts equity class details  
   - Step 2: Normalizes with OpenAI
4. Provides progress tracking and error handling
5. Generates summary report of results

Usage:
    python batch_pipeline.py                    # Process all companies (auto-cleanup)
    python batch_pipeline.py --start-from 10    # Resume from company index 10
    python batch_pipeline.py --limit 5          # Process only first 5 companies
    python batch_pipeline.py --ciks 1084869,4977  # Process specific CIKs only
    python batch_pipeline.py --skip-existing    # Skip companies with existing results
    python batch_pipeline.py --dry-run          # Show what would be processed without running
    python batch_pipeline.py --no-cleanup       # Skip automatic workspace cleanup
"""

import json
import os
import sys
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Optional, Set
import argparse
import shutil
from pathlib import Path


def cleanup_workspace(skip_confirmation: bool = False) -> int:
    """Clean up staging and output files while preserving SEC filings"""
    
    def cleanup_staging_files():
        """Remove all files from staging/ folder except the source data"""
        staging_dir = Path("staging")
        if not staging_dir.exists():
            return []
        
        deleted_files = []
        preserve_files = ["1_dual_class_output.json"]  # Keep the source company list
        
        for file_path in staging_dir.iterdir():
            if file_path.is_file() and file_path.name not in preserve_files:
                deleted_files.append(str(file_path))
        
        return deleted_files

    def cleanup_output_files():
        """Remove all files from fileoutput/ folder"""
        output_dir = Path("fileoutput")
        if not output_dir.exists():
            return []
        
        deleted_files = []
        
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                deleted_files.append(str(file_path))
        
        return deleted_files

    def cleanup_results_files():
        """Remove batch processing reports from results/ folder"""
        results_dir = Path("results")
        if not results_dir.exists():
            return []
        
        deleted_files = []
        
        for file_path in results_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith('batch_pipeline_report_'):
                deleted_files.append(str(file_path))
        
        return deleted_files
    
    # Collect files that would be deleted
    staging_files = cleanup_staging_files()
    output_files = cleanup_output_files()
    results_files = cleanup_results_files()
    
    all_files = staging_files + output_files + results_files
    
    if not all_files:
        print("📋 Workspace is already clean!")
        return 0
    
    print(f"🧹 Cleaning workspace - removing {len(all_files)} files...")
    
    if not skip_confirmation:
        print("   📁 Staging files (except source data)")
        print("   📁 Output files") 
        print("   📁 Old batch reports")
        print("   ✅ Preserving: SEC 10-K filings, source company data")
        
        response = input("\nProceed with cleanup? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("❌ Cleanup cancelled")
            return 0
    
    # Perform actual deletion
    deleted_count = 0
    
    # Delete files
    for file_path in all_files:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"  ⚠️  Could not delete {file_path}: {e}")
    
    # Remove empty directories
    try:
        output_dir = Path("fileoutput")
        if output_dir.exists():
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    if not any(dir_path.iterdir()):  # If directory is empty
                        dir_path.rmdir()
    except Exception:
        pass  # Ignore directory cleanup errors
    
    print(f"✅ Cleanup complete! Removed {deleted_count} files")
    return deleted_count


def load_companies_data(filepath: str = "staging/1_dual_class_output.json") -> Dict:
    """Load the companies data from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Companies file not found: {filepath}")
        print("Please run step 1 first: python 1_dual_class_csv_to_json_converter.py")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {filepath}: {e}")
        sys.exit(1)


def get_companies_with_ciks(companies_data: Dict) -> List[Dict]:
    """Filter companies that have valid CIK numbers"""
    companies = companies_data.get("companies", [])
    valid_companies = []
    
    for company in companies:
        cik = company.get("cik", "").strip()
        company_name = company.get("company_name", "").strip()
        
        # Skip empty rows and companies without CIKs
        if cik and cik != "" and company_name and company_name != "Company Name":
            valid_companies.append(company)
    
    return valid_companies


def check_existing_results(cik: str) -> Dict[str, bool]:
    """Check which pipeline steps have already been completed for a CIK"""
    cik_padded = str(cik).zfill(10)
    
    results = {
        "download": False,
        "extraction": False, 
        "normalization": False
    }
    
    # Check Step 1.5 - Download
    download_file = f"staging/cik_{cik_padded}_10k_download.json"
    if os.path.exists(download_file):
        try:
            with open(download_file, 'r') as f:
                data = json.load(f)
                if data.get("filing", {}).get("downloaded"):
                    results["download"] = True
        except:
            pass
    
    # Check Step 1.6 - Extraction
    extraction_file = f"staging/cik_{cik_padded}_equity_extraction.json"
    if os.path.exists(extraction_file):
        results["extraction"] = True
    
    # Check Step 2 - Normalization
    normalization_file = f"fileoutput/equity_classes/cik_{cik_padded}_equity_classes.json"
    if os.path.exists(normalization_file):
        results["normalization"] = True
    
    return results


def run_pipeline_step(script_name: str, cik: str, company_name: str, step_name: str) -> bool:
    """Run a single pipeline step and return success status"""
    print(f"  🔄 {step_name}: Running {script_name}")
    
    try:
        cmd = ["python", script_name, "--cik", cik]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"  ✅ {step_name}: Completed successfully")
            return True
        else:
            print(f"  ❌ {step_name}: Failed with return code {result.returncode}")
            if result.stderr:
                print(f"     Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {step_name}: Timeout after 5 minutes")
        return False
    except Exception as e:
        print(f"  ❌ {step_name}: Exception - {e}")
        return False


def process_company(company: Dict, skip_existing: bool = False, dry_run: bool = False) -> Dict:
    """Process a single company through the complete pipeline"""
    
    cik = company.get("cik", "").strip()
    company_name = company.get("company_name", "Unknown")
    ticker = company.get("primary_ticker", "")
    
    print(f"\n🏢 Processing: {company_name} ({ticker}) - CIK: {cik}")
    
    result = {
        "cik": cik,
        "company_name": company_name,
        "ticker": ticker,
        "steps_completed": [],
        "steps_failed": [],
        "skipped": False,
        "processing_time": 0
    }
    
    if dry_run:
        print(f"  🔍 DRY RUN: Would process {company_name}")
        result["skipped"] = True
        return result
    
    start_time = time.time()
    
    # Check existing results
    existing = check_existing_results(cik)
    
    if skip_existing and all(existing.values()):
        print(f"  ⏭️ Skipping - all steps already completed")
        result["skipped"] = True
        result["steps_completed"] = ["download", "extraction", "normalization"]
        return result
    
    # Step 1.5: Download 10-K
    if existing["download"] and skip_existing:
        print(f"  ⏭️ Step 1.5: Skipping - already downloaded")
        result["steps_completed"].append("download")
    else:
        if run_pipeline_step("1.5_Download10K.py", cik, company_name, "Step 1.5"):
            result["steps_completed"].append("download")
        else:
            result["steps_failed"].append("download")
            result["processing_time"] = time.time() - start_time
            return result
    
    # Step 1.6: Extract Equity Data
    if existing["extraction"] and skip_existing:
        print(f"  ⏭️ Step 1.6: Skipping - already extracted")
        result["steps_completed"].append("extraction")
    else:
        if run_pipeline_step("1.6_Extract10K.py", cik, company_name, "Step 1.6"):
            result["steps_completed"].append("extraction")
        else:
            result["steps_failed"].append("extraction")
            result["processing_time"] = time.time() - start_time
            return result
    
    # Step 2: OpenAI Normalization
    if existing["normalization"] and skip_existing:
        print(f"  ⏭️ Step 2: Skipping - already normalized")
        result["steps_completed"].append("normalization")
    else:
        if run_pipeline_step("2_RetrieveData.py", cik, company_name, "Step 2"):
            result["steps_completed"].append("normalization")
        else:
            result["steps_failed"].append("normalization")
            result["processing_time"] = time.time() - start_time
            return result
    
    result["processing_time"] = time.time() - start_time
    print(f"  🎉 Completed all steps in {result['processing_time']:.1f}s")
    
    return result


def save_batch_report(results: List[Dict], total_time: float):
    """Save a comprehensive batch processing report"""
    
    # Calculate statistics
    total_companies = len(results)
    successful = len([r for r in results if len(r["steps_failed"]) == 0 and not r["skipped"]])
    skipped = len([r for r in results if r["skipped"]])
    failed = len([r for r in results if len(r["steps_failed"]) > 0])
    
    # Count step-specific failures
    step_failures = {
        "download": len([r for r in results if "download" in r["steps_failed"]]),
        "extraction": len([r for r in results if "extraction" in r["steps_failed"]]),
        "normalization": len([r for r in results if "normalization" in r["steps_failed"]])
    }
    
    report = {
        "batch_run_at": datetime.utcnow().isoformat() + "Z",
        "total_processing_time": round(total_time, 1),
        "summary": {
            "total_companies": total_companies,
            "successful": successful,
            "skipped": skipped, 
            "failed": failed,
            "success_rate": round((successful / max(total_companies - skipped, 1)) * 100, 1) if total_companies > skipped else 0
        },
        "step_failures": step_failures,
        "company_results": results
    }
    
    # Save report
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"results/batch_pipeline_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report_file, report


def main():
    parser = argparse.ArgumentParser(description="Run the complete equity normalization pipeline for all companies")
    parser.add_argument("--start-from", type=int, help="Start processing from company index N")
    parser.add_argument("--limit", type=int, help="Process only N companies")
    parser.add_argument("--ciks", help="Process only specific CIKs (comma-separated)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip companies with existing results")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without running")
    parser.add_argument("--input-file", default="staging/1_dual_class_output.json", help="Input companies JSON file")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip automatic workspace cleanup at start")
    
    args = parser.parse_args()
    
    print("🚀 Batch Pipeline Runner")
    print("=" * 50)
    
    # Automatic workspace cleanup (unless disabled)
    if not args.no_cleanup and not args.dry_run:
        cleanup_workspace(skip_confirmation=True)
        print()
    
    # Load companies data
    print(f"📂 Loading companies from: {args.input_file}")
    companies_data = load_companies_data(args.input_file)
    
    # Filter companies with valid CIKs
    valid_companies = get_companies_with_ciks(companies_data)
    print(f"📊 Found {len(valid_companies)} companies with valid CIKs")
    
    # Filter by specific CIKs if provided
    if args.ciks:
        target_ciks = set(cik.strip().zfill(10) for cik in args.ciks.split(","))
        valid_companies = [c for c in valid_companies if c.get("cik", "").zfill(10) in target_ciks]
        print(f"🎯 Filtered to {len(valid_companies)} companies matching specified CIKs")
    
    # Apply start-from filter
    if args.start_from:
        valid_companies = valid_companies[args.start_from:]
        print(f"⏩ Starting from index {args.start_from}, {len(valid_companies)} companies remaining")
    
    # Apply limit filter
    if args.limit:
        valid_companies = valid_companies[:args.limit]
        print(f"🔢 Limited to first {len(valid_companies)} companies")
    
    if not valid_companies:
        print("❌ No companies to process")
        sys.exit(1)
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No actual processing will occur")
    
    print(f"\n📋 Processing {len(valid_companies)} companies...")
    print("=" * 50)
    
    # Process each company
    results = []
    start_time = time.time()
    
    for i, company in enumerate(valid_companies, 1):
        print(f"\n[{i}/{len(valid_companies)}]", end=" ")
        
        try:
            result = process_company(company, args.skip_existing, args.dry_run)
            results.append(result)
            
            # Brief pause between companies to be respectful to SEC API
            if not args.dry_run:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted by user at company {i}")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error processing {company.get('company_name', 'Unknown')}: {e}")
            results.append({
                "cik": company.get("cik", ""),
                "company_name": company.get("company_name", "Unknown"),
                "ticker": company.get("primary_ticker", ""),
                "steps_completed": [],
                "steps_failed": ["unexpected_error"],
                "skipped": False,
                "processing_time": 0,
                "error": str(e)
            })
    
    total_time = time.time() - start_time
    
    # Generate and save report
    if not args.dry_run:
        report_file, report = save_batch_report(results, total_time)
        
        # Print summary
        print(f"\n" + "=" * 50)
        print("📊 BATCH PROCESSING COMPLETE")
        print("=" * 50)
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"🏢 Companies processed: {report['summary']['total_companies']}")
        print(f"✅ Successful: {report['summary']['successful']}")
        print(f"⏭️  Skipped: {report['summary']['skipped']}")
        print(f"❌ Failed: {report['summary']['failed']}")
        print(f"📈 Success rate: {report['summary']['success_rate']}%")
        
        if report['summary']['failed'] > 0:
            print(f"\n📋 Step-specific failures:")
            for step, count in report['step_failures'].items():
                if count > 0:
                    print(f"   {step}: {count} failures")
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        # Show next steps
        successful_companies = [r for r in results if len(r["steps_failed"]) == 0 and not r["skipped"]]
        if successful_companies:
            print(f"\n🎉 Successfully processed {len(successful_companies)} companies!")
            print("📁 Results available in:")
            print("   - fileoutput/equity_classes/cik_XXXXXXXXXX_equity_classes.json (normalized data)")
            print("   - staging/cik_XXXXXXXXXX_equity_extraction.json (raw extraction)")
            print("   - staging/cik_XXXXXXXXXX_10k_download.json (download metadata)")
    
    else:
        print(f"\n🔍 DRY RUN COMPLETE - Would have processed {len(valid_companies)} companies")


if __name__ == "__main__":
    main()
