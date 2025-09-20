#!/usr/bin/env python3
"""
cleanup_workspace.py - Clean up staging and output files while preserving SEC filings

This script removes all intermediate processing files and final outputs to provide
a clean workspace for fresh pipeline runs. Preserves the downloaded SEC 10-K filings
in sec_filings/10K/ to avoid re-downloading.

Usage:
    python cleanup_workspace.py
    python cleanup_workspace.py --dry-run  # Preview what would be deleted
"""

import os
import shutil
import argparse
from pathlib import Path

def cleanup_staging_files():
    """Remove all files from staging/ folder except the source data"""
    staging_dir = Path("staging")
    if not staging_dir.exists():
        print("[INFO] No staging/ folder found")
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
        print("[INFO] No fileoutput/ folder found")
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
        print("[INFO] No results/ folder found")
        return []
    
    deleted_files = []
    
    for file_path in results_dir.iterdir():
        if file_path.is_file() and file_path.name.endswith('.json'):
            deleted_files.append(str(file_path))
    
    return deleted_files

def main():
    parser = argparse.ArgumentParser(description="Clean up workspace while preserving SEC 10-K filings")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    print("🧹 Workspace Cleanup Tool")
    print("=" * 50)
    print("This will delete:")
    print("  ✓ All staging files (except 1_dual_class_output.json)")
    print("  ✓ All fileoutput files")
    print("  ✓ All batch processing reports")
    print("This will PRESERVE:")
    print("  ✓ sec_filings/10K/ (downloaded 10-K filings)")
    print("  ✓ staging/1_dual_class_output.json (company list)")
    print("  ✓ All source code and configuration files")
    print()
    
    # Collect files that would be deleted
    staging_files = cleanup_staging_files()
    output_files = cleanup_output_files()
    results_files = cleanup_results_files()
    
    all_files = staging_files + output_files + results_files
    
    if not all_files:
        print("[INFO] No files found to clean up. Workspace is already clean!")
        return
    
    print(f"Found {len(all_files)} files to delete:")
    print()
    
    if staging_files:
        print(f"📁 Staging files ({len(staging_files)}):")
        for file_path in staging_files:
            print(f"  - {file_path}")
        print()
    
    if output_files:
        print(f"📁 Output files ({len(output_files)}):")
        for file_path in output_files:
            print(f"  - {file_path}")
        print()
    
    if results_files:
        print(f"📁 Results files ({len(results_files)}):")
        for file_path in results_files:
            print(f"  - {file_path}")
        print()
    
    if args.dry_run:
        print("🔍 DRY RUN COMPLETE - No files were actually deleted")
        return
    
    # Confirm deletion
    response = input("Do you want to proceed with deletion? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        print("❌ Cleanup cancelled")
        return
    
    # Perform actual deletion
    deleted_count = 0
    
    print("\n🗑️  Deleting files...")
    
    # Delete staging files
    for file_path in staging_files:
        try:
            os.remove(file_path)
            print(f"  ✓ Deleted: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"  ❌ Failed to delete {file_path}: {e}")
    
    # Delete output files and directories
    for file_path in output_files:
        try:
            os.remove(file_path)
            print(f"  ✓ Deleted: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"  ❌ Failed to delete {file_path}: {e}")
    
    # Delete results files
    for file_path in results_files:
        try:
            os.remove(file_path)
            print(f"  ✓ Deleted: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"  ❌ Failed to delete {file_path}: {e}")
    
    # Remove empty directories
    try:
        output_dir = Path("fileoutput")
        if output_dir.exists():
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    if not any(dir_path.iterdir()):  # If directory is empty
                        dir_path.rmdir()
                        print(f"  ✓ Removed empty directory: {dir_path}")
    except Exception as e:
        print(f"  ⚠️  Note: {e}")
    
    print(f"\n✅ Cleanup complete! Deleted {deleted_count} files")
    print("\n📋 Next steps:")
    print("  1. Run: python batch_pipeline.py --limit 3  # Test with 3 companies")
    print("  2. Run: python batch_pipeline.py           # Process all 53 companies")
    print("\n💾 Preserved:")
    
    # Count preserved 10-K files
    tenk_dir = Path("sec_filings/10K")
    if tenk_dir.exists():
        tenk_files = list(tenk_dir.glob("*.htm"))
        print(f"  ✓ {len(tenk_files)} downloaded 10-K filings in sec_filings/10K/")
    
    # Check if source data exists
    source_file = Path("staging/1_dual_class_output.json")
    if source_file.exists():
        print(f"  ✓ Company source data in staging/1_dual_class_output.json")

if __name__ == "__main__":
    main()
