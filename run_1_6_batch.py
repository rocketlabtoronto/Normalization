import os
import sys
import json
import argparse
import subprocess
from datetime import datetime, UTC
from typing import List, Dict


def load_companies(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [c for c in data.get('companies', []) if c.get('cik')]


def ensure_input_json(python_exe: str, input_path: str) -> None:
    """Ensure the companies list JSON exists; if missing and path is the default,
    run step 1 to generate it."""
    default_path = 'staging/1_dual_class_output.json'
    if os.path.exists(input_path):
        return
    if input_path != default_path:
        raise FileNotFoundError(f"[ERROR] {input_path} not found and auto-generate only supported for {default_path}")
    print(f"[INIT] {default_path} not found; running step 1 to generate it...")
    cmd = [python_exe, '-u', '1_dual_class_csv_to_json_converter.py']
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0 or not os.path.exists(default_path):
        raise RuntimeError(f"[ERROR] Failed to generate {default_path}")


def main():
    parser = argparse.ArgumentParser(description="Run 1.5 (download) and 1.6 (extract) for multiple companies")
    parser.add_argument('--input', default='staging/1_dual_class_output.json', help='Path to dual-class output JSON')
    parser.add_argument('--limit', type=int, default=20, help='How many companies to process')
    parser.add_argument('--skip-existing', action='store_true', help='Skip if staging output already exists')
    args = parser.parse_args()

    # Prefer venv python if present
    python_exe = sys.executable
    if os.path.exists('.venv'):
        venv_py = os.path.join('.venv', 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join('.venv', 'bin', 'python')
        if os.path.exists(venv_py):
            python_exe = venv_py

    # Ensure the input JSON exists (generate if needed)
    ensure_input_json(python_exe, args.input)

    companies = load_companies(args.input)
    if not companies:
        print('[ERROR] No companies with CIKs found in', args.input)
        sys.exit(1)

    # Choose up to limit
    targets = companies[: args.limit]

    successes, failures, skipped = [], [], []
    start = datetime.now(UTC)

    for c in targets:
        cik = str(c['cik']).zfill(10)
        out_path = os.path.join('staging', f'cik_{cik}_equity_extraction.json')

        if args.skip_existing and os.path.exists(out_path):
            print(f"[SKIP] {cik} already has {out_path}")
            skipped.append(cik)
            continue

        # Step 1.5: Download 10-K before extraction (skip if download record exists and skipping is enabled)
        dl_record = os.path.join('staging', f'cik_{cik}_10k_download.json')
        if args.skip_existing and os.path.exists(dl_record):
            print(f"[SKIP-DL] {cik} already has {dl_record}")
        else:
            dl_cmd = [python_exe, '-u', '1.5_Download10K.py', '--cik', cik]
            print(f"[RUN-DL] {' '.join(dl_cmd)}")
            dl_res = subprocess.run(dl_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if dl_res.returncode != 0:
                print(f"[FAIL-DL] {cik} rc={dl_res.returncode}\n{dl_res.stdout}\n{dl_res.stderr}")
                failures.append(cik)
                continue

        # Step 1.6: Extract sections
        cmd = [python_exe, '-u', '1.6_Extract10K.py', '--cik', cik]
        print(f"[RUN] {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode == 0 and os.path.exists(out_path):
                print(f"[OK] {cik} -> {out_path}")
                successes.append(cik)
            else:
                print(f"[FAIL] {cik} rc={result.returncode}\n{result.stdout}\n{result.stderr}")
                failures.append(cik)
        except Exception as e:
            print(f"[ERROR] {cik} {e}")
            failures.append(cik)

    end = datetime.now(UTC)
    print('\n=== 1.5 + 1.6 Batch Summary ===')
    print('Started:', start.isoformat())
    print('Ended  :', end.isoformat())
    print(f"Success: {len(successes)} | Fail: {len(failures)} | Skipped: {len(skipped)}")
    if successes:
        print('Successful CIKs:', ', '.join(successes))
    if failures:
        print('Failed CIKs    :', ', '.join(failures))
    if skipped:
        print('Skipped CIKs   :', ', '.join(skipped))


if __name__ == '__main__':
    main()
