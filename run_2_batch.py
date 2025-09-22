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
    parser = argparse.ArgumentParser(description="Run Step 2 (OpenAI normalization) for multiple companies")
    parser.add_argument('--input', default='staging/1_dual_class_output.json', help='Path to dual-class output JSON')
    parser.add_argument('--limit', type=int, default=20, help='How many companies to process')
    parser.add_argument('--skip-existing', action='store_true', help='Skip if normalized output already exists')
    args = parser.parse_args()

    # Prefer venv python if present
    python_exe = sys.executable
    if os.path.exists('.venv'):
        venv_py = os.path.join('.venv', 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join('.venv', 'bin', 'python')
        if os.path.exists(venv_py):
            python_exe = venv_py

    ensure_input_json(python_exe, args.input)
    companies = load_companies(args.input)
    if not companies:
        print('[ERROR] No companies with CIKs found in', args.input)
        sys.exit(1)

    targets = companies[: args.limit]

    successes, failures, skipped = [], [], []
    start = datetime.now(UTC)

    for c in targets:
        cik = str(c['cik']).zfill(10)
        out_path = os.path.join('fileoutput', 'equity_classes', f'cik_{cik}_equity_classes.json')
        if args.skip_existing and os.path.exists(out_path):
            print(f"[SKIP] {cik} already has {out_path}")
            skipped.append(cik)
            continue

        cmd = [python_exe, '-u', '2_RetrieveData.py', '--cik', cik]
        print(f"[RUN-2] {' '.join(cmd)}")
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
    print('\n=== Step 2 Batch Summary ===')
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
