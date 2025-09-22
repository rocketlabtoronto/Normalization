#!/usr/bin/env python3
"""
Main Pipeline Controller for LookThroughProfits Dual-Class Share Normalization

This script orchestrates the entire pipeline from raw CSV input to final economic weight analysis.
Runs all steps in sequence with proper error handling and progress tracking.

Usage:
    python main.py                          # Run ALL CIKs (requires OpenAI API key)
    python main.py --test                   # Run first 3 CIKs (default test count)
    python main.py --test 5                 # Run first 5 CIKs
    python main.py --noOpenQuestion         # Use original SEC filing download method (requires OpenAI API key)
    python main.py --skip-step 3            # Skip economic analysis (no API key needed)
    python main.py --resume-from 3          # Resume from economic analysis (requires OpenAI API key)
    python main.py --ai-check               # Enable AI investigation for missing CIKs (requires OpenAI API key)
    python main.py --output-dir results     # Specify custom output directory
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path
import shutil
from datetime import datetime

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    try:
        # Try to set console to UTF-8 mode
        os.system('chcp 65001 > nul 2>&1')
        # Also set environment variable for subprocess calls
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except Exception:
        pass

# Attempt to load .env if python-dotenv is installed
try:  # optional
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Manual fallback .env loader if OPENAI_API_KEY still unset
if not os.getenv("OPENAI_API_KEY") and os.path.exists(".env"):
    try:
        with open(".env", "r", encoding="utf-8") as _f:
            for _ln in _f:
                if not _ln.strip() or _ln.strip().startswith("#"):
                    continue
                if "=" in _ln:
                    k, v = _ln.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# Helper: ensure OpenAI is installed (best-effort). Returns True if available.
def _ensure_openai_installed(logger_print=lambda m: None) -> bool:
    try:
        import openai  # noqa: F401
        return True
    except Exception:
        try:
            logger_print("OpenAI not installed; attempting: pip install openai")
            subprocess.run([sys.executable, "-m", "pip", "install", "openai"], check=True)
            import openai  # noqa: F401
            return True
        except Exception:
            return False

# Helper: load OPENAI_API_KEY and validate it's not a placeholder.
def _get_valid_openai_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key or key.strip() in ("", "your_openai_api_key_here"):
        return None
    return key.strip()

# Helper: choose Python exe (prefer venv)
def _python_exe() -> str:
    py = sys.executable
    if os.path.exists('.venv'):
        if sys.platform.startswith('win'):
            vpy = os.path.join('.venv', 'Scripts', 'python.exe')
        else:
            vpy = os.path.join('.venv', 'bin', 'python')
        if os.path.exists(vpy):
            py = vpy
    return py

class PipelineController:
    """Controls the execution of the dual-class share normalization pipeline"""
    
    def __init__(self, 
                 output_dir: str = ".",
                 test_mode: bool = False,
                 test_count: Optional[int] = None,
                 ai_check: bool = False,
                 verbose: bool = True,
                 no_open_question: bool = False,
                 cik: Optional[str] = None,
                 with_10q: bool = False):
        """
        Initialize the pipeline controller
        
        Args:
            output_dir: Directory for output files
            test_mode: Run in test mode with limited companies
            test_count: If test_mode True, limit to this number of companies (default None -> scripts default)
            ai_check: Enable AI investigation for missing CIKs
            verbose: Enable verbose logging
            no_open_question: Use original SEC filing download instead of OpenQuestion mode
            cik: Optional single CIK to process downstream steps for
            with_10q: Include 10-Q download/extraction in normalization flow
        """
        self.output_dir = Path(output_dir)
        self.test_mode = test_mode
        self.test_count = test_count
        self.ai_check = ai_check
        self.verbose = verbose
        self.no_open_question = no_open_question
        self.cik = (cik or '').strip()
        self.with_10q = with_10q
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Pipeline step definitions (modernized)
        self.steps: Dict[Any, Dict[str, Any]] = {
            1: {
                "name": "Data Ingestion",
                "script": "1_dual_class_csv_to_json_converter.py",
                "input_files": ["DualClassList.csv"],
                "output_files": ["staging/1_dual_class_output.json"],
                "required": True
            },
            1.5: {
                "name": "Download 10-K",
                "script": "1.5_Download10K.py",
                "required": False
            },
            1.6: {
                "name": "Extract 10-K Equity Data",
                "script": "1.6_Extract10K.py",
                "required": False
            },
            1.51: {
                "name": "Download 10-Q",
                "script": "1.51_Download10Q.py",
                "required": False,
                "conditional": True
            },
            1.61: {
                "name": "Extract 10-Q Equity Data",
                "script": "1.61_Extract10Q.py",
                "required": False,
                "conditional": True
            },
            1.75: {
                "name": "Missing CIK Resolution", 
                "script": "1.75_missing_company_investigator.py",
                "input_files": ["staging/1_dual_class_output.json"],
                "output_files": ["staging/1.75_dual_class_output_nocik.json"],
                "required": False,
                "conditional": True  # Only run if there are companies without CIKs
            },
            2: {
                "name": "Holistic Equity Normalization",
                "script": "2_RetrieveData.py",
                "required": True
            },
            3: {
                "name": "Economic Weight Analysis",
                "script": "3_ai_powered_financial_analyzer.py",  # Default to OpenQuestion version
                "script_fallback": "3_placeholder_economic_weight_analyzer.py",  # Original version when --noOpenQuestion
                "input_files": ["staging/1_dual_class_output.json"],
                "output_files": ["results/dual_class_economic_weights.json"],
                "required": True
            }
        }
        
        # Track pipeline state
        self.completed_steps = set()
        self.failed_steps = set()
        self.step_outputs: Dict[Any, Any] = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        if self.verbose or level == "ERROR":
            timestamp = datetime.now().strftime("%H:%M:%S")
            try:
                print(f"[{timestamp}] {level}: {message}")
            except UnicodeEncodeError:
                # Fallback for systems that can't handle Unicode
                safe_message = message.encode('ascii', errors='replace').decode('ascii')
                print(f"[{timestamp}] {level}: {safe_message}")
    
    def check_prerequisites(self, skip_steps: List = None) -> bool:
        """Check if all required files and dependencies are available"""
        skip_steps = skip_steps or []
        self.log("Checking prerequisites...")
        
        # Check for required input files
        required_files = ["DualClassList.csv"]
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(file):
                # Also allow input/DualClassList.csv as a fallback
                if file == "DualClassList.csv" and os.path.exists("input/DualClassList.csv"):
                    continue
                missing_files.append(file)
        
        if missing_files:
            self.log(f"Missing required input files: {', '.join(missing_files)}", "ERROR")
            return False
        
        # Check for required Python packages
        required_packages = ["requests", "beautifulsoup4", "openai"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                if package == "beautifulsoup4":
                    try:
                        __import__("bs4")
                    except ImportError:
                        missing_packages.append(package)
                else:
                    missing_packages.append(package)
        
        if missing_packages:
            self.log(f"Missing required packages: {', '.join(missing_packages)}", "ERROR")
            self.log("Install with: pip install " + " ".join(missing_packages), "ERROR")
            return False
        
        # Check for OpenAI API key only if AI features will definitely be used
        ai_steps_to_run = []
        # Step 2 and Step 3 need AI
        ai_needed = True  # Modern pipeline uses AI for normalization (Step 2) and analysis (Step 3)
        if ai_needed and not os.getenv("OPENAI_API_KEY"):
            self.log("OPENAI_API_KEY not set in environment", "ERROR")
            self.log("Options to set the API key:", "ERROR")
            self.log("  1. Create a .env file with: OPENAI_API_KEY=your_key_here", "ERROR")
            self.log("  2. Set environment variable: export OPENAI_API_KEY=your_key_here", "ERROR")
            self.log("  3. On Windows: set OPENAI_API_KEY=your_key_here", "ERROR")
            return False
        
        # Check for pipeline scripts existence
        missing_scripts = []
        for step_key, step_info in self.steps.items():
            script = step_info.get("script")
            if not script:
                continue
            if not os.path.exists(script):
                # Optional steps (conditional) shouldn't block pipeline, but warn
                if step_info.get("conditional"):
                    self.log(f"Warning: Optional script missing: {script}", "INFO")
                else:
                    missing_scripts.append(script)
        
        if missing_scripts:
            self.log(f"Missing pipeline scripts: {', '.join(missing_scripts)}", "ERROR")
            return False
        
        self.log("All prerequisites satisfied ✓")
        return True
    
    def run_step_1(self) -> bool:
        """Run Step 1: Data Ingestion"""
        self.log("=" * 60)
        self.log("STEP 1: Data Ingestion")
        self.log("=" * 60)
        
        try:
            # Import and run the ingestion script
            sys.path.insert(0, ".")
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("step1", "1_dual_class_csv_to_json_converter.py")
            step1_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(step1_module)
            
            # Run the main function
            step1_module.main()
            
            # Check if output file was created
            output_file = "staging/1_dual_class_output.json"
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    total_companies = data.get('total_companies', 0)
                    companies_with_cik = data.get('companies_with_cik', 0)
                    
                self.log(f"✓ Processed {total_companies} companies")
                self.log(f"✓ Successfully mapped {companies_with_cik} companies to CIKs")
                self.step_outputs[1] = {
                    "total_companies": total_companies,
                    "companies_with_cik": companies_with_cik,
                    "output_file": output_file
                }
                return True
            else:
                self.log("Output file not created", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Step 1 failed: {e}", "ERROR")
            return False
    
    def run_step_1_75(self) -> bool:
        """Run Step 1.75: Missing CIK Resolution (conditional)"""
        self.log("=" * 60)
        self.log("STEP 1.75: Missing CIK Resolution")
        self.log("=" * 60)
        
        # Check if we need to run this step
        if not os.path.exists("staging/1_dual_class_output.json"):
            self.log("staging/1_dual_class_output.json not found, skipping step 1.75")
            return True
        
        try:
            with open("staging/1_dual_class_output.json", 'r') as f:
                data = json.load(f)
            
            companies_without_cik = [c for c in data.get('companies', []) if not c.get('cik')]
            
            if not companies_without_cik:
                self.log("All companies have CIKs, skipping missing CIK resolution")
                return True
            
            self.log(f"Found {len(companies_without_cik)} companies without CIKs")
            
            # Create input file for step 1.75
            nocik_data = data.copy()
            nocik_data['companies'] = companies_without_cik
            
            with open("staging/1.75_dual_class_output_nocik.json", 'w') as f:
                json.dump(nocik_data, f, indent=2)
            
            # Run step 1.75
            cmd = ["python", "1.75_missing_company_investigator.py"]
            if self.ai_check:
                cmd.append("--ai-check")
            
            # Set encoding environment for subprocess
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, 
                                  encoding='utf-8', errors='replace', env=env)  # 30 min timeout
            
            if result.returncode == 0:
                self.log("✓ Missing CIK resolution completed")
                
                # Check results
                if os.path.exists("staging/1.75_dual_class_output_nocik.json"):
                    with open("staging/1.75_dual_class_output_nocik.json", 'r') as f:
                        nocik_data = json.load(f)
                        
                    resolved_companies = sum(1 for c in nocik_data.get('companies', []) if c.get('cik'))
                    self.log(f"✓ Resolved {resolved_companies} additional CIKs")
                    
                    # Merge results back into main file
                    self._merge_nocik_results(data, nocik_data)
                
                return True
            else:
                # Clean up stderr output for display
                stderr_msg = result.stderr or "Unknown error"
                try:
                    clean_stderr = stderr_msg.encode('ascii', errors='replace').decode('ascii')
                except:
                    clean_stderr = "Error message contains invalid characters"
                self.log(f"Step 1.75 failed: {clean_stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("Step 1.75 timed out after 30 minutes", "ERROR")
            return False
        except Exception as e:
            self.log(f"Step 1.75 failed: {e}", "ERROR")
            return False

    # --- New per-CIK steps for 10-K / 10-Q / Normalization ---
    def _run_subprocess_stream(self, cmd: List[str], env: Optional[Dict[str, str]] = None) -> bool:
        self.log(f"Running: {' '.join(cmd)}")
        try:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1, encoding='utf-8', errors='replace') as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    try:
                        print(line.rstrip())
                    except UnicodeEncodeError:
                        print(line.rstrip().encode('ascii', 'replace').decode('ascii'))
                ret = proc.wait()
                if ret != 0:
                    self.log(f"Command failed with return code {ret}", "ERROR")
                    return False
            return True
        except Exception as e:
            self.log(f"Command failed: {e}", "ERROR")
            return False

    def run_download_10k(self, cik: str) -> bool:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return self._run_subprocess_stream([_python_exe(), "-u", "1.5_Download10K.py", "--cik", cik], env)

    def run_extract_10k(self, cik: str) -> bool:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return self._run_subprocess_stream([_python_exe(), "-u", "1.6_Extract10K.py", "--cik", cik], env)

    def run_download_10q(self, cik: str) -> bool:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return self._run_subprocess_stream([_python_exe(), "-u", "1.51_Download10Q.py", "--cik", cik], env)

    def run_extract_10q(self, cik: str) -> bool:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return self._run_subprocess_stream([_python_exe(), "-u", "1.61_Extract10Q.py", "--cik", cik], env)

    def run_normalize_equity(self, cik: str) -> bool:
        # Ensure OpenAI present and key valid (LLM is required for this step)
        if not _ensure_openai_installed(self.log):
            self.log("❌ openai package not installed; run: pip install openai", "ERROR")
            return False
        if not _get_valid_openai_key():
            self.log("❌ OPENAI_API_KEY missing/placeholder. Set in environment or .env", "ERROR")
            return False
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return self._run_subprocess_stream([_python_exe(), "-u", "2_RetrieveData.py", "--cik", cik], env)

    def _get_target_ciks(self) -> List[str]:
        # If a single CIK passed on CLI, use it
        if self.cik:
            return [self.cik]
        # Otherwise, read from staging output of Step 1
        ciks: List[str] = []
        try:
            if os.path.exists("staging/1_dual_class_output.json"):
                with open("staging/1_dual_class_output.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for comp in data.get('companies', []):
                    cik = (comp.get('cik') or '').strip()
                    if cik:
                        ciks.append(cik)
        except Exception as e:
            self.log(f"Warning: failed to load CIK list from staging: {e}", "ERROR")
        # Limit in test mode
        if self.test_mode and self.test_count:
            ciks = ciks[: max(1, int(self.test_count))]
        return ciks

    def run_pipeline(self, skip_steps: List = None, resume_from: int = 1) -> bool:
        """Run the complete pipeline"""
        skip_steps = skip_steps or []
        
        self.log("🚀 Starting LookThroughProfits Dual-Class Share Normalization Pipeline")
        self.log(f"Mode: {'Test' if self.test_mode else 'Full'}")
        self.log(f"Economic Analysis: {'SEC Filing Download' if self.no_open_question else 'OpenQuestion (Direct AI)'}")
        self.log(f"AI Check: {'Enabled' if self.ai_check else 'Disabled'}")
        self.log(f"Output Directory: {self.output_dir}")
        if self.cik:
            self.log(f"Target CIK: {self.cik}")
        if self.with_10q:
            self.log("Including 10-Q download/extraction in normalization flow")
        
        # Check prerequisites
        if not self.check_prerequisites(skip_steps):
            return False
        
        start_time = time.time()
        
        # Define step execution order (modern)
        step_order: List[Any] = [1, 1.75, 1.5, 1.6]
        if self.with_10q:
            step_order.extend([1.51, 1.61])
        step_order.extend([2, 3])
        
        # Filter steps based on resume_from and skip_steps
        steps_to_run = [s for s in step_order if (isinstance(s, (int, float)) and s >= resume_from and s not in skip_steps)]
        
        self.log(f"Steps to run: {steps_to_run}")
        
        # Execute steps
        for step_num in steps_to_run:
            step_info = self.steps[step_num]
            self.log(f"\n🔄 Starting {step_info['name']}...")
            success = False
            
            # Global steps
            if step_num == 1:
                success = self.run_step_1()
            elif step_num == 1.75:
                success = self.run_step_1_75()
            
            # Per-CIK steps
            elif step_num in (1.5, 1.6, 1.51, 1.61, 2):
                ciks = self._get_target_ciks()
                if not ciks:
                    self.log("No CIKs provided or found in staging. Skipping per-CIK steps.", "ERROR")
                    success = False
                else:
                    all_ok = True
                    for cik in ciks:
                        self.log(f"→ CIK {cik} :: {step_info['name']}")
                        if step_num == 1.5:
                            ok = self.run_download_10k(cik)
                        elif step_num == 1.6:
                            ok = self.run_extract_10k(cik)
                        elif step_num == 1.51:
                            ok = self.run_download_10q(cik)
                        elif step_num == 1.61:
                            ok = self.run_extract_10q(cik)
                        elif step_num == 2:
                            ok = self.run_normalize_equity(cik)
                        else:
                            ok = True
                        all_ok = all_ok and ok
                        if not ok:
                            self.log(f"Step {step_info['name']} failed for CIK {cik}", "ERROR")
                            # Continue to next CIK but mark failure
                    success = all_ok
            
            elif step_num == 3:
                success = self.run_step_3()
            
            if success:
                self.completed_steps.add(step_num)
                self.log(f"✅ {step_info['name']} completed successfully")
            else:
                self.failed_steps.add(step_num)
                self.log(f"❌ {step_info['name']} failed")
                if step_info.get('required', False):
                    self.log("Required step failed, stopping pipeline", "ERROR")
                    return False
        
        elapsed = time.time() - start_time
        
        # Generate summary
        self.log("\n" + "=" * 60)
        self.log("📊 PIPELINE SUMMARY")
        self.log("=" * 60)
        self.log(f"⏱️  Total time: {elapsed/60:.1f} minutes")
        self.log(f"✅ Completed steps: {sorted(self.completed_steps, key=lambda x: (float(x)))})")
        if self.failed_steps:
            self.log(f"❌ Failed steps: {sorted(self.failed_steps, key=lambda x: (float(x)))})")
        
        # Output file locations
        self.log("\n📁 Output Files:")
        if os.path.exists("staging/1_dual_class_output.json"):
            self.log(f"  • staging/1_dual_class_output.json - Main dataset with CIKs")
        if os.path.exists("results/dual_class_economic_weights.json"):
            self.log(f"  • results/dual_class_economic_weights.json - Final economic weights")
        if os.path.exists("results/dual_class_economic_weights_test.json"):
            self.log(f"  • results/dual_class_economic_weights_test.json - Test results")
        # Example normalized outputs
        out_dir = Path("fileoutput/equity_classes")
        if out_dir.exists():
            try:
                examples = sorted([p.name for p in out_dir.glob("cik_*.json")])[:5]
                for ex in examples:
                    self.log(f"  • fileoutput/equity_classes/{ex}")
            except Exception:
                pass
        
        # Success criteria
        success = 1 in self.completed_steps and 2 in self.completed_steps and 3 in self.completed_steps
        
        if success:
            self.log("\n🎉 Pipeline completed successfully!")
        else:
            self.log("\n⚠️  Pipeline completed with issues")
        
        return success

    def run_step_3(self) -> bool:
        """Run Step 3: Economic Weight Analysis"""
        self.log("STEP 3: Economic Weight Analysis", "INFO")
        
        # Choose script based on no_open_question flag
        script_name = "3_placeholder_economic_weight_analyzer.py" if self.no_open_question else "3_ai_powered_financial_analyzer.py"
        mode_name = "SEC Filing Download" if self.no_open_question else "OpenQuestion (Direct AI)"
        
        self.log(f"Using mode: {mode_name}")
        
        # Ensure OpenAI present and key valid (LLM is required for this step)
        if not _ensure_openai_installed(self.log):
            self.log("❌ openai package not installed; run: pip install openai", "ERROR")
            return False
        if not _get_valid_openai_key():
            self.log("❌ OPENAI_API_KEY missing/placeholder. Set in environment or .env", "ERROR")
            return False
        try:
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            
            cmd = [_python_exe(), "-u", script_name]
            if self.test_mode:
                cmd.extend(["--test", str(self.test_count or 3)])
            self.log(f"Running: {' '.join(cmd)}")
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1, encoding='utf-8', errors='replace') as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    try:
                        print(line.rstrip())
                    except Exception:
                        print(line.rstrip().encode('ascii', 'replace').decode('ascii'))
                ret = proc.wait()
                if ret != 0:
                    self.log(f"Step 3 failed with return code {ret}", "ERROR")
                    return False
            self.log("✅ Economic Weight Analysis completed", "INFO")
            return True
        except Exception as e:
            self.log(f"Step 3 failed: {e}", "ERROR")
            return False
    
    def _merge_nocik_results(self, main_data: Dict, nocik_data: Dict):
        """Merge results from nocik resolution back into main data"""
        try:
            # Create lookup for nocik companies by name/ticker
            nocik_lookup = {}
            for company in nocik_data.get('companies', []):
                key = (company.get('company_name', ''), company.get('primary_ticker', ''))
                nocik_lookup[key] = company
            
            # Update main data with resolved CIKs
            updated_count = 0
            for company in main_data.get('companies', []):
                if not company.get('cik'):
                    key = (company.get('company_name', ''), company.get('primary_ticker', ''))
                    if key in nocik_lookup:
                        nocik_company = nocik_lookup[key]
                        if nocik_company.get('cik'):
                            company['cik'] = nocik_company['cik']
                            company['cik_resolution'] = {
                                'reason': nocik_company.get('reason'),
                                'reason_detail': nocik_company.get('reason_detail'),
                                'AI_REASON': nocik_company.get('AI_REASON')
                            }
                            updated_count += 1
            
            # Update company count
            main_data['companies_with_cik'] = sum(1 for c in main_data.get('companies', []) if c.get('cik'))
            
            # Save updated main file
            with open("staging/1_dual_class_output.json", 'w') as f:
                json.dump(main_data, f, indent=2)
            
            self.log(f"✓ Merged {updated_count} resolved CIKs into main dataset")
            
        except Exception as e:
            self.log(f"Failed to merge nocik results: {e}", "ERROR")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LookThroughProfits Dual-Class Share Normalization Pipeline")
    # Allow --test optional integer. If provided without value, default to 3.
    parser.add_argument("--test", nargs='?', const=3, type=int, help="Run in test mode with optional count (default 3). Example: --test 5")
    parser.add_argument("--ai-check", action="store_true", help="Enable AI investigation for missing CIKs")
    parser.add_argument("--noOpenQuestion", action="store_true", help="Use original SEC filing download instead of OpenQuestion mode")
    parser.add_argument("--skip-step", type=float, action="append", help="Skip specific steps (can be used multiple times)")
    parser.add_argument("--resume-from", type=float, default=1, help="Resume pipeline from specific step (supports decimals like 1.5)")
    parser.add_argument("--output-dir", default=".", help="Output directory for results")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    parser.add_argument("--cik", help="Run downstream steps for a single CIK (e.g., 0001799448)")
    parser.add_argument("--with-10q", action="store_true", help="Include 10-Q download/extraction (Steps 1.51 and 1.61)")
    
    args = parser.parse_args()
    
    # Determine test_mode and optional count
    test_count = args.test  # will be None if not provided, or an int (const 3 if --test alone)
    test_mode_flag = test_count is not None
    
    # Initialize controller
    controller = PipelineController(
        output_dir=args.output_dir,
        test_mode=test_mode_flag,
        test_count=test_count,
        ai_check=args.ai_check,
        verbose=not args.quiet,
        no_open_question=args.noOpenQuestion,
        cik=args.cik,
        with_10q=args.with_10q
    )
    
    # Run pipeline
    success = controller.run_pipeline(
        skip_steps=args.skip_step or [],
        resume_from=args.resume_from
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
