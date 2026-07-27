#!/usr/bin/env python3
"""
Master Test Runner — Orchestrates all 4 test levels:
  1. Unit Tests       (cargo test)
  2. Integration Tests (Godot headless FFI runner)
  3. System Tests      (parse training logs)
  4. UAT              (Godot --test greedy evaluation)
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
RESULTS = {}

def run_cmd(name: str, cmd: list, cwd: str = None, timeout: int = 300) -> int:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, timeout=timeout)
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return 1
    except FileNotFoundError:
        print(f"  SKIP — command not found: {cmd[0]}")
        return -1

def main():
    print("Capstone2PCASSIE — Master Test Runner\n")

    # 1. Unit Tests (Rust)
    rc = run_cmd(
        "1. UNIT TESTS (cargo test)",
        ["cargo", "test"],
        cwd=os.path.join(PROJECT_ROOT, "rust_dqn"),
        timeout=180,
    )
    RESULTS["Unit Tests (Rust)"] = "PASS" if rc == 0 else "FAIL"

    # 2. Integration Tests (Godot FFI)
    rc = run_cmd(
        "2. INTEGRATION TESTS (Godot headless FFI)",
        ["godot", "--headless", "res://tests/test_ffi_runner.tscn"],
        timeout=60,
    )
    RESULTS["Integration Tests (Godot FFI)"] = "PASS" if rc == 0 else ("SKIP" if rc == -1 else "FAIL")

    # 3. System Tests (training log validation)
    rc = run_cmd(
        "3. SYSTEM TESTS (training log validation)",
        [sys.executable, os.path.join(TESTS_DIR, "test_system.py")],
        timeout=30,
    )
    RESULTS["System Tests (Logs)"] = "PASS" if rc == 0 else ("SKIP" if rc == -1 else "FAIL")

    # 4. UAT (Godot greedy evaluation) — optional, requires model
    rc = run_cmd(
        "4. UAT (Godot greedy evaluation)",
        [sys.executable, os.path.join(TESTS_DIR, "test_uat.py")],
        timeout=600,
    )
    RESULTS["UAT (Greedy Eval)"] = "PASS" if rc == 0 else ("SKIP" if rc == -1 else "FAIL")

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    for level, status in RESULTS.items():
        icon = "PASS" if status == "PASS" else ("SKIP" if status == "SKIP" else "FAIL")
        print(f"  [{icon}] {level}")

    all_pass = all(v == "PASS" for v in RESULTS.values())
    none_fail = all(v != "FAIL" for v in RESULTS.values())

    if all_pass:
        print("\nAll tests PASSED")
    elif none_fail:
        print("\nTests passed (some skipped due to missing tools)")
    else:
        print("\nSome tests FAILED")

    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
