#!/usr/bin/env python3
"""
System Tests — Training Log Validation
Parses log_seed_*.txt and validates acceptance criteria.
"""

import os
import re
import sys
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS = 0
FAIL = 0
RESULTS = []

def check(test_id: str, description: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        msg = f"[PASS] {test_id}: {description}"
    else:
        FAIL += 1
        msg = f"[FAIL] {test_id}: {description}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    RESULTS.append(msg)

def parse_log(path: str) -> list[dict]:
    episodes = []
    pattern = re.compile(
        r"Episode\s+(\d+)\s*\|\s*reward:\s*([-\d.]+)\s*\|\s*steps:\s*(\d+)\s*\|\s*eps:\s*([-\d.]+)\s*\|\s*best:\s*([-\d.]+)"
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                episodes.append({
                    "episode": int(m.group(1)),
                    "reward": float(m.group(2)),
                    "steps": int(m.group(3)),
                    "epsilon": float(m.group(4)),
                    "best": float(m.group(5)),
                })
    return episodes

def validate_log(filepath: str, scenario: str, seed: int):
    basename = os.path.basename(filepath)
    episodes = parse_log(filepath)

    check(f"ST-PARSE-{seed}", f"Log {basename} has episodes", len(episodes) > 0,
          f"found {len(episodes)} episodes")

    if len(episodes) == 0:
        return

    n = len(episodes)
    check(f"ST-COUNT-{seed}", f"Log {basename} episode count >= 100", n >= 100,
          f"{n} episodes")

    final_30 = episodes[-min(30, n):]
    rewards = [e["reward"] for e in final_30]
    mean_r = sum(rewards) / len(rewards)

    check(f"ST-REWARD-{seed}", f"Log {basename} final 30 mean reward > 0",
          mean_r > 0, f"mean={mean_r:.1f}")

    steps = [e["steps"] for e in final_30]
    avg_steps = sum(steps) / len(steps)
    check(f"ST-STEPS-{seed}", f"Log {basename} avg steps > 100",
          avg_steps > 100, f"avg_steps={avg_steps:.0f}")

    epsilons = [e["epsilon"] for e in final_30]
    check(f"ST-EPS-{seed}", f"Log {basename} epsilon decays toward 0",
          all(e < 0.1 for e in epsilons),
          f"final_eps={epsilons[-1]:.4f}")

    best = max(rewards)
    check(f"ST-BEST-{seed}", f"Log {basename} best reward in final 30 > mean",
          best >= mean_r, f"best={best:.1f}")

def main():
    print("=== System Tests: Training Log Validation ===\n")

    log_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "log_seed_*.txt")))
    if not log_files:
        print("No log_seed_*.txt files found!")
        sys.exit(1)

    print(f"Found {len(log_files)} log files\n")

    for lf in log_files:
        basename = os.path.basename(lf)
        seed_match = re.search(r"seed_(\d+)", basename)
        seed = int(seed_match.group(1)) if seed_match else 0
        scenario = "Complex" if "complex" in basename.lower() else "Simple"
        validate_log(lf, scenario, seed)

    print(f"\n=== Results: {PASS}/{PASS+FAIL} passed, {FAIL} failed ===")

    report = "=== SYSTEM TEST REPORT ===\n"
    report += f"Passed: {PASS}/{PASS+FAIL}\nFailed: {FAIL}\n\n"
    for r in RESULTS:
        report += r + "\n"

    report_path = os.path.join(PROJECT_ROOT, "system_test_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    sys.exit(0 if FAIL == 0 else 1)

if __name__ == "__main__":
    main()
