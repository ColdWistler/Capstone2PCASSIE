#!/usr/bin/env python3
"""
UAT — User Acceptance Test
Runs Godot headless test mode and validates test_report.txt output.
"""

import os
import re
import subprocess
import sys

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

def parse_test_report(path: str) -> dict:
    data = {
        "avg_reward": 0.0,
        "min_reward": 0.0,
        "max_reward": 0.0,
        "avg_steps": 0.0,
        "landings": 0,
        "crashes": 0,
        "total_episodes": 0,
        "episodes": [],
    }
    if not os.path.exists(path):
        return data

    with open(path) as f:
        lines = f.readlines()

    ep_pattern = re.compile(
        r"(\d+)\s+([-\d.]+)\s+(\d+)\s+(LANDED|CRASH|TIMEOUT)\s+([-\d.]+)\s+(\d+)"
    )
    for line in lines:
        m = ep_pattern.search(line)
        if m:
            ep = {
                "num": int(m.group(1)),
                "reward": float(m.group(2)),
                "steps": int(m.group(3)),
                "status": m.group(4),
                "avg_alt": float(m.group(5)),
                "stalls": int(m.group(6)),
            }
            data["episodes"].append(ep)
            if ep["status"] == "LANDED":
                data["landings"] += 1
            elif ep["status"] == "CRASH":
                data["crashes"] += 1

    summary_pattern = re.compile(r"Avg Reward:\s+([-\d.]+)\s+\(min\s+([-\d.]+),\s+max\s+([-\d.]+)\)")
    for line in lines:
        m = summary_pattern.search(line)
        if m:
            data["avg_reward"] = float(m.group(1))
            data["min_reward"] = float(m.group(2))
            data["max_reward"] = float(m.group(3))

    steps_pattern = re.compile(r"Avg Steps:\s+([-\d.]+)")
    for line in lines:
        m = steps_pattern.search(line)
        if m:
            data["avg_steps"] = float(m.group(1))

    data["total_episodes"] = len(data["episodes"])
    return data

def run_godot_test(scene: str = "", report_name: str = "test_report.txt"):
    cmd = ["godot", "--headless", "--path", PROJECT_ROOT]
    if scene:
        cmd.append(f"res://example/{scene}")
    cmd.extend(["--", "--test"])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    user_data = os.path.expanduser("~/.local/share/godot/app_userdata/Flight Sim/")
    report_path = os.path.join(user_data, report_name)
    return report_path, result

def run_uat():
    print("=== UAT: Godot Headless Greedy Evaluation ===\n")

    report_path, result = run_godot_test("", "test_report.txt")
    check("UAT-RUN", "Godot test mode executed", result.returncode == 0 or os.path.exists(report_path),
          f"exit_code={result.returncode}")

    if not os.path.exists(report_path):
        alt_path = os.path.join(PROJECT_ROOT, "test_report.txt")
        if os.path.exists(alt_path):
            report_path = alt_path

    if not os.path.exists(report_path):
        check("UAT-REPORT", "Test report file exists", False, report_path)
        return

    check("UAT-REPORT", "Test report file exists", True, report_path)
    data = parse_test_report(report_path)

    check("UAT-EPISODES", "Ran at least 1 test episode",
          data["total_episodes"] >= 1, f"episodes={data['total_episodes']}")

    if data["total_episodes"] > 0:
        check("UAT-REWARD", "Average reward is positive",
              data["avg_reward"] > 0, f"avg={data['avg_reward']:.1f}")

        check("UAT-LANDING", "At least one landing achieved",
              data["landings"] > 0,
              f"landings={data['landings']}/{data['total_episodes']}")

        check("UAT-CRASHES", "Crash rate below 50%",
              data["crashes"] < data["total_episodes"] * 0.5,
              f"crashes={data['crashes']}/{data['total_episodes']}")

        check("UAT-STEPS", "Average steps > 100 (agent survives)",
              data["avg_steps"] > 100,
              f"avg_steps={data['avg_steps']:.0f}")

def main():
    run_uat()

    print(f"\n=== Results: {PASS}/{PASS+FAIL} passed, {FAIL} failed ===")

    report = "=== UAT REPORT ===\n"
    report += f"Passed: {PASS}/{PASS+FAIL}\nFailed: {FAIL}\n\n"
    for r in RESULTS:
        report += r + "\n"

    report_path = os.path.join(PROJECT_ROOT, "uat_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    sys.exit(0 if FAIL == 0 else 1)

if __name__ == "__main__":
    main()
