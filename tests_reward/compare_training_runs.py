"""
compare_training_runs.py

Parses real training log files (the log_seed_*.txt / Episode-line
format produced by train loop, e.g. "Episode 12 | reward: 355.5 |
steps: 142 | eps: 0.930 | best: 355.5") and prints summary stats so you
can compare different training runs (e.g. baseline vs reward-shaped
vs different learning rate) side by side.

This gives real numbers to put in Chapter 6 (System Testing) as
evidence that your reward shaping change actually improved training,
not just that training "ran without crashing".

USAGE:
    python compare_training_runs.py log_seed_123.txt log_seed_456.txt ...

You can pass as many log files as you want, and label them by renaming
the RUN_LABELS dict below, or just read the filenames as-is.
"""

import re
import sys
import statistics

EPISODE_RE = re.compile(
    r"Episode\s+(\d+)\s*\|\s*reward:\s*(-?[\d.]+)\s*\|\s*steps:\s*(\d+)\s*\|\s*eps:\s*([\d.]+)\s*\|\s*best:\s*(-?[\d.]+)"
)


def parse_log(path):
    episodes = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = EPISODE_RE.search(line)
            if m:
                ep, reward, steps, eps, best = m.groups()
                episodes.append({
                    "episode": int(ep),
                    "reward": float(reward),
                    "steps": int(steps),
                    "epsilon": float(eps),
                    "best": float(best),
                })
    return episodes


def summarize(path, episodes, last_n=20):
    if not episodes:
        print(f"{path}: no Episode lines found (check the file/format)")
        return

    all_rewards = [e["reward"] for e in episodes]
    tail_rewards = [e["reward"] for e in episodes[-last_n:]]

    print(f"\n=== {path} ===")
    print(f"  total episodes logged     : {len(episodes)}")
    print(f"  best reward ever          : {max(e['best'] for e in episodes):.1f}")
    print(f"  average reward (all eps)  : {statistics.mean(all_rewards):.1f}")
    print(f"  average reward (last {last_n:>2}) : {statistics.mean(tail_rewards):.1f}")
    print(f"  reward std dev (last {last_n:>2}) : {statistics.pstdev(tail_rewards):.1f}")
    print(f"  final epsilon              : {episodes[-1]['epsilon']:.3f}")


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        print("Usage: python compare_training_runs.py <log_file_1> <log_file_2> ...")
        sys.exit(1)

    for path in files:
        episodes = parse_log(path)
        summarize(path, episodes)
