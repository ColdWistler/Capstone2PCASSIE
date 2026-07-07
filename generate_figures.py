#!/usr/bin/env python3
"""Generate publication-quality figures for the preliminary results section.

Usage:
  /tmp/plotenv/bin/python3 generate_figures.py

Output: PDF figures in paper/figures/
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = ROOT / "paper" / "figures"

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (5.5, 3.5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def ensure_output():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_latest(pattern, min_size=100):
    files = sorted(ROOT.glob(pattern))
    for f in reversed(files):
        if f.stat().st_size >= min_size:
            return f
    return files[-1] if files else None


# ── Data Loaders ───────────────────────────────────────────────────────

def load_telemetry_csv(path):
    """Load telemetry CSV (telemetry/csv/telemetry_*.csv)."""
    if not path or not path.exists() or path.stat().st_size == 0:
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def load_resource_csv(path):
    """Load resource monitor CSV (resources_*.csv)."""
    if not path or not path.exists() or path.stat().st_size == 0:
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def load_telemetry_jsonl(path):
    """Load telemetry JSONL (telemetry/telemetry.jsonl)."""
    if not path or not path.exists():
        return None
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data if data else None


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def load_godot_log(path):
    """Parse episode data from Godot log as fallback when no telemetry CSV exists."""
    if not path or not path.exists():
        return []
    import re
    with open(path) as f:
        text = f.read()
    pattern = r'Episode (\d+) \| reward: ([\d\.\-]+) \| steps: (\d+) \| eps: ([\d\.]+) \| best: ([\d\.\-]+)'
    rows = []
    for m in re.finditer(pattern, text):
        rows.append({
            "episode": m.group(1),
            "reward": m.group(2),
            "step_count": m.group(3),
            "epsilon": m.group(4),
        })
    return rows


def episodes_from_telemetry(rows):
    """Aggregate telemetry rows into per-episode stats."""
    eps = {}
    for r in rows:
        ep = int(safe_float(r.get("episode", 0)))
        if ep not in eps:
            eps[ep] = {"rewards": [], "steps": 0, "epsilons": [], "stalls": 0, "altitudes": []}
        eps[ep]["rewards"].append(safe_float(r.get("reward", 0)))
        eps[ep]["steps"] += 1
        eps[ep]["epsilons"].append(safe_float(r.get("epsilon", 1)))
        eps[ep]["stalls"] += int(safe_float(r.get("stall", 0)))
        eps[ep]["altitudes"].append(safe_float(r.get("alt_m", 0)))
    return eps


# ── Figure Generators ──────────────────────────────────────────────────

def fig_reward_curve(rows, fname="fig_reward_curve.pdf"):
    """Episode vs cumulative reward with smoothing."""
    eps = episodes_from_telemetry(rows)
    if not eps:
        return False

    episodes = sorted(eps.keys())
    rewards = [sum(eps[e]["rewards"]) for e in episodes]
    n = len(rewards)

    # moving average
    window = max(1, n // 20)
    kernel = np.ones(window) / window
    smoothed = np.convolve(rewards, kernel, mode="valid")
    pad = n - len(smoothed)
    x_smooth = episodes[pad // 2: n - (pad - pad // 2)]

    fig, ax = plt.subplots()
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", linewidth=0.8, label="Raw")
    ax.plot(x_smooth, smoothed, color="steelblue", linewidth=1.8, label=f"Smoothed (w={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.set_title("Training Reward Convergence")
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    return True


def fig_epsilon(rows, fname="fig_epsilon.pdf"):
    """Episode vs epsilon (exploration rate)."""
    eps = episodes_from_telemetry(rows)
    if not eps:
        return False

    episodes = sorted(eps.keys())
    epsilons = [np.mean(eps[e]["epsilons"]) for e in episodes]

    fig, ax = plt.subplots()
    ax.plot(episodes, epsilons, color="coral", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Schedule")
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    return True


def fig_episode_duration(rows, fname="fig_episode_duration.pdf"):
    """Episode vs steps survived."""
    eps = episodes_from_telemetry(rows)
    if not eps:
        return False

    episodes = sorted(eps.keys())
    steps = [eps[e]["steps"] for e in episodes]

    fig, ax = plt.subplots()
    ax.bar(episodes, steps, width=0.8, color="seagreen", alpha=0.7, edgecolor="seagreen", linewidth=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps Survived")
    ax.set_title("Episode Duration Over Training")
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    return True


def fig_flight_trace(rows, fname="fig_flight_trace.pdf"):
    """Altitude vs time for the last complete episode."""
    eps = episodes_from_telemetry(rows)
    if not eps:
        return False

    last_ep = max(eps.keys())
    ep_rows = [r for r in rows if int(safe_float(r.get("episode", 0))) == last_ep]
    if len(ep_rows) < 10:
        return False

    t0 = safe_float(ep_rows[0].get("t_ms", 0))
    times = [(safe_float(r.get("t_ms", 0)) - t0) / 1000 for r in ep_rows]
    alts = [safe_float(r.get("alt_m", 0)) for r in ep_rows]

    fig, ax = plt.subplots()
    ax.plot(times, alts, color="steelblue", linewidth=1.2)
    ax.axhline(200, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Target 200 m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Altitude Trace (Episode {last_ep})")
    ax.legend()
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    return True


def fig_flight_path(rows, fname="fig_flight_path.pdf"):
    """Top-down XZ flight path."""
    if not rows:
        return False

    # use middle episode with enough data
    eps = episodes_from_telemetry(rows)
    if not eps:
        return False

    candidates = [(ep, eps[ep]["steps"]) for ep in eps if eps[ep]["steps"] > 100]
    if not candidates:
        return False

    target_ep = sorted(candidates, key=lambda x: -x[1])[0][0]
    ep_rows = [r for r in rows if int(safe_float(r.get("episode", 0))) == target_ep]

    xs = [safe_float(r.get("px", 0)) for r in ep_rows]
    zs = [safe_float(r.get("pz", 0)) for r in ep_rows]

    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(xs)))
    ax.scatter(xs, zs, c=colors, s=8, alpha=0.6, edgecolors="none")
    ax.plot(xs, zs, color="steelblue", linewidth=0.8, alpha=0.5)
    ax.scatter(xs[0], zs[0], c="green", s=40, marker="o", zorder=5, label="Start")
    ax.scatter(xs[-1], zs[-1], c="red", s=40, marker="x", zorder=5, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Flight Path (Episode {target_ep})")
    ax.legend()
    ax.set_aspect("equal")
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    return True


def fig_action_distribution(rows, fname="fig_action_dist.pdf"):
    """Histogram of action selections."""
    if not rows:
        return False

    # first vs last half
    half = len(rows) // 2
    first_half = [int(safe_float(r.get("action", 0))) for r in rows[:half]]
    last_half = [int(safe_float(r.get("action", 0))) for r in rows[half:]]

    action_labels = ["None", "Pitch Up", "Pitch Dn", "Roll L", "Roll R", "Thr Up", "Thr Dn"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True)
    bins = np.arange(-0.5, 7.5, 1)

    ax1.hist(first_half, bins=bins, color="coral", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(7))
    ax1.set_xticklabels(action_labels, rotation=45, ha="right", fontsize=7)
    ax1.set_title("Early Training")
    ax1.set_xlabel("Action")
    ax1.set_ylabel("Count")

    ax2.hist(last_half, bins=bins, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(action_labels, rotation=45, ha="right", fontsize=7)
    ax2.set_title("Late Training")
    ax2.set_xlabel("Action")

    fig.suptitle("Action Selection Distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    return True


def fig_stall_events(rows, fname="fig_stall_events.pdf"):
    """Stalls per episode over training."""
    eps = episodes_from_telemetry(rows)
    if not eps:
        return False

    episodes = sorted(eps.keys())
    stalls = [eps[e]["stalls"] for e in episodes]

    fig, ax = plt.subplots()
    ax.bar(episodes, stalls, width=0.8, color="tomato", alpha=0.6, edgecolor="tomato", linewidth=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Stall Events")
    ax.set_title("Stall Frequency Over Training")
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    return True


def fig_q_values(rows, fname="fig_q_values.pdf"):
    """Q-value traces over time for a segment of an episode."""
    if not rows:
        return False

    # find a run segment with Q-values
    q_rows = []
    for r in rows:
        if any(f"Q_{i}" in r for i in range(7)):
            q_rows.append(r)
    if len(q_rows) < 10:
        return False

    segment = q_rows[:500]
    t0 = safe_float(segment[0].get("t_ms", 0))
    times = [(safe_float(r.get("t_ms", 0)) - t0) / 1000 for r in segment]

    action_labels = ["None", "Pitch Up", "Pitch Dn", "Roll L", "Roll R", "Thr Up", "Thr Dn"]
    colors = plt.cm.Set1(np.linspace(0, 1, 7))

    fig, ax = plt.subplots()
    for i in range(7):
        key = f"Q_{i}"
        vals = [safe_float(r.get(key, 0)) for r in segment]
        ax.plot(times, vals, color=colors[i], linewidth=0.8, label=action_labels[i])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Q-Value")
    ax.set_title("Q-Value Evolution")
    ax.legend(ncol=2, fontsize=6)
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    return True


def fig_resource_usage(fname="fig_resource_usage.pdf"):
    """CPU, memory, GPU usage during training from resource CSVs."""
    # try to find the largest resource CSV
    candidates = [p for p in sorted(ROOT.glob("resources_*.csv"), key=lambda p: p.stat().st_size, reverse=True) if p.stat().st_size > 100]
    if not candidates:
        return False

    rows = load_resource_csv(candidates[0])
    if not rows:
        return False

    times = list(range(len(rows)))
    cpus = [safe_float(r.get("cpu_pct", 0)) for r in rows]
    mems = [safe_float(r.get("mem_pct", 0)) for r in rows]
    gpus = [safe_float(r.get("gpu_util_pct", 0)) for r in rows]

    fig, ax = plt.subplots()
    ax.plot(times, cpus, color="steelblue", linewidth=1.0, label="CPU %")
    ax.plot(times, mems, color="seagreen", linewidth=1.0, label="Memory %")
    has_gpu = any(g > 0 for g in gpus)
    if has_gpu:
        ax.plot(times, gpus, color="coral", linewidth=1.0, label="GPU %")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Utilization (%)")
    ax.set_title("System Resource Usage During Training")
    ax.legend()
    ax.set_ylim(0, 100)
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    return True


def fig_altitude_boxplot(rows, fname="fig_altitude_boxplot.pdf"):
    """Box plot of altitude per episode group (early/middle/late)."""
    eps = episodes_from_telemetry(rows)
    if not eps:
        return False

    episodes = sorted(eps.keys())
    n = len(episodes)
    if n < 10:
        return False

    third = max(1, n // 3)
    groups = {
        "Early\n(1–{})".format(third): [eps[e]["altitudes"] for e in episodes[:third]],
        "Middle\n({}–{})".format(third + 1, 2 * third): [eps[e]["altitudes"] for e in episodes[third:2 * third]],
        "Late\n({}–{})".format(2 * third + 1, n): [eps[e]["altitudes"] for e in episodes[2 * third:]],
    }

    fig, ax = plt.subplots()
    positions = [1, 2, 3]
    bp = ax.boxplot(
        [np.concatenate(v) if v else [0] for v in groups.values()],
        positions=positions,
        widths=0.5,
        patch_artist=True,
    )
    colors = ["coral", "gold", "steelblue"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_xticklabels(groups.keys())
    ax.set_ylabel("Altitude ASL (m)")
    ax.set_title("Flight Altitude Distribution by Training Phase")
    # Note: these are absolute altitudes (meters above sea level), not AGL.
    # The ~200 m AGL target corresponds to ~700–800 m ASL in this terrain.
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    return True


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ensure_output()

    # Find best data sources
    telemetry_csv = find_latest("telemetry/csv/telemetry_*.csv")
    godot_log = Path.home() / ".local/share/godot/app_userdata/Flight Sim/logs/godot.log"
    resource_csv = find_latest("resources_*.csv")

    rows = load_telemetry_csv(telemetry_csv)
    source_name = ""
    if not rows:
        rows = load_godot_log(godot_log)
        source_name = "Godot log"
    else:
        source_name = telemetry_csv.name if telemetry_csv else ""

    print(f"[1/1] Generating figures...")
    print(f"  Data source: {source_name if rows else 'NOT FOUND'}")
    print(f"  Episodes:    {len(set(r.get('episode', 0) for r in rows)) if rows else 0} unique")
    print(f"  Resource CSV: {resource_csv.name if resource_csv else 'NOT FOUND'}")

    generators = [
        ("Reward Curve", lambda: fig_reward_curve(rows)),
        ("Epsilon Schedule", lambda: fig_epsilon(rows)),
        ("Episode Duration", lambda: fig_episode_duration(rows)),
        ("Flight Trace", lambda: fig_flight_trace(rows)),
        ("Flight Path", lambda: fig_flight_path(rows)),
        ("Action Distribution", lambda: fig_action_distribution(rows)),
        ("Stall Events", lambda: fig_stall_events(rows)),
        ("Q-Values", lambda: fig_q_values(rows)),
        ("Altitude Boxplot", lambda: fig_altitude_boxplot(rows)),
        ("Resource Usage", lambda: fig_resource_usage()),
    ]

    generated = 0
    for name, func in generators:
        try:
            if func():
                print(f"  ✓ {name}")
                generated += 1
            else:
                print(f"  - {name} (skipped — insufficient data)")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print(f"\nDone — {generated} figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
