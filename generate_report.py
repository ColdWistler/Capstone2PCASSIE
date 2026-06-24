#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).parent.resolve()
TELEMETRY_DIR = PROJECT_ROOT / "telemetry"
DEFAULT_TELEMETRY_PATH = TELEMETRY_DIR / "telemetry.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "reports"


def read_telemetry(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path) as f:
            lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return data
    except (OSError, IOError):
        return []


def read_test_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        text = path.read_text()
    except (OSError, IOError):
        return None

    summary = {}
    m = re.search(r"Avg Reward:\s+([\d\.\-]+)\s+\(min ([\d\.\-]+), max ([\d\.\-]+)\)", text)
    if m:
        summary["avg_reward"] = float(m.group(1))
        summary["min_reward"] = float(m.group(2))
        summary["max_reward"] = float(m.group(3))

    m = re.search(r"Avg Steps:\s+([\d\.]+)\s*/\s*(\d+)", text)
    if m:
        summary["avg_steps"] = float(m.group(1))
        summary["max_steps"] = int(m.group(2))

    m = re.search(r"Avg Alt:\s+([\d\.]+)\s+m", text)
    if m:
        summary["avg_alt"] = float(m.group(1))

    m = re.search(r"Avg Stalls:\s+([\d\.]+)", text)
    if m:
        summary["avg_stalls"] = float(m.group(1))

    m = re.search(r"Landings:\s+(\d+)\s*/\s*(\d+)\s*\(([\d\.]+)%\)", text)
    if m:
        summary["landings"] = int(m.group(1))
        summary["total_episodes"] = int(m.group(2))
        summary["landing_rate"] = float(m.group(3))

    m = re.search(r"Crashes:\s+(\d+)\s*/\s*(\d+)", text)
    if m:
        summary["crashes"] = int(m.group(1))

    episodes = []
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 6:
            try:
                ep_num = int(parts[0].rstrip(":"))
                reward = float(parts[1])
                steps = int(parts[2])
                status = parts[3]
                avg_alt = float(parts[4])
                stalls = int(parts[5])
                episodes.append({
                    "episode": ep_num, "reward": reward, "steps": steps,
                    "status": status, "avg_alt": avg_alt, "stalls": stalls,
                })
            except (ValueError, IndexError):
                continue

    return {"summary": summary, "episodes": episodes} if summary else None


def compute_ground_speed(data: list[dict]) -> list[dict]:
    out = []
    prev_pos = None
    prev_t = None
    for item in data:
        t = item.get("t", 0) / 1000.0
        px, py, pz = item.get("px", 0), item.get("py", 0), item.get("pz", 0)
        gs = 0.0
        if prev_pos is not None and prev_t is not None:
            dt = t - prev_t
            if dt > 0:
                gs = (((px - prev_pos[0]) ** 2 + (pz - prev_pos[2]) ** 2) ** 0.5) / dt
        item["ground_speed"] = gs
        prev_pos = (px, py, pz)
        prev_t = t
        out.append(item)
    return out


def build_telemetry_section(df: pd.DataFrame) -> str:
    sections_html = ""

    t_rel = df["t_rel"]

    # Altitude
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_rel, y=df["alt"], mode="lines", name="Altitude (m)"))
    fig.update_layout(title="Altitude vs Time", xaxis_title="Time (s)", yaxis_title="Altitude (m)", height=350)
    sections_html += fig.to_html(full_html=False, include_plotlyjs="cdn") + "\n"

    # Speed
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=t_rel, y=df["spd"], mode="lines", name="Airspeed (m/s)"))
    if "ground_speed" in df.columns:
        fig.add_trace(go.Scatter(x=t_rel, y=df["ground_speed"], mode="lines", name="Ground Speed (m/s)"))
    fig.add_trace(go.Scatter(x=t_rel, y=df["vspd"], mode="lines", name="Vert. Speed (m/s)", yaxis="y2"))
    fig.update_layout(title="Speed vs Time", xaxis_title="Time (s)", height=350)
    fig.update_yaxes(title_text="m/s", secondary_y=False)
    fig.update_yaxes(title_text="m/s", secondary_y=True)
    sections_html += fig.to_html(full_html=False, include_plotlyjs="cdn") + "\n"

    # Attitude
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "roll" in df.columns:
        fig.add_trace(go.Scatter(x=t_rel, y=df["roll"], mode="lines", name="Roll (°)"))
    if "pitch" in df.columns:
        fig.add_trace(go.Scatter(x=t_rel, y=df["pitch"], mode="lines", name="Pitch (°)"))
    if "hdg" in df.columns:
        fig.add_trace(go.Scatter(x=t_rel, y=df["hdg"], mode="lines", name="Heading (°)", yaxis="y2"))
    fig.update_layout(title="Attitude vs Time", xaxis_title="Time (s)", height=350)
    fig.update_yaxes(title_text="Degrees", secondary_y=False)
    fig.update_yaxes(title_text="Heading (°)", secondary_y=True)
    sections_html += fig.to_html(full_html=False, include_plotlyjs="cdn") + "\n"

    # Engine & Fuel
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "epow" in df.columns:
        fig.add_trace(go.Scatter(x=t_rel, y=df["epow"] * 100, mode="lines", name="Engine Power (%)"))
    if "eact" in df.columns:
        fig.add_trace(go.Scatter(x=t_rel, y=df["eact"] * 100, mode="lines", name="Engine Active"))
    if "fuel" in df.columns:
        fig.add_trace(go.Scatter(x=t_rel, y=df["fuel"] * 100, mode="lines", name="Fuel Level (%)", yaxis="y2"))
    if "flap" in df.columns:
        fig.add_trace(go.Scatter(x=t_rel, y=df["flap"] * 100, mode="lines", name="Flaps (%)", yaxis="y2"))
    fig.update_layout(title="Engine & Fuel vs Time", xaxis_title="Time (s)", height=350)
    fig.update_yaxes(title_text="%", secondary_y=False)
    fig.update_yaxes(title_text="%", secondary_y=True)
    sections_html += fig.to_html(full_html=False, include_plotlyjs="cdn") + "\n"

    # G-Force & Load Factor
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "g" in df.columns:
        fig.add_trace(go.Scatter(x=t_rel, y=df["g"], mode="lines", name="G-Force"))
    if "load" in df.columns:
        fig.add_trace(go.Scatter(x=t_rel, y=df["load"], mode="lines", name="Load Factor", yaxis="y2"))
    fig.update_layout(title="G-Force & Load Factor", xaxis_title="Time (s)", height=350)
    fig.update_yaxes(title_text="g", secondary_y=False)
    fig.update_yaxes(title_text="", secondary_y=True)
    sections_html += fig.to_html(full_html=False, include_plotlyjs="cdn") + "\n"

    # Stall indicator
    if "stall" in df.columns and df["stall"].sum() > 0:
        stall_times = df[df["stall"] == 1]["t_rel"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stall_times, y=[1] * len(stall_times), mode="markers", name="Stall Event"))
        fig.update_layout(
            title="Stall Events", xaxis_title="Time (s)",
            yaxis=dict(showticklabels=False), height=150,
        )
        sections_html += fig.to_html(full_html=False, include_plotlyjs="cdn") + "\n"

    # Flight Path
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top-Down Flight Path", "Vertical Profile"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )
    fig.add_trace(go.Scatter(x=df["pz"], y=df["px"], mode="lines+markers", name="Track"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_rel, y=df["py"], mode="lines", name="Altitude", fill="tozeroy"), row=1, col=2)
    fig.update_xaxes(title_text="Z (m)", row=1, col=1)
    fig.update_yaxes(title_text="X (m)", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Height Y (m)", row=1, col=2)
    fig.update_layout(title="Flight Path", height=450)
    sections_html += fig.to_html(full_html=False, include_plotlyjs="cdn") + "\n"

    return sections_html


def build_test_report_section(test_data: dict) -> str:
    summary = test_data["summary"]
    episodes = test_data["episodes"]

    html = '<div class="section"><h2>Test Report</h2>'

    # Summary table
    html += '<table class="summary-table">'
    html += "<tr><th>Metric</th><th>Value</th></tr>"
    html += f"<tr><td>Avg Reward</td><td>{summary.get('avg_reward', 'N/A'):.1f} (min {summary.get('min_reward', 'N/A'):.1f}, max {summary.get('max_reward', 'N/A'):.1f})</td></tr>"
    html += f"<tr><td>Avg Steps</td><td>{summary.get('avg_steps', 'N/A'):.0f} / {summary.get('max_steps', 'N/A')}</td></tr>"
    html += f"<tr><td>Avg Altitude</td><td>{summary.get('avg_alt', 'N/A'):.0f} m</td></tr>"
    html += f"<tr><td>Avg Stalls / ep</td><td>{summary.get('avg_stalls', 'N/A'):.1f}</td></tr>"
    html += f"<tr><td>Landing Rate</td><td>{summary.get('landings', 'N/A')} / {summary.get('total_episodes', 'N/A')} ({summary.get('landing_rate', 'N/A'):.0f}%)</td></tr>"
    html += f"<tr><td>Crashes</td><td>{summary.get('crashes', 'N/A')} / {summary.get('total_episodes', 'N/A')}</td></tr>"
    html += "</table>"

    # Per-episode bar chart
    if episodes:
        df_ep = pd.DataFrame(episodes)
        status_colors = {"LANDED": "green", "CRASH": "red", "TIMEOUT": "orange"}
        colors = [status_colors.get(s, "gray") for s in df_ep["status"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_ep["episode"], y=df_ep["reward"],
            marker_color=colors,
            text=[f"{s}<br>{r:.1f}" for s, r in zip(df_ep["status"], df_ep["reward"])],
            textposition="outside",
            name="Reward",
        ))
        fig.update_layout(
            title="Per-Episode Rewards (colored by status)",
            xaxis_title="Episode",
            yaxis_title="Reward",
            height=400,
        )
        html += fig.to_html(full_html=False, include_plotlyjs="cdn") + "\n"

        # Steps per episode
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_ep["episode"], y=df_ep["steps"],
            marker_color=colors,
            text=df_ep["steps"],
            textposition="outside",
            name="Steps",
        ))
        fig.add_trace(go.Bar(
            x=df_ep["episode"], y=df_ep["stalls"],
            marker_color="purple",
            text=df_ep["stalls"],
            textposition="outside",
            name="Stalls",
        ))
        fig.update_layout(
            title="Per-Episode Steps & Stalls",
            xaxis_title="Episode",
            yaxis_title="Count",
            height=400,
            barmode="group",
        )
        html += fig.to_html(full_html=False, include_plotlyjs="cdn") + "\n"

    html += "</div>"
    return html


def generate_report(
    telemetry_path: Path = DEFAULT_TELEMETRY_PATH,
    test_report_path: Path | None = None,
    output_name: str = "report.html",
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name

    data = read_telemetry(telemetry_path)
    data = compute_ground_speed(data)

    test_data = None
    if test_report_path and test_report_path.exists():
        test_data = read_test_report(test_report_path)

    sections = []

    if data:
        df = pd.DataFrame(data)
        t_sec = df["t"] / 1000.0
        t_start = t_sec.iloc[0]
        df["t_rel"] = t_sec - t_start

        total_time = t_sec.iloc[-1] - t_start
        avg_alt = df["alt"].mean()
        max_alt = df["alt"].max()
        avg_spd = df["spd"].mean()
        max_spd = df["spd"].max()
        stalls = int(df["stall"].sum()) if "stall" in df.columns else 0

        # Overview stats
        overview = f"""\
<div class="section">
  <h2>Flight Overview</h2>
  <table class="summary-table">
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total Time</td><td>{total_time:.1f} s</td></tr>
    <tr><td>Samples</td><td>{len(df)}</td></tr>
    <tr><td>Avg Altitude</td><td>{avg_alt:.1f} m</td></tr>
    <tr><td>Max Altitude</td><td>{max_alt:.1f} m</td></tr>
    <tr><td>Avg Airspeed</td><td>{avg_spd:.1f} m/s</td></tr>
    <tr><td>Max Airspeed</td><td>{max_spd:.1f} m/s</td></tr>
    <tr><td>Stall Events</td><td>{stalls}</td></tr>
  </table>
</div>"""
        sections.append(overview)

        sections.append('<div class="section"><h2>Telemetry Graphs</h2>')
        sections.append(build_telemetry_section(df))
        sections.append("</div>")
    else:
        sections.append('<div class="section"><p class="warn">No telemetry data found.</p></div>')

    if test_data:
        sections.append(build_test_report_section(test_data))

    # Training progress from test report
    if test_data:
        summary = test_data["summary"]
        html = '<div class="section"><h2>Training Summary</h2><table class="summary-table">'
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        html += f"<tr><td>Episodes Evaluated</td><td>{summary.get('total_episodes', 'N/A')}</td></tr>"
        html += f"<tr><td>Landing Success Rate</td><td>{summary.get('landing_rate', 'N/A'):.1f}%</td></tr>"
        html += f"<tr><td>Avg Reward</td><td>{summary.get('avg_reward', 'N/A'):.1f}</td></tr>"
        html += f"<tr><td>Avg Steps / Episode</td><td>{summary.get('avg_steps', 'N/A'):.0f}</td></tr>"
        html += f"<tr><td>Avg Altitude Held</td><td>{summary.get('avg_alt', 'N/A'):.0f} m</td></tr>"
        html += f"<tr><td>Crashes</td><td>{summary.get('crashes', 'N/A')}</td></tr>"
        html += "</table></div>"
        sections.append(html)

    full_html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Flight Sim Report</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #f5f5f5; color: #333; padding: 20px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ text-align: center; margin-bottom: 10px; color: #1a1a2e; }}
    .subtitle {{ text-align: center; color: #666; margin-bottom: 30px; }}
    .section {{ background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    h2 {{ margin-bottom: 15px; color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; }}
    table.summary-table {{ width: 100%; border-collapse: collapse; margin-bottom: 15px; }}
    table.summary-table th, table.summary-table td {{ padding: 8px 12px; text-align: left;
      border-bottom: 1px solid #e0e0e0; }}
    table.summary-table th {{ background: #f8f9fa; font-weight: 600; }}
    tr:hover {{ background: #f1f3f5; }}
    .warn {{ color: #856404; background: #fff3cd; padding: 10px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Flight Simulator Report</h1>
    <p class="subtitle">Generated from telemetry data</p>
    {"".join(sections)}
  </div>
</body>
</html>"""

    output_path.write_text(full_html)
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate flight sim HTML report with graphs")
    parser.add_argument(
        "-t", "--telemetry", type=str, default=str(DEFAULT_TELEMETRY_PATH),
        help="Path to telemetry JSONL file",
    )
    parser.add_argument(
        "-r", "--test-report", type=str, default=None,
        help="Path to test_report.txt (auto-looked up in Godot user data if not specified)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="report.html",
        help="Output HTML report filename (saved in reports/)",
    )
    args = parser.parse_args()

    telemetry_path = Path(args.telemetry)
    test_report_path = None
    if args.test_report:
        test_report_path = Path(args.test_report)
    else:
        # Common Godot user data paths
        candidates = [
            Path.home() / ".local/share/godot/app_userdata/Flight Sim/test_report.txt",
            Path.home() / "AppData/Roaming/Godot/app_userdata/Flight Sim/test_report.txt",
            Path.home() / "Library/Application Support/Godot/app_userdata/Flight Sim/test_report.txt",
        ]
        for c in candidates:
            if c.exists():
                test_report_path = c
                break

    path = generate_report(telemetry_path, test_report_path, args.output)
    print(f"Report saved to: {path}")


if __name__ == "__main__":
    main()
