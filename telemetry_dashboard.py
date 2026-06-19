import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_TELEMETRY_PATH = PROJECT_ROOT / "telemetry" / "telemetry.jsonl"
MAX_POINTS = 500
REFRESH_MS = 200


def read_telemetry(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path, "r") as f:
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


def main():
    st.set_page_config(
        page_title="Flight Sim Telemetry",
        page_icon="✈",
        layout="wide",
    )

    telemetry_path_str = st.sidebar.text_input(
        "Telemetry file path",
        value=str(DEFAULT_TELEMETRY_PATH),
    )
    telemetry_path = Path(telemetry_path_str)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Controls")
    paused = st.sidebar.checkbox("Pause updates", value=False)
    st.sidebar.markdown(
        "Open Godot scene with a TelemetryExporter node attached, "
        "then launch the flight sim. Telemetry data will appear here."
    )

    if "buffer" not in st.session_state:
        st.session_state.buffer = deque(maxlen=MAX_POINTS)
    if "prev_pos" not in st.session_state:
        st.session_state.prev_pos = None
    if "prev_t" not in st.session_state:
        st.session_state.prev_t = None

    st.title("✈ Flight Simulator Telemetry")

    placeholder_gauges = st.empty()
    placeholder_charts = st.empty()

    while True:
        if not paused:
            raw = read_telemetry(telemetry_path)
            if raw:
                # Find new data since last read
                known_count = len(st.session_state.buffer)
                new_items = raw[known_count:] if known_count < len(raw) else []
                if new_items:
                    # Compute vertical speed from position delta if not set
                    for item in new_items:
                        t = item.get("t", 0) / 1000.0
                        if st.session_state.prev_pos is not None and st.session_state.prev_t is not None:
                            dt = t - st.session_state.prev_t
                            if dt > 0:
                                item["ground_speed"] = (
                                    ((item["px"] - st.session_state.prev_pos[0]) ** 2
                                     + (item["pz"] - st.session_state.prev_pos[2]) ** 2) ** 0.5
                                ) / dt
                            else:
                                item["ground_speed"] = 0.0
                        else:
                            item["ground_speed"] = 0.0
                        st.session_state.prev_pos = (item["px"], item["py"], item["pz"])
                        st.session_state.prev_t = t
                    st.session_state.buffer.extend(new_items)

            data = list(st.session_state.buffer)

            if data:
                _render_gauges(placeholder_gauges, data)
                _render_charts(placeholder_charts, data)
            else:
                placeholder_gauges.info(
                    "Waiting for telemetry data... Make sure the Godot flight sim is running "
                    "with a TelemetryExporter node attached to the scene."
                )
                placeholder_charts.empty()

        time.sleep(REFRESH_MS / 1000.0)
        st.rerun()


def _render_gauges(placeholder, data: list[dict]):
    latest = data[-1]

    cols = placeholder.columns(6)

    _metric(cols[0], "Altitude", f"{latest.get('alt', 0):.1f}", "m")
    _metric(cols[1], "Airspeed", f"{latest.get('spd', 0):.1f}", "m/s")
    _metric(cols[2], "Ground Speed", f"{latest.get('ground_speed', 0):.1f}", "m/s")
    _metric(cols[3], "Vert. Speed", f"{latest.get('vspd', 0):.1f}", "m/s")
    _metric(cols[4], "G-Force", f"{latest.get('g', 0):.2f}", "g")
    _metric(cols[5], "Load Factor", f"{latest.get('load', 0):.2f}", "")

    cols2 = placeholder.columns(6)
    _metric(cols2[0], "Roll", f"{latest.get('roll', 0):.1f}", "°")
    _metric(cols2[1], "Pitch", f"{latest.get('pitch', 0):.1f}", "°")
    _metric(cols2[2], "Heading", f"{latest.get('hdg', 0):.1f}", "°")
    _metric(cols2[3], "Engine", f"{latest.get('epow', 0) * 100:.0f}", "%")
    _metric(cols2[4], "Fuel", f"{latest.get('fuel', 0) * 100:.0f}", "%")
    _metric(cols2[5], "Flaps", f"{latest.get('flap', 0) * 100:.0f}", "%")

    stall = latest.get("stall", 0)
    gear = latest.get("gear", 0)
    eact = latest.get("eact", 0)
    temp = latest.get("temp", None)

    cols3 = placeholder.columns(6)
    cols3[0].metric("Stalled", "YES 🚨" if stall else "NO ✅")
    cols3[1].metric("Landing Gear", "DOWN ⬇" if gear else "UP ⬆")
    cols3[2].metric("Engine", "ON 🔥" if eact else "OFF ⬜")
    if temp is not None:
        cols3[3].metric("Temperature", f"{temp:.1f}", "°C")


def _metric(col, label, value, unit=""):
    suffix = f" {unit}" if unit else ""
    col.metric(label, f"{value}{suffix}")


def _render_charts(placeholder, data: list[dict]):
    df = pd.DataFrame(data)
    if df.empty:
        placeholder.info("No data available for charts")
        return

    t_sec = df["t"] / 1000.0
    if "t_start" not in st.session_state:
        st.session_state.t_start = t_sec.iloc[0]
    t_rel = t_sec - st.session_state.t_start
    df["t_rel"] = t_rel

    with placeholder.container():
        tabs = st.tabs(["Altitude & Speed", "Attitude", "Engine & Fuel", "Flight Path"])

        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["t_rel"], y=df["alt"], mode="lines", name="Altitude (m)"))
                fig.update_layout(title="Altitude", xaxis_title="Time (s)", yaxis_title="m", height=300)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["t_rel"], y=df["spd"], mode="lines", name="Airspeed (m/s)"))
                if "ground_speed" in df.columns:
                    fig.add_trace(go.Scatter(x=df["t_rel"], y=df["ground_speed"], mode="lines", name="Ground Speed (m/s)"))
                fig.update_layout(title="Speed", xaxis_title="Time (s)", yaxis_title="m/s", height=300)
                st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                if "roll" in df.columns:
                    fig.add_trace(go.Scatter(x=df["t_rel"], y=df["roll"], mode="lines", name="Roll (°)"))
                if "pitch" in df.columns:
                    fig.add_trace(go.Scatter(x=df["t_rel"], y=df["pitch"], mode="lines", name="Pitch (°)"))
                fig.update_layout(title="Roll & Pitch", xaxis_title="Time (s)", yaxis_title="Degrees", height=300)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure()
                if "hdg" in df.columns:
                    fig.add_trace(go.Scatter(x=df["t_rel"], y=df["hdg"], mode="lines", name="Heading (°)"))
                if "g" in df.columns:
                    fig.add_trace(go.Scatter(x=df["t_rel"], y=df["g"], mode="lines", name="G-Force", yaxis="y2"))
                fig.update_layout(
                    title="Heading & G-Force",
                    xaxis_title="Time (s)",
                    yaxis=dict(title="Heading (°)"),
                    yaxis2=dict(title="G-Force", overlaying="y", side="right"),
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                if "epow" in df.columns:
                    fig.add_trace(go.Scatter(x=df["t_rel"], y=df["epow"] * 100, mode="lines", name="Engine Power (%)"))
                if "eact" in df.columns:
                    fig.add_trace(go.Scatter(x=df["t_rel"], y=df["eact"] * 100, mode="lines", name="Engine Active"))
                fig.update_layout(title="Engine", xaxis_title="Time (s)", yaxis_title="%", height=300)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure()
                if "fuel" in df.columns:
                    fig.add_trace(go.Scatter(x=df["t_rel"], y=df["fuel"] * 100, mode="lines", name="Fuel Level (%)"))
                if "flap" in df.columns:
                    fig.add_trace(go.Scatter(x=df["t_rel"], y=df["flap"] * 100, mode="lines", name="Flaps (%)"))
                fig.update_layout(title="Fuel & Flaps", xaxis_title="Time (s)", yaxis_title="%", height=300)
                st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["pz"], y=df["px"], mode="lines+markers", name="Track"))
                fig.update_layout(
                    title="Top-Down Flight Path",
                    xaxis_title="Z (m)",
                    yaxis_title="X (m)",
                    height=400,
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["t_rel"],
                    y=df["py"],
                    mode="lines",
                    name="Altitude (world Y)",
                    fill="tozeroy",
                ))
                fig.update_layout(
                    title="Vertical Profile",
                    xaxis_title="Time (s)",
                    yaxis_title="Height Y (m)",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
