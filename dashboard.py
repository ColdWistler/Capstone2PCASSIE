import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(
    page_title="UAV Telemetry Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .section-header {
        font-size: 12px; font-weight: 600; color: #6c757d;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 0.4rem; margin-top: 1.2rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    div[data-testid="stMetricDelta"] { font-size: 0.75rem; }
    .stAlert { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
CHART_H = 220
CHART_LAYOUT = dict(
    margin=dict(l=8, r=8, t=8, b=8),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False),
)

# ── data loaders ──────────────────────────────────────────────────────────────

def load_jsonl(file_obj) -> pd.DataFrame:
    """Read telemetry.jsonl — one JSON object per line.
    Fields from team lead's Godot exporter:
      t, alt, spd, vspd, roll, pitch, hdg,
      px, py, pz, g, load, fuel, epow, flap,
      gear, stall, eact, ground_speed (computed)
    """
    rows = []
    for raw in file_obj:
        line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # compute relative time in seconds
    if "t" in df.columns:
        df["t_sec"] = (df["t"] - df["t"].iloc[0]) / 1000.0
    # compute ground speed if positions exist but field missing
    if "ground_speed" not in df.columns and "px" in df.columns:
        dx = df["px"].diff().fillna(0)
        dz = df["pz"].diff().fillna(0)
        dt = df["t_sec"].diff().replace(0, np.nan).fillna(0.1)
        df["ground_speed"] = np.sqrt(dx**2 + dz**2) / dt
    return df


def load_resource_csv(file_obj) -> pd.DataFrame:
    """Read resource_YYYYMMDD_HHMMSS.csv from resource_monitor.py.
    Columns: timestamp, cpu_pct, mem_pct, gpu_util_pct, gpu_mem_mb,
             disk_read_mbs, disk_write_mbs
    """
    df = pd.read_csv(file_obj)
    df.columns = [c.strip() for c in df.columns]
    if "timestamp" in df.columns:
        df["t_sec"] = range(len(df))  # 1 sample/sec
    return df


def make_demo_flight(n=120) -> pd.DataFrame:
    """Synthetic flight data matching real JSONL field names."""
    t = np.linspace(0, n, n)
    rng = np.random.default_rng(42)
    rows = []
    for i, ti in enumerate(t):
        alt   = round(float(80 + np.sin(ti * 0.08) * 30 + ti * 0.5 + rng.normal(0, 0.3)), 2)
        spd   = round(float(15 + np.sin(ti * 0.12) * 5  + rng.normal(0, 0.2)), 2)
        vspd  = round(float(np.cos(ti * 0.08) * 2 + rng.normal(0, 0.1)), 2)
        roll  = round(float(np.sin(ti * 0.2) * 8  + rng.normal(0, 0.15)), 2)
        pitch = round(float(np.cos(ti * 0.15) * 5 + rng.normal(0, 0.1)), 2)
        hdg   = round(float(np.mod(180 + ti * 0.4, 360)), 1)
        g     = round(float(1.0 + abs(np.sin(ti * 0.2)) * 0.3 + rng.normal(0, 0.02)), 3)
        fuel  = round(float(max(0, 1.0 - ti / (n * 1.5))), 3)
        epow  = round(float(np.clip(0.6 + np.sin(ti * 0.1) * 0.2, 0.3, 0.95)), 3)
        stall = int(abs(pitch) > 14)
        rows.append({
            "t": int(ti * 1000),
            "t_sec": ti,
            "alt": alt, "spd": spd, "vspd": vspd,
            "roll": roll, "pitch": pitch, "hdg": hdg,
            "px": round(ti * spd * 0.6, 1),
            "py": alt,
            "pz": round(np.sin(ti * 0.05) * 50, 1),
            "g": g, "load": round(g, 3),
            "fuel": fuel, "epow": epow,
            "flap": 0.0, "gear": int(alt < 10),
            "stall": stall, "eact": 1,
            "ground_speed": round(spd * 0.9, 2),
        })
    return pd.DataFrame(rows)


def make_demo_resources(n=60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    t = np.arange(n)
    return pd.DataFrame({
        "t_sec": t,
        "cpu_pct":      np.round(np.clip(45 + np.sin(t*0.3)*20 + rng.normal(0,2,n), 5, 99), 1),
        "mem_pct":      np.round(np.clip(55 + t*0.1 + rng.normal(0,1,n), 10, 95), 1),
        "gpu_util_pct": np.round(np.clip(60 + np.sin(t*0.25)*25 + rng.normal(0,3,n), 0, 100), 1),
        "gpu_mem_mb":   np.round(np.clip(2400 + t*5 + rng.normal(0,30,n), 0, 8000), 0),
        "disk_read_mbs":  np.round(np.abs(rng.normal(0.5, 0.3, n)), 2),
        "disk_write_mbs": np.round(np.abs(rng.normal(0.2, 0.15, n)), 2),
    })


def make_demo_rewards(n=80) -> pd.DataFrame:
    ep = np.arange(1, n+1)
    rng = np.random.default_rng(3)
    r_task    = np.clip(0.2 + ep*0.008 + rng.normal(0, 0.04, n), 0, 1)
    r_smooth  = np.clip(0.3 + ep*0.006 + rng.normal(0, 0.05, n), 0, 1)
    r_safety  = np.clip(0.4 + ep*0.005 + rng.normal(0, 0.03, n), 0, 1)
    violations = np.maximum(0, (10 - ep*0.1 + rng.normal(0,0.5,n)).astype(int))
    return pd.DataFrame({
        "episode": ep,
        "r_task": np.round(r_task, 3),
        "r_smooth": np.round(r_smooth, 3),
        "r_safety": np.round(r_safety, 3),
        "violations": violations,
    })

# ── chart helpers ─────────────────────────────────────────────────────────────

def line(df, x, ys, colors, names, h=CHART_H, y_label=""):
    fig = go.Figure()
    for y, c, nm in zip(ys, colors, names):
        if y in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x], y=df[y], name=nm,
                line=dict(color=c, width=1.8), mode="lines",
                hovertemplate=f"{nm}: %{{y}}<extra></extra>"
            ))
    layout = dict(**CHART_LAYOUT, height=h, yaxis_title=y_label)
    fig.update_layout(**layout)
    fig.update_xaxes(nticks=8)
    return fig


def area(df, x, y, color, h=CHART_H, y_label=""):
    fig = go.Figure(go.Scatter(
        x=df[x], y=df[y], fill="tozeroy",
        line=dict(color=color, width=1.8),
        fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba"),
        hovertemplate=f"%{{y}}<extra></extra>"
    ))
    fig.update_layout(**CHART_LAYOUT, height=h, yaxis_title=y_label)
    return fig


def bar_components(last_r_task, last_r_smooth, last_r_safety, h=CHART_H):
    layout = {k: v for k, v in CHART_LAYOUT.items() if k != "yaxis"}
    fig = go.Figure(go.Bar(
        x=["R_task", "R_smooth", "R_safety"],
        y=[last_r_task, last_r_smooth, last_r_safety],
        marker_color=["#1D9E75", "#378ADD", "#7F77DD"],
        text=[f"{v:.3f}" for v in [last_r_task, last_r_smooth, last_r_safety]],
        textposition="outside",
    ))
    fig.update_layout(**layout, height=h)
    fig.update_yaxes(range=[0, 1.15], showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def flight_path(df, h=280):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["pz"], y=df["px"], mode="lines+markers",
        marker=dict(size=3), line=dict(color="#378ADD", width=1.5),
        name="Flight path",
        hovertemplate="X:%{y:.1f}  Z:%{x:.1f}<extra></extra>"
    ))
    layout = {k: v for k, v in CHART_LAYOUT.items() if k != "yaxis"}
    fig.update_layout(**layout, height=h,
                      xaxis_title="Z (m)", yaxis_title="X (m)",
                      yaxis=dict(scaleanchor="x", scaleratio=1,
                                 showgrid=True, gridcolor="rgba(128,128,128,0.15)"))
    return fig

# ── badge helper ──────────────────────────────────────────────────────────────

def badge(label, text, color):
    return f"""<div style='background:{color}22;border:1px solid {color}55;
    border-radius:8px;padding:6px 14px;white-space:nowrap;display:inline-block;margin-right:8px;'>
    <span style='font-size:13px;font-weight:600;color:{color};'>{label} {text}</span></div>"""

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — CSV history store
# ══════════════════════════════════════════════════════════════════════════════
# We keep a dict in session_state so uploaded files survive reruns:
#   st.session_state.resource_history = { "filename": dataframe, ... }
#   st.session_state.flight_history   = { "filename": dataframe, ... }

if "resource_history" not in st.session_state:
    st.session_state.resource_history = {}   # { name: df }

if "flight_history" not in st.session_state:
    st.session_state.flight_history = {}     # { name: df }

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ✈️ UAV Dashboard")
    st.markdown("---")

    st.markdown("**Data source**")
    source = st.radio("src", ["Demo (no files needed)",
                               "Upload telemetry.jsonl",
                               "Upload resources CSV"],
                      label_visibility="collapsed")

    # ── FLIGHT JSONL section ──────────────────────────────────────────────────
    flight_file = None
    selected_flight_name = None

    if source == "Upload telemetry.jsonl":
        flight_file = st.file_uploader(
            "Upload telemetry.jsonl",
            type=["jsonl", "json", "txt"],
            help="File written by Godot TelemetryExporter node"
        )
        # save newly uploaded file into history
        if flight_file is not None:
            fname = flight_file.name
            if fname not in st.session_state.flight_history:
                parsed = load_jsonl(flight_file)
                if not parsed.empty:
                    st.session_state.flight_history[fname] = parsed
                    st.sidebar.success(f"Saved: {fname}")
                else:
                    st.sidebar.warning("Could not parse — check file format")

        # dropdown to pick from history
        if st.session_state.flight_history:
            st.markdown("**Loaded flight sessions**")
            selected_flight_name = st.selectbox(
                "Select session to view",
                options=list(st.session_state.flight_history.keys()),
                help="All sessions you've uploaded this browser session"
            )
            # delete button
            if st.button("🗑 Remove selected session", key="del_flight"):
                del st.session_state.flight_history[selected_flight_name]
                st.rerun()
        else:
            st.info("No flight files uploaded yet.")

    # ── RESOURCES CSV section ─────────────────────────────────────────────────
    resource_file = None
    selected_res_name = None

    if source == "Upload resources CSV":
        resource_file = st.file_uploader(
            "Upload resources_*.csv",
            type=["csv"],
            help="CSV exported by resource_monitor.py"
        )
        # save newly uploaded file into history
        if resource_file is not None:
            rname = resource_file.name
            if rname not in st.session_state.resource_history:
                parsed_r = load_resource_csv(resource_file)
                if not parsed_r.empty:
                    st.session_state.resource_history[rname] = parsed_r
                    st.sidebar.success(f"Saved: {rname}")
                else:
                    st.sidebar.warning("Could not parse CSV")

        # dropdown to pick from history
        if st.session_state.resource_history:
            st.markdown("**Uploaded resource files**")
            selected_res_name = st.selectbox(
                "Select file to view",
                options=list(st.session_state.resource_history.keys()),
                help="All CSVs you've uploaded this browser session"
            )
            # show quick info
            _preview = st.session_state.resource_history[selected_res_name]
            st.caption(f"{len(_preview)} rows · {len(_preview.columns)} columns")
            # delete button
            if st.button("🗑 Remove selected file", key="del_res"):
                del st.session_state.resource_history[selected_res_name]
                st.rerun()
        else:
            st.info("No resource CSVs uploaded yet.")

    st.markdown("---")
    st.markdown("**View**")
    show_n = st.slider("Data points shown", 30, 500, 120)

    st.markdown("---")
    st.markdown("**Reward weights** *(AI section)*")
    w1 = st.slider("w1 — R_task",   0.0, 1.0, 0.4, 0.05)
    w2 = st.slider("w2 — R_smooth", 0.0, 1.0, 0.3, 0.05)
    w3 = st.slider("w3 — R_safety", 0.0, 1.0, 0.3, 0.05)
    ws = round(w1+w2+w3, 2)
    st.success(f"Sum = {ws} ✓") if ws == 1.0 else st.warning(f"Sum = {ws} (needs 1.0)")

    st.markdown("---")
    st.caption("Capstone II · UAV Co-pilot Project\nRija Bhomi — Data Architect")

# ── load data ─────────────────────────────────────────────────────────────────

# flight data — use selected history item if available
if (source == "Upload telemetry.jsonl"
        and selected_flight_name
        and selected_flight_name in st.session_state.flight_history):
    fdf = st.session_state.flight_history[selected_flight_name]
    using_real_flight = True
else:
    fdf = make_demo_flight()
    using_real_flight = False

# resource data — use selected history item if available
if (source == "Upload resources CSV"
        and selected_res_name
        and selected_res_name in st.session_state.resource_history):
    rdf = st.session_state.resource_history[selected_res_name]
    using_real_res = True
else:
    rdf = make_demo_resources()
    using_real_res = False

rew_df = make_demo_rewards()
rew_df["total"] = np.round(w1*rew_df["r_task"] + w2*rew_df["r_smooth"] + w3*rew_df["r_safety"], 3)

# trim to show_n
fdf_view = fdf.tail(show_n).copy()
latest   = fdf.iloc[-1]
prev     = fdf.iloc[-2] if len(fdf) > 1 else fdf.iloc[-1]

# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════

col_title, col_badges = st.columns([2, 3])
with col_title:
    tag = "🟢 Real data" if using_real_flight else "🔵 Demo data"
    st.markdown(f"## ✈️ UAV Telemetry & AI Dashboard")
    st.caption(tag)
with col_badges:
    stall_val = int(latest.get("stall", 0))
    gear_val  = int(latest.get("gear", 0))
    eact_val  = int(latest.get("eact", 1))
    stall_color = "#dc3545" if stall_val else "#198754"
    gear_txt  = "DOWN ⬇" if gear_val else "UP ⬆"
    eng_color = "#198754" if eact_val else "#6c757d"
    st.markdown(
        badge("🚨 STALL", "YES" if stall_val else "NO", stall_color) +
        badge("⚙️ Engine", "ON" if eact_val else "OFF", eng_color) +
        badge("🛬 Gear", gear_txt, "#378ADD"),
        unsafe_allow_html=True
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FLIGHT TELEMETRY  (real JSONL fields)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📡 Flight Telemetry</div>', unsafe_allow_html=True)

# row 1 — primary metrics
c1,c2,c3,c4,c5,c6 = st.columns(6)
def delta(field):
    return round(float(latest.get(field,0)) - float(prev.get(field,0)), 2)

c1.metric("Altitude",     f"{latest.get('alt',0):.1f} m",    f"{delta('alt'):+.1f}")
c2.metric("Airspeed",     f"{latest.get('spd',0):.1f} m/s",  f"{delta('spd'):+.1f}")
c3.metric("Vert. Speed",  f"{latest.get('vspd',0):.1f} m/s", f"{delta('vspd'):+.1f}")
c4.metric("Ground Speed", f"{latest.get('ground_speed',0):.1f} m/s")
c5.metric("G-Force",      f"{latest.get('g',0):.2f} g",      f"{delta('g'):+.2f}")
c6.metric("Heading",      f"{latest.get('hdg',0):.1f}°")

# row 2 — secondary metrics
d1,d2,d3,d4,d5,d6 = st.columns(6)
d1.metric("Roll",    f"{latest.get('roll',0):.1f}°",  f"{delta('roll'):+.1f}")
d2.metric("Pitch",   f"{latest.get('pitch',0):.1f}°", f"{delta('pitch'):+.1f}")
d3.metric("Engine",  f"{latest.get('epow',0)*100:.0f}%")
d4.metric("Fuel",    f"{latest.get('fuel',0)*100:.0f}%")
d5.metric("Flaps",   f"{latest.get('flap',0)*100:.0f}%")
d6.metric("Load",    f"{latest.get('load',0):.2f}")

st.markdown("")

# charts row 1
ch1, ch2 = st.columns(2)
with ch1:
    st.markdown("**Altitude & Vertical Speed**")
    st.plotly_chart(
        line(fdf_view, "t_sec", ["alt","vspd"], ["#378ADD","#1D9E75"],
             ["Altitude (m)","Vert. Speed (m/s)"], y_label="m / m·s⁻¹"),
        use_container_width=True)
with ch2:
    st.markdown("**Airspeed & Ground Speed**")
    st.plotly_chart(
        line(fdf_view, "t_sec", ["spd","ground_speed"], ["#BA7517","#7F77DD"],
             ["Airspeed (m/s)","Ground Speed (m/s)"], y_label="m/s"),
        use_container_width=True)

# charts row 2
ch3, ch4 = st.columns(2)
with ch3:
    st.markdown("**Roll & Pitch**")
    st.plotly_chart(
        line(fdf_view, "t_sec", ["roll","pitch"], ["#D4537E","#1D9E75"],
             ["Roll (°)","Pitch (°)"], y_label="degrees"),
        use_container_width=True)
with ch4:
    st.markdown("**G-Force & Load Factor**")
    st.plotly_chart(
        line(fdf_view, "t_sec", ["g","load"], ["#E24B4A","#BA7517"],
             ["G-Force","Load Factor"], y_label="g"),
        use_container_width=True)

# charts row 3
ch5, ch6 = st.columns(2)
with ch5:
    st.markdown("**Engine Power & Fuel**")
    fuel_pct = fdf_view.copy()
    fuel_pct["fuel_pct"] = fuel_pct["fuel"] * 100
    fuel_pct["epow_pct"] = fuel_pct["epow"] * 100
    st.plotly_chart(
        line(fuel_pct, "t_sec", ["epow_pct","fuel_pct"], ["#E24B4A","#378ADD"],
             ["Engine Power (%)","Fuel Level (%)"], y_label="%"),
        use_container_width=True)
with ch6:
    st.markdown("**3D Flight Path (top-down)**")
    if "px" in fdf_view.columns and "pz" in fdf_view.columns:
        st.plotly_chart(flight_path(fdf_view), use_container_width=True)
    else:
        st.info("Position data (px/pz) not available in this file.")

# safety banner
stall_count = int(fdf["stall"].sum()) if "stall" in fdf.columns else 0
high_g      = int((fdf["g"] > 3.0).sum()) if "g" in fdf.columns else 0
if stall_count > 0:
    st.error(f"⚠️ {stall_count} stall event(s) detected in this session.")
elif high_g > 0:
    st.warning(f"🟡 {high_g} high G-force event(s) detected (>3g).")
else:
    st.success("✅ No stall or high-G events detected.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SYSTEM RESOURCES  (from resource_monitor.py CSV)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🖥️ System Resources (Simulation Performance)</div>',
            unsafe_allow_html=True)

rdf_view = rdf.tail(show_n)

r1,r2,r3,r4 = st.columns(4)
r1.metric("CPU",       f"{rdf['cpu_pct'].iloc[-1]:.1f}%")
r2.metric("Memory",    f"{rdf['mem_pct'].iloc[-1]:.1f}%")
if "gpu_util_pct" in rdf.columns and rdf["gpu_util_pct"].notna().any():
    r3.metric("GPU",       f"{rdf['gpu_util_pct'].iloc[-1]:.1f}%")
    r4.metric("GPU Mem",   f"{rdf['gpu_mem_mb'].iloc[-1]:.0f} MB")
else:
    r3.metric("Disk Read",  f"{rdf['disk_read_mbs'].iloc[-1]:.2f} MB/s")
    r4.metric("Disk Write", f"{rdf['disk_write_mbs'].iloc[-1]:.2f} MB/s")

rc1, rc2 = st.columns(2)
with rc1:
    st.markdown("**CPU & Memory over time**")
    st.plotly_chart(
        line(rdf_view, "t_sec", ["cpu_pct","mem_pct"], ["#378ADD","#1D9E75"],
             ["CPU (%)","Memory (%)"], y_label="%"),
        use_container_width=True)
with rc2:
    if "gpu_util_pct" in rdf_view.columns and rdf_view["gpu_util_pct"].notna().any():
        st.markdown("**GPU Utilisation & Memory**")
        st.plotly_chart(
            line(rdf_view, "t_sec", ["gpu_util_pct"], ["#7F77DD"],
                 ["GPU Util (%)"], y_label="%"),
            use_container_width=True)
    else:
        st.markdown("**Disk I/O over time**")
        st.plotly_chart(
            line(rdf_view, "t_sec", ["disk_read_mbs","disk_write_mbs"],
                 ["#BA7517","#E24B4A"], ["Read MB/s","Write MB/s"], y_label="MB/s"),
            use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AI / RL MONITORING  (your unique section)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🤖 AI Co-pilot Monitoring</div>', unsafe_allow_html=True)

a1,a2,a3,a4 = st.columns(4)
last_rew = rew_df.iloc[-1]
prev_rew = rew_df.iloc[-2]
a1.metric("Total Reward",    f"{last_rew['total']:.3f}",
          f"{round(last_rew['total']-prev_rew['total'],3):+.3f}")
a2.metric("Episodes",        f"{int(rew_df['episode'].max())}")
a3.metric("Total Violations",f"{int(rew_df['violations'].sum())}")
trend = "📈 Improving" if rew_df["total"].iloc[-10:].mean() > rew_df["total"].iloc[:10].mean() else "📉 Needs tuning"
a4.metric("Convergence",     trend)

ac1, ac2 = st.columns(2)
with ac1:
    st.markdown("**Reward convergence over episodes**")
    st.plotly_chart(
        area(rew_df, "episode", "total", "rgb(127,119,221)", y_label="Total reward"),
        use_container_width=True)
with ac2:
    st.markdown("**Reward components — latest episode**")
    st.plotly_chart(
        bar_components(last_rew["r_task"], last_rew["r_smooth"], last_rew["r_safety"]),
        use_container_width=True)

ac3, ac4 = st.columns(2)
with ac3:
    st.markdown("**All components over episodes**")
    st.plotly_chart(
        line(rew_df, "episode", ["r_task","r_smooth","r_safety"],
             ["#1D9E75","#378ADD","#7F77DD"],
             ["R_task","R_smooth","R_safety"], y_label="reward"),
        use_container_width=True)
with ac4:
    st.markdown("**Envelope violations per episode**")
    st.plotly_chart(
        area(rew_df, "episode", "violations", "rgb(226,75,74)", y_label="violations"),
        use_container_width=True)

st.info(f"Reward weights → w1={w1} · w2={w2} · w3={w3} — adjust in sidebar to experiment.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FLIGHT LOG TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📋 Flight Log</div>', unsafe_allow_html=True)

display_cols = [c for c in ["t_sec","alt","spd","vspd","roll","pitch","hdg",
                              "g","fuel","epow","stall","gear"] if c in fdf.columns]
log_df = fdf[display_cols].tail(show_n).copy()
log_df.columns = [c.replace("t_sec","Time(s)").replace("alt","Alt(m)")
                   .replace("spd","Spd(m/s)").replace("vspd","VSp(m/s)")
                   .replace("roll","Roll(°)").replace("pitch","Pitch(°)")
                   .replace("hdg","Hdg(°)").replace("g","G-Force")
                   .replace("fuel","Fuel").replace("epow","EngPow")
                   .replace("stall","Stall").replace("gear","Gear")
                  for c in log_df.columns]

st.dataframe(log_df.round(2), use_container_width=True, hide_index=True)

# download
csv_bytes = fdf.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download full flight log as CSV", csv_bytes,
                   file_name=f"flight_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                   mime="text/csv")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SESSION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📊 Session Summary</div>', unsafe_allow_html=True)

s1,s2,s3,s4,s5,s6 = st.columns(6)
s1.metric("Max Altitude",  f"{fdf['alt'].max():.1f} m"  if 'alt'  in fdf.columns else "—")
s2.metric("Max Speed",     f"{fdf['spd'].max():.1f} m/s" if 'spd'  in fdf.columns else "—")
s3.metric("Max G-Force",   f"{fdf['g'].max():.2f} g"    if 'g'    in fdf.columns else "—")
s4.metric("Max  |Roll|",   f"{fdf['roll'].abs().max():.1f}°" if 'roll' in fdf.columns else "—")
s5.metric("Max |Pitch|",   f"{fdf['pitch'].abs().max():.1f}°" if 'pitch' in fdf.columns else "—")
s6.metric("Total Samples", f"{len(fdf)}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL OPTIMIZATION EXPERIMENT COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🔬 Model Optimization — Experiment Comparison</div>',
            unsafe_allow_html=True)

st.caption("DQN training runs — comparing reward function and hyperparameter changes. "
           "Best reward = highest reward value reached during that training run.")

# ── editable experiment table ─────────────────────────────────────────────────
default_experiments = pd.DataFrame([
    {"Experiment": "Baseline",  "LR": 0.0001, "Hidden Units": 256,
     "Stall Penalty": 1.0, "Landing Reward": 15,  "Best Reward": 4055.5,
     "Notes": "Original config — best result"},
    {"Experiment": "Exp 1",     "LR": 0.0001, "Hidden Units": 256,
     "Stall Penalty": 2.0, "Landing Reward": 50,  "Best Reward": 2791.5,
     "Notes": "Harsher penalty hurt training"},
    {"Experiment": "Exp 2",     "LR": 0.0005, "Hidden Units": 256,
     "Stall Penalty": 2.0, "Landing Reward": 50,  "Best Reward": 806.3,
     "Notes": "Higher LR on top of Exp1 broke convergence"},
])

# let user add rows manually
with st.expander("➕ Add / edit experiments", expanded=False):
    st.caption("Edit the table below — add new rows as you run more experiments in Godot.")
    exp_df = st.data_editor(
        default_experiments,
        num_rows="dynamic",
        use_container_width=True,
        key="exp_editor"
    )

if "exp_editor" not in st.session_state:
    exp_df = default_experiments

# ── highlight best row ────────────────────────────────────────────────────────
best_idx = exp_df["Best Reward"].idxmax()

def highlight_best(row):
    if row.name == best_idx:
        return ["background-color: #1D9E7533"] * len(row)
    return [""] * len(row)

st.dataframe(
    exp_df.style.apply(highlight_best, axis=1).format({
        "LR": "{:.4f}",
        "Best Reward": "{:.1f}",
        "Stall Penalty": "{:.1f}",
    }),
    use_container_width=True,
    hide_index=True
)
st.caption("🟢 Green row = best performing experiment")

# ── bar chart comparing best rewards ─────────────────────────────────────────
exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    st.markdown("**Best Reward per Experiment**")
    colors_exp = ["#1D9E75" if i == best_idx else "#378ADD"
                  for i in range(len(exp_df))]
    fig_exp = go.Figure(go.Bar(
        x=exp_df["Experiment"],
        y=exp_df["Best Reward"],
        marker_color=colors_exp,
        text=exp_df["Best Reward"].round(1),
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}<extra></extra>"
    ))
    layout_exp = {k: v for k, v in CHART_LAYOUT.items() if k != "yaxis"}
    fig_exp.update_layout(**layout_exp, height=280, yaxis_title="Best Reward")
    fig_exp.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig_exp, use_container_width=True)

with exp_col2:
    st.markdown("**What the results mean**")
    best_exp = exp_df.loc[best_idx, "Experiment"]
    best_val = exp_df.loc[best_idx, "Best Reward"]
    worst_val = exp_df["Best Reward"].min()
    drop_pct  = round((1 - worst_val / best_val) * 100, 1)
    st.markdown(f"""
**Best run:** {best_exp} → reward **{best_val:.1f}**

**Worst run:** reward **{worst_val:.1f}** ({drop_pct}% drop from best)

**Key finding:** Increasing the stall penalty from 1.0 → 2.0 combined
with a larger landing reward destabilized training — the agent was
penalized too heavily and stopped exploring. Then raising the learning
rate compounded the problem.

**Conclusion for report:** The baseline reward configuration produced
the most stable convergence. Aggressive simultaneous changes to multiple
reward components should be avoided — one change per experiment is
the correct methodology.
    """)
    # download experiment table
    exp_csv = exp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download experiment table as CSV",
        exp_csv,
        file_name="experiment_comparison.csv",
        mime="text/csv"
    )