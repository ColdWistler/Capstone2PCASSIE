import json
import re
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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


def parse_training_log(file_obj) -> pd.DataFrame:
    """Parse log_seed_*.txt training logs.
    Extracts per-episode lines like:
      Episode 12 | reward: 1984.6 | steps: 1088 | eps: 0.930 | best: 1984.6
    Also extracts crash/landing/stall events.
    """
    lines = []
    for raw in file_obj:
        line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else raw.strip()
        if line:
            lines.append(line)

    ep_re = re.compile(
        r"Episode\s+(\d+)\s*\|\s*reward:\s*([-\d.]+)\s*\|\s*steps:\s*(\d+)\s*\|\s*eps:\s*([0-9.]+)\s*\|\s*best:\s*([-\d.]+)"
    )
    rows = []
    for line in lines:
        m = ep_re.search(line)
        if m:
            rows.append({
                "episode": int(m.group(1)),
                "reward": float(m.group(2)),
                "steps": int(m.group(3)),
                "epsilon": float(m.group(4)),
                "best_reward": float(m.group(5)),
            })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # smooth reward curve
    window = min(10, len(df))
    if window > 1:
        df["reward_smooth"] = df["reward"].rolling(window, min_periods=1).mean()
    else:
        df["reward_smooth"] = df["reward"]

    # count events from raw lines
    landing_count = sum(1 for l in lines if "Landing approach" in l)
    crash_count = sum(1 for l in lines if "crash" in l.lower() or "Crash" in l)
    stall_count = sum(1 for l in lines if "stall" in l.lower() or "Stall" in l)
    weight_snapshots = sum(1 for l in lines if "Weight snapshot" in l)

    df.attrs["landing_count"] = landing_count
    df.attrs["crash_count"] = crash_count
    df.attrs["stall_count"] = stall_count
    df.attrs["weight_snapshots"] = weight_snapshots
    df.attrs["total_episodes"] = len(df)

    return df


def load_rich_csv(file_obj) -> pd.DataFrame:
    """Read telemetry/csv/*.csv with extended DQN columns:
    t_ms, alt_m, spd_ms, fwd_spd_ms, vspd_ms, g_force, load_factor,
    stall, px, py, pz, roll_deg, pitch_deg, hdg_deg, epow, eact,
    flap, gear, fuel, action, state_0..state_15, Q_0..Q_6,
    reward, epsilon, episode, step_count
    """
    df = pd.read_csv(file_obj)
    df.columns = [c.strip() for c in df.columns]
    # normalise column names so existing chart helpers work
    rename = {}
    col_map = {
        "t_ms": "t", "alt_m": "alt", "spd_ms": "spd", "fwd_spd_ms": "fwd_spd",
        "vspd_ms": "vspd", "g_force": "g", "load_factor": "load",
        "roll_deg": "roll", "pitch_deg": "pitch", "hdg_deg": "hdg",
    }
    for old, new in col_map.items():
        if old in df.columns:
            rename[old] = new
    if rename:
        df.rename(columns=rename, inplace=True)
    if "t" in df.columns:
        df["t_sec"] = (df["t"] - df["t"].iloc[0]) / 1000.0
    return df


def read_from_disk(path_str: str, file_type: str) -> pd.DataFrame:
    """Read a file directly from disk by path."""
    p = Path(path_str.strip())
    if not p.exists():
        return pd.DataFrame()
    try:
        if file_type == "jsonl":
            with open(p, "r", encoding="utf-8") as f:
                return load_jsonl(f)
        elif file_type == "csv":
            with open(p, "r", encoding="utf-8") as f:
                return load_resource_csv(f)
        elif file_type == "rich_csv":
            with open(p, "r", encoding="utf-8") as f:
                return load_rich_csv(f)
        elif file_type == "log":
            with open(p, "r", encoding="utf-8") as f:
                return parse_training_log(f)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


PROJECT_ROOT = Path(__file__).parent.resolve()


def auto_discover_files():
    """Scan disk for existing data files and load the most recent into session state.
    Called once on startup so the dashboard shows real data immediately.
    """
    # ── rich telemetry CSVs ────────────────────────────────────────────────
    csv_dirs = [PROJECT_ROOT / "telemetry" / "telemetry" / "csv",
                PROJECT_ROOT / "telemetry" / "csv"]
    for csv_dir in csv_dirs:
        if csv_dir.exists():
            for csv_file in sorted(csv_dir.glob("telemetry_*.csv")):
                fname = csv_file.name
                if fname not in st.session_state.rich_csv_history:
                    try:
                        df = load_rich_csv(open(csv_file, "r", encoding="utf-8"))
                        if not df.empty:
                            st.session_state.rich_csv_history[fname] = df
                    except Exception:
                        pass

    # ── training logs ──────────────────────────────────────────────────────
    for log_file in sorted(PROJECT_ROOT.glob("log_seed_*.txt")):
        fname = log_file.name
        if fname not in st.session_state.log_history:
            try:
                df = parse_training_log(open(log_file, "r", encoding="utf-8"))
                if not df.empty:
                    st.session_state.log_history[fname] = df
            except Exception:
                pass

    # ── resource CSVs ──────────────────────────────────────────────────────
    for res_file in sorted(PROJECT_ROOT.glob("resources_*.csv")):
        fname = res_file.name
        if fname not in st.session_state.resource_history:
            try:
                df = load_resource_csv(open(res_file, "r", encoding="utf-8"))
                if not df.empty:
                    st.session_state.resource_history[fname] = df
            except Exception:
                pass
    for res_file in sorted(PROJECT_ROOT.glob("telemetry_log_*.csv")):
        fname = res_file.name
        if fname not in st.session_state.resource_history:
            try:
                df = load_resource_csv(open(res_file, "r", encoding="utf-8"))
                if not df.empty:
                    st.session_state.resource_history[fname] = df
            except Exception:
                pass


def build_real_rewards_from_rich_csv(rcdf: pd.DataFrame) -> pd.DataFrame:
    """Build a rewards DataFrame from the rich CSV's episode + reward columns.
    Produces per-episode rows with: episode, reward, best_reward, epsilon, steps, violations.
    """
    if rcdf.empty or "episode" not in rcdf.columns or "reward" not in rcdf.columns:
        return pd.DataFrame()

    grouped = rcdf.groupby("episode", sort=True)
    rows = []
    for ep, grp in grouped:
        total_reward = float(grp["reward"].sum())
        best_reward = float(rcdf.loc[rcdf["episode"] <= ep, "reward"].sum())
        eps = float(grp["epsilon"].iloc[-1]) if "epsilon" in grp.columns else 0.0
        steps = len(grp)
        violations = int((grp.get("stall", pd.Series(dtype=float)) == 1).sum()) if "stall" in grp.columns else 0
        rows.append({
            "episode": int(ep),
            "reward": round(total_reward, 2),
            "best_reward": round(best_reward, 2),
            "epsilon": round(eps, 4),
            "steps": steps,
            "violations": violations,
        })

    df = pd.DataFrame(rows)
    if len(df) > 1:
        window = min(10, len(df))
        df["reward_smooth"] = df["reward"].rolling(window, min_periods=1).mean()
    else:
        df["reward_smooth"] = df["reward"]
    return df

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

if "log_history" not in st.session_state:
    st.session_state.log_history = {}        # { name: df }

if "rich_csv_history" not in st.session_state:
    st.session_state.rich_csv_history = {}   # { name: df }

if "auto_loaded" not in st.session_state:
    st.session_state.auto_loaded = False

if not st.session_state.auto_loaded:
    auto_discover_files()
    st.session_state.auto_loaded = True

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ✈️ UAV Dashboard")
    st.markdown("---")

    # show auto-discovered file counts
    n_rich = len(st.session_state.rich_csv_history)
    n_logs = len(st.session_state.log_history)
    n_res  = len(st.session_state.resource_history)
    if n_rich or n_logs or n_res:
        parts = []
        if n_rich: parts.append(f"{n_rich} CSV")
        if n_logs: parts.append(f"{n_logs} log")
        if n_res:  parts.append(f"{n_res} resource")
        st.success(f"Auto-loaded: {', '.join(parts)}")
    else:
        st.info("No data files found on disk — upload or set a path below.")

    st.markdown("**Data source**")
    source = st.radio("src", ["Demo (no files needed)",
                               "Upload telemetry.jsonl",
                               "Upload resources CSV",
                               "Upload training log",
                               "Upload rich telemetry CSV",
                               "From disk path"],
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

    # ── TRAINING LOG section ───────────────────────────────────────────────
    log_file = None
    selected_log_name = None

    if source == "Upload training log":
        log_file = st.file_uploader(
            "Upload log_seed_*.txt",
            type=["txt", "log"],
            help="Training log from Godot DQN runs (log_seed_42.txt etc.)"
        )
        if log_file is not None:
            lname = log_file.name
            if lname not in st.session_state.log_history:
                parsed_l = parse_training_log(log_file)
                if not parsed_l.empty:
                    st.session_state.log_history[lname] = parsed_l
                    st.sidebar.success(f"Saved: {lname} ({len(parsed_l)} episodes)")
                else:
                    st.sidebar.warning("No episode data found in log — check format")

        if st.session_state.log_history:
            st.markdown("**Loaded training logs**")
            selected_log_name = st.selectbox(
                "Select log to view",
                options=list(st.session_state.log_history.keys()),
            )
            _lp = st.session_state.log_history[selected_log_name]
            st.caption(f"{len(_lp)} episodes · best reward: {_lp['reward'].max():.1f}")
            if st.button("🗑 Remove selected log", key="del_log"):
                del st.session_state.log_history[selected_log_name]
                st.rerun()
        else:
            st.info("No training logs uploaded yet.")

    # ── RICH CSV section ───────────────────────────────────────────────────
    rich_file = None
    selected_rich_name = None

    if source == "Upload rich telemetry CSV":
        rich_file = st.file_uploader(
            "Upload telemetry CSV (with DQN columns)",
            type=["csv"],
            help="CSV from TelemetryCsvExporter — has state_*, Q_*, reward, epsilon columns"
        )
        if rich_file is not None:
            rname = rich_file.name
            if rname not in st.session_state.rich_csv_history:
                parsed_rc = load_rich_csv(rich_file)
                if not parsed_rc.empty:
                    st.session_state.rich_csv_history[rname] = parsed_rc
                    st.sidebar.success(f"Saved: {rname} ({len(parsed_rc)} rows)")
                else:
                    st.sidebar.warning("Could not parse CSV")

        if st.session_state.rich_csv_history:
            st.markdown("**Uploaded rich telemetry CSVs**")
            selected_rich_name = st.selectbox(
                "Select CSV to view",
                options=list(st.session_state.rich_csv_history.keys()),
            )
            _rc = st.session_state.rich_csv_history[selected_rich_name]
            st.caption(f"{len(_rc)} rows · {len(_rc.columns)} columns")
            if st.button("🗑 Remove selected CSV", key="del_rich"):
                del st.session_state.rich_csv_history[selected_rich_name]
                st.rerun()
        else:
            st.info("No rich telemetry CSVs uploaded yet.")

    # ── DISK PATH section ──────────────────────────────────────────────────
    disk_path = ""
    disk_file_type = "jsonl"
    disk_refresh = False
    disk_refresh_sec = 3

    if source == "From disk path":
        disk_path = st.text_input(
            "File path",
            value="",
            placeholder="e.g. telemetry/telemetry/csv/telemetry_2026-07-14T07-50-44.csv",
            help="Relative or absolute path. Auto-discovered files are already loaded — "
                 "use this to point at a specific file."
        )
        disk_file_type = st.selectbox(
            "File type",
            ["jsonl", "csv", "rich_csv", "log"],
            format_func=lambda x: {
                "jsonl": "telemetry.jsonl (flight data)",
                "csv": "resource CSV (CPU/GPU/etc)",
                "rich_csv": "rich telemetry CSV (with DQN columns)",
                "log": "training log (log_seed_*)",
            }[x]
        )
        # show available files as hints
        if st.session_state.rich_csv_history:
            with st.expander("📁 Available rich CSVs on disk", expanded=False):
                for name in st.session_state.rich_csv_history:
                    st.code(name, language=None)
        if st.session_state.log_history:
            with st.expander("📁 Available training logs on disk", expanded=False):
                for name in st.session_state.log_history:
                    st.code(name, language=None)
        disk_refresh = st.checkbox("Auto-refresh", value=False,
                                    help="Periodically re-read the file from disk")
        if disk_refresh:
            disk_refresh_sec = st.slider("Refresh interval (s)", 1, 30, 3)

    # ── REFRESH control ────────────────────────────────────────────────────
    if disk_refresh:
        st.session_state._disk_path = disk_path
        st.session_state._disk_type = disk_file_type
        st.session_state._disk_interval = disk_refresh_sec

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
elif source == "From disk path" and disk_file_type == "jsonl" and disk_path:
    fdf = read_from_disk(disk_path, "jsonl")
    using_real_flight = not fdf.empty
else:
    fdf = make_demo_flight()
    using_real_flight = False

# resource data — use selected or auto-discovered
if (source == "Upload resources CSV"
        and selected_res_name
        and selected_res_name in st.session_state.resource_history):
    rdf = st.session_state.resource_history[selected_res_name]
    using_real_res = True
elif source == "From disk path" and disk_file_type == "csv" and disk_path:
    rdf = read_from_disk(disk_path, "csv")
    using_real_res = not rdf.empty
elif st.session_state.resource_history:
    first_res = list(st.session_state.resource_history.values())[0]
    rdf = first_res
    using_real_res = True
else:
    rdf = make_demo_resources()
    using_real_res = False

# training log data — use selected or auto-discovered
if (source == "Upload training log"
        and selected_log_name
        and selected_log_name in st.session_state.log_history):
    log_df = st.session_state.log_history[selected_log_name]
    using_real_log = True
elif source == "From disk path" and disk_file_type == "log" and disk_path:
    log_df = read_from_disk(disk_path, "log")
    using_real_log = not log_df.empty
elif st.session_state.log_history:
    first_log = list(st.session_state.log_history.values())[0]
    log_df = first_log
    using_real_log = True
else:
    log_df = pd.DataFrame()
    using_real_log = False

# rich CSV data — use selected or auto-discovered
if (source == "Upload rich telemetry CSV"
        and selected_rich_name
        and selected_rich_name in st.session_state.rich_csv_history):
    rcdf = st.session_state.rich_csv_history[selected_rich_name]
    using_real_rich = True
elif source == "From disk path" and disk_file_type == "rich_csv" and disk_path:
    rcdf = read_from_disk(disk_path, "rich_csv")
    using_real_rich = not rcdf.empty
elif st.session_state.rich_csv_history:
    first_rich = list(st.session_state.rich_csv_history.values())[0]
    rcdf = first_rich
    using_real_rich = True
else:
    rcdf = pd.DataFrame()
    using_real_rich = False

# build reward data for Section 3 — prefer real rich CSV, fall back to demo
if using_real_rich and not rcdf.empty and "episode" in rcdf.columns:
    rew_df = build_real_rewards_from_rich_csv(rcdf)
    if rew_df.empty:
        rew_df = make_demo_rewards()
else:
    rew_df = make_demo_rewards()
rew_df["total"] = np.round(w1*rew_df.get("reward", rew_df.get("r_task", pd.Series([0])))
                            + w2*rew_df.get("r_smooth", rew_df.get("r_smooth", pd.Series([0])))
                            + w3*rew_df.get("r_safety", rew_df.get("r_safety", pd.Series([0]))), 3)

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

using_real_rewards = using_real_rich and "reward" in rew_df.columns and "reward_smooth" in rew_df.columns

a1, a2, a3, a4 = st.columns(4)
if using_real_rewards and len(rew_df) > 1:
    last_rew = rew_df.iloc[-1]
    prev_rew = rew_df.iloc[-2]
    a1.metric("Total Reward (cumulative)",
              f"{last_rew['best_reward']:.1f}",
              f"{round(last_rew['best_reward']-prev_rew['best_reward'],1):+.1f}")
    a2.metric("Episodes", f"{int(rew_df['episode'].max())}")
    a3.metric("Total Violations", f"{int(rew_df['violations'].sum())}")
    recent = rew_df["reward_smooth"].iloc[-min(10, len(rew_df)):]
    early  = rew_df["reward_smooth"].iloc[:min(10, len(rew_df))]
    trend  = "📈 Improving" if recent.mean() > early.mean() else "📉 Needs tuning"
    a4.metric("Convergence", trend)
    st.caption("📡 Data source: rich telemetry CSV (per-step reward aggregated per episode)")
elif len(rew_df) > 1:
    last_rew = rew_df.iloc[-1]
    prev_rew = rew_df.iloc[-2]
    a1.metric("Total Reward", f"{last_rew['total']:.3f}",
              f"{round(last_rew['total']-prev_rew['total'],3):+.3f}")
    a2.metric("Episodes", f"{int(rew_df['episode'].max())}")
    a3.metric("Total Violations", f"{int(rew_df['violations'].sum())}")
    trend = "📈 Improving" if rew_df["total"].iloc[-10:].mean() > rew_df["total"].iloc[:10].mean() else "📉 Needs tuning"
    a4.metric("Convergence", trend)
    st.caption("🔵 Demo data — upload a rich telemetry CSV for real reward tracking")
else:
    a1.metric("Total Reward", "—")
    a2.metric("Episodes", "—")
    a3.metric("Total Violations", "—")
    a4.metric("Convergence", "—")

ac1, ac2 = st.columns(2)
with ac1:
    st.markdown("**Reward convergence over episodes**")
    if using_real_rewards and "best_reward" in rew_df.columns:
        st.plotly_chart(
            area(rew_df, "episode", "best_reward", "rgb(127,119,221)", y_label="Cumulative reward"),
            use_container_width=True)
    elif "total" in rew_df.columns:
        st.plotly_chart(
            area(rew_df, "episode", "total", "rgb(127,119,221)", y_label="Total reward"),
            use_container_width=True)

with ac2:
    if using_real_rewards and "reward" in rew_df.columns:
        st.markdown("**Per-episode reward + smoothed**")
        fig_sm = go.Figure()
        fig_sm.add_trace(go.Scatter(
            x=rew_df["episode"], y=rew_df["reward"],
            mode="markers", name="Raw reward",
            marker=dict(size=4, color="rgba(55,138,221,0.4)"),
            hovertemplate="Ep %{x}: %{y:.1f}<extra></extra>"
        ))
        fig_sm.add_trace(go.Scatter(
            x=rew_df["episode"], y=rew_df["reward_smooth"],
            mode="lines", name="Smoothed (10-ep window)",
            line=dict(color="#378ADD", width=2.5)
        ))
        fig_sm.update_layout(**CHART_LAYOUT, height=CHART_H, yaxis_title="Reward")
        st.plotly_chart(fig_sm, use_container_width=True)
    elif "r_task" in rew_df.columns:
        st.markdown("**Reward components — latest episode**")
        st.plotly_chart(
            bar_components(last_rew["r_task"], last_rew["r_smooth"], last_rew["r_safety"]),
            use_container_width=True)

ac3, ac4 = st.columns(2)
with ac3:
    if using_real_rewards and "epsilon" in rew_df.columns:
        st.markdown("**Epsilon decay (exploration rate)**")
        st.plotly_chart(
            area(rew_df, "episode", "epsilon", "rgb(55,138,221)", y_label="epsilon"),
            use_container_width=True)
    elif "r_task" in rew_df.columns:
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

if using_real_rewards:
    st.info("Reward weights → w1={w1} · w2={w2} · w3={w3} — adjust in sidebar to experiment.".format(
        w1=w1, w2=w2, w3=w3))
else:
    st.info(f"Reward weights → w1={w1} · w2={w2} · w3={w3} — adjust in sidebar to experiment.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FLIGHT LOG TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📋 Flight Log</div>', unsafe_allow_html=True)

display_cols = [c for c in ["t_sec","alt","spd","vspd","roll","pitch","hdg",
                              "g","fuel","epow","stall","gear"] if c in fdf.columns]
flight_log_df = fdf[display_cols].tail(show_n).copy()
flight_log_df.columns = [c.replace("t_sec","Time(s)").replace("alt","Alt(m)")
                   .replace("spd","Spd(m/s)").replace("vspd","VSp(m/s)")
                   .replace("roll","Roll(°)").replace("pitch","Pitch(°)")
                   .replace("hdg","Hdg(°)").replace("g","G-Force")
                   .replace("fuel","Fuel").replace("epow","EngPow")
                   .replace("stall","Stall").replace("gear","Gear")
                  for c in flight_log_df.columns]

st.dataframe(flight_log_df.round(2), use_container_width=True, hide_index=True)

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

# ── build experiment table from real training logs if available ────────────
if using_real_log and not log_df.empty:
    # Each training log = one experiment row
    log_rows = []
    for fname, ldf in st.session_state.log_history.items():
        if ldf.empty:
            continue
        seed_match = re.search(r"seed[_-]?(\d+)", fname, re.IGNORECASE)
        seed_label = f"seed {seed_match.group(1)}" if seed_match else fname.replace(".txt", "")
        log_rows.append({
            "Experiment": seed_label,
            "Best Reward": round(float(ldf["reward"].max()), 1),
            "Episodes": int(len(ldf)),
            "Final Epsilon": round(float(ldf["epsilon"].iloc[-1]), 4) if "epsilon" in ldf.columns else 0.0,
            "Landings": int(ldf.attrs.get("landing_count", 0)),
            "Crashes": int(ldf.attrs.get("crash_count", 0)),
            "Notes": f"Parsed from {fname}",
        })
    default_experiments = pd.DataFrame(log_rows)
else:
    default_experiments = pd.DataFrame([
        {"Experiment": "Baseline", "Best Reward": 4055.5, "Episodes": 289,
         "Final Epsilon": 0.028, "Landings": 5, "Crashes": 0,
         "Notes": "Original config — best result"},
        {"Experiment": "Exp 1", "Best Reward": 2791.5, "Episodes": 200,
         "Final Epsilon": 0.05, "Landings": 3, "Crashes": 2,
         "Notes": "Harsher stall penalty hurt training"},
        {"Experiment": "Exp 2", "Best Reward": 806.3, "Episodes": 150,
         "Final Epsilon": 0.1, "Landings": 1, "Crashes": 5,
         "Notes": "Higher LR broke convergence"},
    ])

# ── editable experiment table ─────────────────────────────────────────────────
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

format_dict = {"Best Reward": "{:.1f}"}
if "Final Epsilon" in exp_df.columns:
    format_dict["Final Epsilon"] = "{:.4f}"

st.dataframe(
    exp_df.style.apply(highlight_best, axis=1).format(format_dict),
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
    if best_val > 0:
        drop_pct = round((1 - worst_val / best_val) * 100, 1)
    else:
        drop_pct = 0

    if using_real_log:
        st.markdown(f"""
**Best run:** {best_exp} → reward **{best_val:.1f}**

**Worst run:** reward **{worst_val:.1f}** ({drop_pct}% drop from best)

**Experiments compared:** {len(exp_df)} training runs parsed from `log_seed_*.txt`

**Key insight:** The agent learns to fly via DQN reinforcement learning. Higher
reward = better flight performance (longer flights, more landings, fewer crashes).
Epsilon decay shows exploration → exploitation transition.
        """)
    else:
        st.markdown(f"""
**Best run:** {best_exp} → reward **{best_val:.1f}**

**Worst run:** reward **{worst_val:.1f}** ({drop_pct}% drop from best)

Upload training logs (`log_seed_*.txt`) to auto-populate this table with real data.
        """)
    # download experiment table
    exp_csv = exp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download experiment table as CSV",
        exp_csv,
        file_name="experiment_comparison.csv",
        mime="text/csv"
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TRAINING LOG VISUALISATION  (from log_seed_*.txt)
# ══════════════════════════════════════════════════════════════════════════════

if using_real_log and not log_df.empty:
    st.markdown('<div class="section-header">📚 Training Log Analysis</div>',
                unsafe_allow_html=True)

    tl1, tl2, tl3, tl4, tl5 = st.columns(5)
    tl1.metric("Total Episodes", f"{log_df.attrs.get('total_episodes', len(log_df))}")
    tl2.metric("Best Reward",    f"{log_df['reward'].max():.1f}")
    tl3.metric("Landings",       f"{log_df.attrs.get('landing_count', 0)}")
    tl4.metric("Crashes",        f"{log_df.attrs.get('crash_count', 0)}")
    tl5.metric("Weight Saves",   f"{log_df.attrs.get('weight_snapshots', 0)}")

    tlc1, tlc2 = st.columns(2)
    with tlc1:
        st.markdown("**Episode reward + smoothed**")
        fig_tlr = go.Figure()
        fig_tlr.add_trace(go.Scatter(
            x=log_df["episode"], y=log_df["reward"],
            mode="markers", name="Raw reward",
            marker=dict(size=4, color="rgba(127,119,221,0.4)"),
            hovertemplate="Ep %{x}: %{y:.1f}<extra></extra>"
        ))
        fig_tlr.add_trace(go.Scatter(
            x=log_df["episode"], y=log_df["reward_smooth"],
            mode="lines", name="Smoothed (10-ep window)",
            line=dict(color="#7F77DD", width=2.5)
        ))
        fig_tlr.update_layout(**CHART_LAYOUT, height=300, yaxis_title="Reward")
        st.plotly_chart(fig_tlr, use_container_width=True)
    with tlc2:
        st.markdown("**Epsilon decay (exploration rate)**")
        st.plotly_chart(
            area(log_df, "episode", "epsilon", "rgb(55,138,221)", y_label="epsilon"),
            use_container_width=True)

    tlc3, tlc4 = st.columns(2)
    with tlc3:
        st.markdown("**Steps per episode**")
        st.plotly_chart(
            line(log_df, "episode", ["steps"], ["#1D9E75"],
                 ["Steps"], y_label="steps"),
            use_container_width=True)
    with tlc4:
        st.markdown("**Best reward over time**")
        st.plotly_chart(
            line(log_df, "episode", ["best_reward"], ["#BA7517"],
                 ["Best Reward"], y_label="reward"),
            use_container_width=True)

    # download log data
    log_csv = log_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download parsed training log as CSV", log_csv,
                       file_name="training_log_parsed.csv", mime="text/csv")

    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RICH TELEMETRY CSV (DQN state/action/reward per timestep)
# ══════════════════════════════════════════════════════════════════════════════

if using_real_rich and not rcdf.empty:
    st.markdown('<div class="section-header">🧠 Rich Telemetry — DQN State & Action</div>',
                unsafe_allow_html=True)

    # metrics row
    rc_m1, rc_m2, rc_m3, rc_m4 = st.columns(4)
    rc_m1.metric("Total Rows",    f"{len(rcdf)}")
    rc_m2.metric("Unique Episodes", f"{rcdf['episode'].nunique()}" if "episode" in rcdf.columns else "—")
    if "reward" in rcdf.columns:
        rc_m3.metric("Episode Reward", f"{rcdf.groupby('episode')['reward'].last().sum():.1f}")
    if "epsilon" in rcdf.columns:
        rc_m4.metric("Final Epsilon",  f"{rcdf['epsilon'].iloc[-1]:.4f}")

    # flight charts (reuse existing helpers for the renamed columns)
    rcc1, rcc2 = st.columns(2)
    with rcc1:
        if "alt" in rcdf.columns:
            st.markdown("**Altitude & Vertical Speed**")
            st.plotly_chart(
                line(rcdf.tail(show_n), "t_sec", ["alt", "vspd"],
                     ["#378ADD", "#1D9E75"],
                     ["Altitude (m)", "VSpd (m/s)"], y_label="m / m·s⁻¹"),
                use_container_width=True)
    with rcc2:
        if "spd" in rcdf.columns:
            st.markdown("**Airspeed & G-Force**")
            st.plotly_chart(
                line(rcdf.tail(show_n), "t_sec", ["spd", "g"],
                     ["#BA7517", "#E24B4A"],
                     ["Airspeed (m/s)", "G-Force"], y_label="m/s / g"),
                use_container_width=True)

    # DQN-specific charts
    if "action" in rcdf.columns:
        rcc3, rcc4 = st.columns(2)
        with rcc3:
            st.markdown("**DQN Action over time**")
            fig_act = go.Figure(go.Scatter(
                x=rcdf["t_sec"], y=rcdf["action"],
                mode="lines+markers", name="Action",
                marker=dict(size=3), line=dict(color="#D4537E", width=1.5),
                hovertemplate="t=%{x:.1f}s  action=%{y}<extra></extra>"
            ))
            fig_act.update_layout(**CHART_LAYOUT, height=300, yaxis_title="Action index")
            st.plotly_chart(fig_act, use_container_width=True)
        with rcc4:
            if "reward" in rcdf.columns:
                st.markdown("**Cumulative reward**")
                rcdf_view = rcdf.tail(show_n).copy()
                rcdf_view["cum_reward"] = rcdf_view["reward"].cumsum()
                st.plotly_chart(
                    area(rcdf_view, "t_sec", "cum_reward", "rgb(29,158,117)",
                         y_label="cumulative reward"),
                    use_container_width=True)

    # Q-value heatmap
    q_cols = [c for c in rcdf.columns if c.startswith("Q_")]
    if q_cols:
        st.markdown("**Q-values over time (heatmap)**")
        q_df = rcdf[q_cols].tail(show_n).copy()
        q_df.index = rcdf["t_sec"].tail(show_n).values
        fig_heat = go.Figure(go.Heatmap(
            z=q_df.values.T,
            x=[f"{v:.1f}" for v in q_df.index],
            y=q_cols,
            colorscale="Viridis",
            hovertemplate="t=%{x}s  Q=%{y}  val=%{z:.2f}<extra></extra>"
        ))
        fig_heat.update_layout(**CHART_LAYOUT, height=250, yaxis_title="Q-output")
        st.plotly_chart(fig_heat, use_container_width=True)

    # state vector heatmap
    state_cols = [c for c in rcdf.columns if c.startswith("state_")]
    if state_cols:
        st.markdown("**DQN Input State Vector (heatmap)**")
        s_df = rcdf[state_cols].tail(min(200, show_n)).copy()
        s_df.index = rcdf["t_sec"].tail(min(200, show_n)).values
        fig_state = go.Figure(go.Heatmap(
            z=s_df.values.T,
            x=[f"{v:.1f}" for v in s_df.index],
            y=state_cols,
            colorscale="RdBu_r",
            hovertemplate="t=%{x}s  s=%{y}  val=%{z:.3f}<extra></extra>"
        ))
        fig_state.update_layout(**CHART_LAYOUT, height=300, yaxis_title="State dim")
        st.plotly_chart(fig_state, use_container_width=True)

    st.download_button("⬇️ Download rich telemetry as CSV",
                       rcdf.to_csv(index=False).encode("utf-8"),
                       file_name="rich_telemetry_export.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH — re-read from disk on interval
# ══════════════════════════════════════════════════════════════════════════════

if (source == "From disk path"
        and disk_refresh
        and disk_path
        and st.session_state.get("_disk_interval")):
    time.sleep(st.session_state._disk_interval)
    st.rerun()