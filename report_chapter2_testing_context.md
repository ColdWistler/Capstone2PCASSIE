## Chapter 2: Testing Context — Data Architect

### 2.1 Role Description

The Data Architect is responsible for the complete data pipeline within the UAV Telemetry Dashboard, covering data ingestion, transformation, validation, and display. This role owns `dashboard.py` — a Streamlit-based web application that visualizes UAV flight telemetry, DQN (Deep Q-Network) training metrics, and system resource utilization.

### 2.2 Scope of Testing

| Area | Functions | Description |
|---|---|---|
| **Data Loading** | `load_jsonl()`, `load_resource_csv()`, `load_rich_csv()` | Parse telemetry JSONL, resource CSV, and rich DQN CSV files into DataFrames |
| **Data Parsing** | `parse_training_log()` | Extract per-episode metrics from `log_seed_*.txt` training logs via regex |
| **Data Transformation** | `build_real_rewards_from_rich_csv()`, `read_from_disk()` | Aggregate raw CSV rows into per-episode reward summaries; route disk reads by file type |
| **File Discovery** | `auto_discover_files()` | Scan disk directories for CSVs, logs, and resource files at startup |
| **Demo Data Generation** | `make_demo_flight()`, `make_demo_resources()`, `make_demo_rewards()` | Produce synthetic data when no real files exist |
| **Chart Rendering** | `line()`, `area()`, `bar_components()`, `flight_path()` | Create Plotly figures for telemetry, resources, rewards, and flight paths |
| **Dashboard Sections** | Section 1–6 in `main()` | Flight Telemetry, System Resources, AI Co-pilot, Training Log, Experiment Comparison, Session Summary |

### 2.3 System Under Test

| Component | Technology | Version |
|---|---|---|
| Dashboard framework | Streamlit | 1.x |
| Charting library | Plotly | 5.x |
| Data processing | Pandas | 2.x |
| Numerical computation | NumPy | 2.x |
| Backend language | Python | 3.13.3 |
| Telemetry source | Godot 4.7 + Rust GDExtension | — |
| DQN model | Dueling DQN (142,088 params) | state_dim=12, action_dim=7 |

### 2.4 Test Environment

| Item | Detail |
|---|---|
| OS | Windows 10/11 (win32) |
| Python | 3.13.3 |
| Test framework | pytest 9.1.1 |
| Test location | `tests/` directory |
| Fixture data | `tests/fixtures/` (8 synthetic files) |
| Real data | `telemetry/telemetry/csv/`, `log_seed_*.txt`, `resources_*.csv` |

### 2.5 Data Formats

| Format | Source | Key Fields |
|---|---|---|
| JSONL | Godot `TelemetryExporter.gd` | t, alt, spd, vspd, roll, pitch, hdg, px, py, pz, g, load, fuel, epow, flap, gear, stall, eact |
| Rich CSV | Godot `TelemetryCsvExporter.gd` | 46 columns: t_ms, alt_m, spd_ms, state_0–15, Q_0–6, reward, epsilon, episode, step_count |
| Training log | `train_dqn.py` (Rust DQN) | `Episode N \| reward: R \| steps: S \| eps: E \| best: B` |
| Resource CSV | `resource_monitor.py` | timestamp, cpu_pct, mem_pct, gpu_util_pct, gpu_mem_mb, disk_read_mbs, disk_write_mbs |

### 2.6 Identified Defects (Pre-existing)

| ID | Severity | Description |
|---|---|---|
| DEF-001 | Medium | Real telemetry CSV files have inconsistent column counts (Expected 47, saw 49 at line 1039). The `TelemetryCsvExporter.gd` occasionally appends extra columns during DQN step recording. `load_rich_csv()` uses `pd.read_csv()` without `on_bad_lines` handling, causing `ParserError`. |
| DEF-002 | Low | Data paths are nested: files are at `telemetry/telemetry/csv/` rather than `telemetry/csv/`. The `auto_discover_files()` function already handles both paths, but the inconsistency suggests a Godot export path misconfiguration. |
