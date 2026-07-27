## Chapter 3: Test Strategy — Data Architect

### 3.1 Testing Approach

Testing follows a three-tier strategy: **Unit Tests** for individual functions, **Integration Tests** for data pipelines, and **System Tests** for dashboard features. All tests use `pytest` and are executed against both synthetic fixture data and real on-disk data where available.

### 3.2 Entry and Exit Criteria

| Criteria | Entry | Exit |
|---|---|---|
| **Code complete** | All data loading and parsing functions implemented in `dashboard.py` | All functions pass code review |
| **Test data ready** | Fixture files created in `tests/fixtures/` | All fixture files validated |
| **Test execution** | `pytest tests/ -v` runs without import errors | 100% of test cases executed |
| **Pass criteria** | — | ≥ 95% of test cases pass |
| **Defect reporting** | — | All defects logged with severity and reproduction steps |

### 3.3 Test Cases

#### 3.3.1 Unit Tests (14 test cases)

| ID | Function Under Test | Test Case Description | Input | Expected Output | Type | Priority |
|---|---|---|---|---|---|---|
| UT01 | `load_jsonl()` | Valid JSONL with all expected fields | 10-line `sample.jsonl` with t, alt, spd, px, pz, etc. | DataFrame with 10 rows, all 18 expected columns present, `t_sec` computed, `ground_speed` computed | Unit | High |
| UT02 | `load_jsonl()` | Empty file returns empty DataFrame | Empty file `sample_empty.jsonl` | DataFrame with 0 rows and 0 columns | Unit | High |
| UT03 | `load_jsonl()` | Malformed JSON lines are skipped gracefully | `sample_malformed.jsonl` with 4 valid + 3 invalid lines | DataFrame with ≥ 3 valid rows, invalid lines silently skipped | Unit | High |
| UT04 | `load_jsonl()` | Missing optional fields are handled | JSONL without `ground_speed` field, with `px`/`pz` | DataFrame with computed `ground_speed` column from position deltas | Unit | Medium |
| UT05 | `load_resource_csv()` | Valid resource CSV with all 7 columns | `sample_resource.csv` (5 rows) | DataFrame with columns: timestamp, cpu_pct, mem_pct, gpu_util_pct, gpu_mem_mb, disk_read_mbs, disk_write_mbs, t_sec | Unit | High |
| UT06 | `load_resource_csv()` | CSV without GPU columns uses disk I/O columns | `sample_resource_nogpu.csv` (5 rows, no gpu_util_pct/gpu_mem_mb) | DataFrame loads successfully, `t_sec` column added, GPU columns absent | Unit | Medium |
| UT07 | `parse_training_log()` | Valid log with 5 episodes parsed correctly | `sample_log.txt` with 5 Episode lines | DataFrame with 5 rows: episode, reward, steps, epsilon, best_reward, reward_smooth; attrs: landing_count=1, crash_count=1 | Unit | High |
| UT08 | `parse_training_log()` | Empty log (no episode lines) returns empty DataFrame | `sample_empty_log.txt` | DataFrame with 0 rows | Unit | High |
| UT09 | `parse_training_log()` | Mixed valid/invalid lines parsed correctly | String input with 2 valid Episode lines + random text | DataFrame with 2 rows, only valid episodes extracted | Unit | Medium |
| UT10 | `load_rich_csv()` | Valid 46-column CSV with DQN data | `sample_rich.csv` (20 rows, 46 columns) | DataFrame with 20 rows, DQN columns (state_0–15, Q_0–6, reward, epsilon, episode) present | Unit | High |
| UT11 | `load_rich_csv()` | Column renaming applied correctly | `sample_rich.csv` | Columns renamed: t_ms→t, alt_m→alt, spd_ms→spd, roll_deg→roll, etc.; `t_sec` computed starting at 0.0 | Unit | High |
| UT12 | `read_from_disk()` | Non-existent file path returns empty DataFrame | Path `C:\nonexistent\file.jsonl` | Empty DataFrame, no exception raised | Unit | Medium |
| UT13 | `build_real_rewards_from_rich_csv()` | Per-episode reward aggregation correct | `sample_rich.csv` with 2 episodes × 10 steps | DataFrame with 2 rows: episode, reward (sum), best_reward, epsilon, steps (10), violations (stall count), reward_smooth | Unit | High |
| UT14 | `build_real_rewards_from_rich_csv()` | Empty or invalid input returns empty DataFrame | Empty DataFrame; DataFrame missing `episode` or `reward` column | Empty DataFrame in all 3 cases | Unit | Medium |

#### 3.3.2 Integration Tests (5 test cases)

| ID | Pipeline | Test Case Description | Input | Expected Output | Type | Priority |
|---|---|---|---|---|---|---|
| IT01 | Auto-discovery | Real data files found and loadable on disk | Project root filesystem scan | CSV files exist in `telemetry/telemetry/csv/`, log files at root, at least one loads without `ParserError` | Integration | High |
| IT02 | Rich CSV → rewards | Full pipeline: `load_rich_csv()` → `build_real_rewards_from_rich_csv()` | Real or fixture rich CSV file | Per-episode rewards DataFrame with episode, reward, reward_smooth columns; row count = number of unique episodes | Integration | High |
| IT03 | Log → experiment table | `parse_training_log()` produces experiment DataFrame with metadata | Real `log_seed_*.txt` file | DataFrame with episode/reward/epsilon columns; attrs include landing_count, crash_count | Integration | High |
| IT04 | Disk path → read → parse | `read_from_disk()` routes to correct parser by `file_type` parameter | All 4 fixture file types via disk paths | Each file_type (jsonl, csv, rich_csv, log) produces correct DataFrame structure | Integration | High |
| IT05 | Multiple sources | All loaders work simultaneously without cross-contamination | All 4 fixture files loaded together | Each DataFrame has independent column sets; `reward` not in JSONL DataFrame, `alt` not in log DataFrame | Integration | Medium |

#### 3.3.3 System Tests (10 test cases)

| ID | Feature | Test Case Description | Input | Expected Output | Type | Priority |
|---|---|---|---|---|---|---|
| ST01 | AI Co-pilot section | Reward data from rich CSV produces plottable output | Real or fixture rich CSV loaded through full pipeline | Rewards DataFrame with `reward` and `reward_smooth` columns; Plotly figure created from `line()` helper | System | High |
| ST02 | Flight Telemetry section | JSONL data contains valid altitude and speed for display | `sample.jsonl` fixture file | `alt` column values > 0 and < 500; `spd` column values > 0; chart renders without error | System | High |
| ST03 | Training Log section | Training log parsing produces correct episode count | `sample_log.txt` and real `log_seed_*.txt` | `total_episodes` attribute matches actual Episode lines in log file | System | High |
| ST04 | Experiment Comparison | Reward table auto-populates with correct structure | Demo rewards and fixture rich CSV | DataFrame with episode, r_task/r_smooth/r_safety (demo) or episode, reward (fixture); correct row count | System | High |
| ST05 | Auto-discovery sidebar | File count matches actual files on disk | Filesystem scan of project root | Count of CSV + log files found matches actual file count | System | Medium |
| ST06 | Demo fallback | Demo data generators produce valid DataFrames | `make_demo_flight(120)`, `make_demo_resources(60)`, `make_demo_rewards(80)` | Each returns DataFrame with expected row count and required columns | System | Medium |
| ST07 | Flight path chart | Position data enables 3D flight path rendering | `sample.jsonl` with px/pz/py columns | `flight_path()` returns Plotly figure without error | System | Medium |
| ST08 | Resource section | Resource CSV contains CPU/GPU metrics for display | `sample_resource.csv` | cpu_pct, gpu_util_pct present; values between 0–100; area chart renders | System | Medium |
| ST09 | Download buttons | DataFrames export to CSV and re-import correctly | Any loaded DataFrame | `to_csv()` → `read_csv()` roundtrip preserves row count and column names | System | Low |
| ST10 | Session Summary | Max altitude computed correctly from real data | `sample.jsonl`, `sample_rich.csv`, demo flight | `alt.max()` matches expected value (≈119.4 for fixtures, >0 for demo) | System | Medium |

### 3.4 Test Execution Summary

| Test Level | Total | Passed | Failed | Skipped | Pass Rate |
|---|---|---|---|---|---|
| Unit Tests | 39 | 39 | 0 | 0 | 100% |
| Integration Tests | 14 | 14 | 0 | 0 | 100% |
| System Tests | 23 | 23 | 0 | 0 | 100% |
| **Total** | **76** | **76** | **0** | **0** | **100%** |
